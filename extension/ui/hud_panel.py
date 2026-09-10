"""#115 手動擊球參數控制面板：疊在 viewport 內的半透明 overlay。

沿用 `debug_menu.py` 的風格：建構子只接受注入的 Callable（不直接摸
`BilliardExtension`），每 frame 輪詢更新狀態列，`destroy()` 先清 subscription
再清 widget。與 debug_menu 的差異：debug_menu 是獨立停靠視窗，本面板是
`viewport_window.get_frame(ext_id)` 疊出來的 overlay（使用者已拍板的視覺
需求）。

面板本身**不保存參數狀態**——擊球參數只活在 omni.ui model（力道欄）與常駐
`ManualController`（`get_parameters`/`on_parameters_changed` 兩端）裡。面板
唯一的私有狀態是拖曳暫態（`self._topview_drag_mode`）與防遞迴旗標
（`self._suppress_speed_callback`）。

⚠️ **本檔幾處明確隔離了 spike 未證實的假設**，實測後若需要調整，只需要改
對應的那一個方法，其餘部分不動：

1. `_create_root_frame()` — overlay 容器怎麼拿到（`get_frame()` vs 退回
   獨立 `ui.Window`）。
2. `_to_local()` — `set_mouse_*_fn` 收到的座標怎麼換算成 widget local 座標。
3. `_on_panel_wheel()` — 面板背板與兩個互動畫布掛了滾輪 no-op handler，
   目的是擋掉滾輪事件穿透到底下的 viewport 相機縮放（2026-09-08 實際使用
   時回報過這個症狀），但「掛了 `set_mouse_wheel_fn()` 是否真的能擋住
   穿透」跟前兩項一樣是未證實的假設，見該方法 docstring；且目前只覆蓋
   三個特定 widget，面板裡 Label／Button／ComboBox／FloatField／列間距／
   外圍 margin 這些空隙位置有沒有殘留穿透，也還沒實測。

三者的細節與可信度標示見
`docs/tech-design/hud-shot-control-panel-tech-design.md` 第 1 節。

其餘實作細節（`Placer.offset_x` 是否接受 `ui.Pixel`、`FloatField`/
`SimpleFloatModel` 的讀寫方法、overlay 上的滑鼠事件會不會被 viewport 相機
操作吃掉……）沒有被抽成獨立方法，因為 tech-design 已在文件層級對每一項給
出可信度標示與備援方案；改動這些假設通常只影響單一個 widget 的建構程式碼，
不需要額外的間接層。實作時的判斷與理由見各私有方法的 docstring。
"""

import dataclasses
from typing import Callable

import carb.settings
import omni.kit.app
import omni.ui as ui
from omni.ui import color as cl

from .table_combo_box_model import TableComboBoxModel

from core.models.action_bounds import CUE_BALL_SPEED
from core.models.manual_shot_parameters import ManualShotParameters
from core.services import shot_panel_input_mapper as mapper
from core.services.break_shot_position_provider import BREAK_SHOT_POSITIONS
from core.services.manual_shot_feasibility import (
    ManualShotFeasibility,
    evaluate_manual_shot_parameters,
)
from core.services.pocket_geometry import POCKET_POSITIONS

# 面板本體尺寸（含背板）：viewport 寬的 1/3、高的 1/2，用 ui.Percent 相對
# self._root_frame（= viewport 的實際渲染尺寸）算，不寫死像素值——viewport
# 窗格多大都會自動跟著縮放，不會固定佔用某個絕對大小。用 Placer 定位到
# 右下角，理由見計畫「HUD 嵌在 Viewport 內」一節與 _build_ui() 內的說明。
_PANEL_WIDTH_PERCENT = 100.0 / 3.0
_PANEL_HEIGHT_PERCENT = 50.0

# 圓形擊球點選擇器：120x120，半徑 60px 填滿整個方框，標記半徑 12px。數值全部
# 寫死常數，不依賴 computed_width（第一 frame 可能是 0，見 tech-design 1.5）。
_CIRCLE_SIZE_PX = 120.0
_CIRCLE_CENTER_PX = 60.0
_CIRCLE_RADIUS_PX = 60.0
_CIRCLE_MARKER_RADIUS_PX = 6.0
# 圓形選擇器開放全物理範圍（±0.5R），不做額外收窄——面板端的「可用偏移能力
# 比例」語意見 shot_panel_input_mapper.offset_from_circle_pixels() docstring。
_CIRCLE_MAX_OFFSET = 1.0

# 俯瞰圖：128x256，128/1.27 == 256/2.54 == 100.79 px/m，兩軸同尺度球才不會
# 被畫成橢圓（見計畫「版面」一節）。
_TOPVIEW_WIDTH_PX = 128.0
_TOPVIEW_HEIGHT_PX = 256.0
_CUE_BALL_MARKER_RADIUS_PX = 4.0
# 命中判定半徑刻意大於視覺半徑（球心對應像素半徑約 2.9px 太小難點擊）——這是
# UX 上的選擇，不是物理量，跟 evaluate_manual_shot 用的 ball_radius 無關。
_CUE_BALL_HIT_RADIUS_PX = 2.0
_RACK_BALL_MARKER_RADIUS_PX = 4.0
_POCKET_MARKER_RADIUS_PX = 4.0
_AIM_DOT_COUNT = 20
_AIM_DOT_RADIUS_PX = 2.0

# 滑鼠懸停在面板上時暫時停用的相機手勢——用來擋掉滾輪穿透到底下 viewport
# 觸發相機縮放。2026-09-08 實測發現掛在 widget 上的 set_mouse_wheel_fn()
# no-op handler 完全沒有擋住穿透，改用這個機制，見 _on_panel_hovered()
# docstring 的完整推導。
#
# 官方文件確認的設定路徑與預設值（camera_manipulator.html）：
#   /exts/omni.kit.viewport.window/bindings/camera
#   -> {'PanGesture': 'Any MiddleButton', 'TumbleGesture': 'Alt LeftButton',
#       'ZoomGesture': 'Alt RightButton', 'LookGesture': 'RightButton',
#       'ZoomScrollGesture': 'Any', 'FlightSpeedGesture': 'RightButton',
#       'FlightMode': 'RightButton'}
# 只停用 ZoomScrollGesture（滾輪縮放）——這是使用者實際回報的症狀。其餘
# 手勢（TumbleGesture/ZoomGesture 需要 Alt、LookGesture 是右鍵）本面板沒有
# 用到對應的滑鼠按鍵，暫時沒有回報衝突，但理論上同一個穿透機制可能一樣
# 影響它們，尚未實測——見 _on_panel_hovered() docstring。
_CAMERA_BINDINGS_SETTING_PATH = "/exts/omni.kit.viewport.window/bindings/camera"
_DISABLED_CAMERA_GESTURES = ("ZoomScrollGesture",)

# 診斷用旗標：True 時只畫純背板、不建任何互動元件，方便排除「是不是內容
# 本身造成穿透/溢出」的問題，跟正式內容分開驗證。除錯歷程見
# docs/CHANGELOG.md「面板定位方式的除錯歷程」。正式版面固定用 False。
_DIAGNOSTIC_BACKGROUND_ONLY = False

# 原本 cl(0.08, 0.08, 0.10, 0.55)（RGB≈20,20,26）跟 ui.Rectangle 沒套上
# style 時的內建預設色 RGB(41,41,41) 肉眼幾乎分不出來，半透明失效跟「style
# 根本沒套上」兩種情況外觀一樣分不清。刻意選明顯偏藍、遠離中性灰的色調——
# 如果畫面上看到藍色調就代表 style 有套上（alpha 是否正確混色再另外判斷），
# 如果看到的是純中性灰 (41,41,41) 才代表 style 沒套上，兩種情況一眼可辨。
_PANEL_BACKGROUND_COLOR = cl(0.05, 0.08, 0.16, 0.55)
_AIM_LINE_COLOR = cl(0.95, 0.85, 0.2, 0.9)
_AIM_LINE_INFEASIBLE_COLOR = cl(0.95, 0.2, 0.2, 0.95)
_FEASIBILITY_TEXT_COLOR = cl(0.95, 0.35, 0.3, 1.0)


class HudPanel:
    """viewport overlay 形式的手動擊球參數控制面板。

    建構子只吃 Callable（見各參數說明），不持有 `BilliardExtension` 的參照，
    也不保存 `ManualShotParameters`——值的單一事實來源是常駐 `ManualController`
    （透過 `get_parameters`/`on_parameters_changed` 讀寫）。
    """

    def __init__(
        self,
        ext_id: str,
        get_parameters: Callable[[str], "ManualShotParameters | None"],
        on_parameters_changed: Callable[[str, ManualShotParameters], None],
        on_shot_requested: Callable[[str], None],
        on_reset_requested: Callable[[str], None],
        get_shot_status_text: Callable[[str], str],
        get_table_geometry: Callable[[str], "tuple[float, float] | None"],
        get_controller_mode_text: Callable[[str], str],
        on_toggle_controller_mode: Callable[[str], None],
        confirm_reset: Callable[[str], None],
        is_ready_to_reset: Callable[[str], bool],
        get_shot_result_text: Callable[[str], str],
    ) -> None:
        self._ext_id = ext_id
        self._get_parameters = get_parameters
        self._on_parameters_changed = on_parameters_changed
        self._on_shot_requested = on_shot_requested
        self._on_reset_requested = on_reset_requested
        self._get_shot_status_text = get_shot_status_text
        self._get_table_geometry = get_table_geometry
        self._get_controller_mode_text = get_controller_mode_text
        self._on_toggle_controller_mode = on_toggle_controller_mode
        self._confirm_reset = confirm_reset
        self._is_ready_to_reset = is_ready_to_reset
        self._get_shot_result_text = get_shot_result_text

        # 這些屬性無論 root frame 拿不拿得到都要先設好，destroy() 才能在
        # headless（root frame 為 None）下安全呼叫。
        self._update_sub = None
        self._table_combo_model: TableComboBoxModel | None = None
        # 2026-09-08 實測回報：面板展開時的版面總高度（粗估 ~626px：120 圓形
        # 選擇器 + 256 俯瞰圖 + 4 個 24px 列 + 6 行狀態文字 + margin/spacing）
        # 超出使用者當時的 viewport 窗格高度，視覺上溢出邊界。預設改成收合，
        # 只留 24px 的標題列，使用者需要調整參數時自己按 ▼ 展開——不是真正
        # 解決「面板可能比 viewport 還高」這件事（那需要知道實際 viewport
        # 尺寸才能對症處理，例如改用可捲動區塊或縮小控制項），只是先把「一
        # 開啟就溢出」這個立即症狀壓下去。
        self._is_collapsed = True
        # 滑鼠懸停在面板上時暫存的相機手勢設定（懸停期間停用 ZoomScrollGesture，
        # 離開時還原），None 代表目前沒有暫停任何東西。見 _on_panel_hovered()。
        self._saved_camera_bindings: dict | None = None
        self._topview_drag_mode: str | None = None  # "placement" | "angle" | None
        self._suppress_speed_callback = False
        self._last_selected_table_id: str | None = None

        self._root_frame = self._create_root_frame()
        if self._root_frame is None:
            # headless／拿不到 viewport：安靜跳過建面板，這是硬性要求（見
            # _create_root_frame() docstring），不能讓 scripts/ 底下的 headless
            # 驗證腳本因為建這個面板而炸掉。
            return

        self._table_combo_model = TableComboBoxModel()
        self._build_ui()
        self._refresh_controls_for_selected_table()
        self._update_sub = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._on_update, name="billiard_hud_panel_refresh")
        )

    # ------------------------------------------------------------------
    # 兩個隔離未知的方法
    # ------------------------------------------------------------------

    def _create_root_frame(self):
        """取得 viewport overlay 用的 `omni.ui.Frame` 容器。

        ⚠️ **此處依賴 spike 未證實的假設，實測後可能需要調整**，見
        `docs/tech-design/hud-shot-control-panel-tech-design.md` 第 1.1／1.2
        節：`get_frame()` 放一般 2D widget（而非 `sc.SceneView`）能不能正常
        建構並收到滑鼠事件、overlay 上的滑鼠拖曳會不會被 viewport 相機操作
        吃掉，兩者都只有文件層級的查證，尚未在 GUI 下實測。若實測發現
        overlay 這條路行不通，備援是退回獨立 `ui.Window`——只需要改這個方法
        的實作內容（回傳一個 `window.frame`），`HudPanel` 其餘部分（版面、
        互動、資料流）完全不動。

        **硬性要求**：headless 或 viewport 尚未建立時 `get_active_viewport_
        window()` 回傳 `None`；`omni.kit.viewport.utility` 這個模組本身若
        未被正確載入也可能 import 失敗。兩種情況都回傳 `None`，讓呼叫端
        （`__init__`）安靜跳過建面板，不拋例外——`scripts/` 底下所有 headless
        驗證腳本都會在 extension 啟動時經過這條路徑，不能因為建 HudPanel 而
        炸掉。
        """
        try:
            from omni.kit.viewport.utility import get_active_viewport_window
        except ImportError:
            return None

        viewport_window = get_active_viewport_window()
        if viewport_window is None:
            return None
        return viewport_window.get_frame(self._ext_id)

    def _to_local(self, x: float, y: float, widget: ui.Widget) -> tuple[float, float]:
        """把 `set_mouse_*_fn` 收到的座標換算成 widget local 座標（左上原點、
        y 往下增加）——`core/services/shot_panel_input_mapper.py` 全部函式都
        吃這個座標系。

        ⚠️ **此處依賴 spike 未證實的假設，實測後可能需要調整**，見 tech-design
        第 1.5 節：`set_mouse_pressed_fn`/`set_mouse_moved_fn` 的 x/y 是螢幕
        座標還是 widget local 座標，官方文件完全沒有陳述。目前實作假設是
        螢幕座標，用 `widget.screen_position_x/y`（文件確認是「上一次繪製
        時」的螢幕像素座標）相減換算。若實測發現傳進來的其實已經是 local
        座標，這個方法應該退化成 identity（`return x, y`）——只改這一處，
        所有呼叫端（`_on_circle_press` 等）完全不用動。
        """
        return x - widget.screen_position_x, y - widget.screen_position_y

    # ------------------------------------------------------------------
    # 版面建構
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # 只會在 __init__ 裡 self._root_frame 已確認非 None 時被呼叫，這裡
        # 加 early return 只是讓 Pyright 對 Optional 欄位做型別窄化，同時
        # 也是執行期防禦（見問題 3 修正說明）。
        if self._root_frame is None:
            return
        with self._root_frame:
            # 面板貼齊 viewport 右下角、佔寬 1/4、高 1/2。用 Placer 定位而不是
            # ZStack 的 alignment——ZStack.alignment 的語意（跨容器定位 vs 排列
            # 自己的子項）查不到文件明確陳述，兩種猜測都實測失敗；Placer 的
            # offset 定位子項左上角是查證過的行為，offset 設成
            # 100%-子項尺寸%，子項右下角就會精準貼齊。除錯歷程見 docs/CHANGELOG.md。
            with ui.ZStack():  # 填滿 self._root_frame（= viewport 實際渲染尺寸）
                with ui.Placer(
                    offset_x=ui.Percent(100.0 - _PANEL_WIDTH_PERCENT),
                    offset_y=ui.Percent(100.0 - _PANEL_HEIGHT_PERCENT),
                    width=ui.Percent(100),
                    height=ui.Percent(100),
                ):
                    self._panel_root_zstack = ui.ZStack(
                        width=ui.Percent(_PANEL_WIDTH_PERCENT),
                        height=ui.Percent(_PANEL_HEIGHT_PERCENT),
                    )
                    with self._panel_root_zstack:
                        if _DIAGNOSTIC_BACKGROUND_ONLY:
                            self._build_diagnostic_background_only()
                        else:
                            self._build_full_panel_content()

        # 見 _on_panel_hovered() docstring：多個 widget 疊在同一個 ZStack
        # 裡，哪一個會收到 hover 事件（geometric containment vs 頂層 widget
        # 專屬）沒有文件可查，外層 ZStack 與背板都掛同一個 handler——handler
        # 本身是冪等的（見該方法 docstring），多次觸發無害。診斷模式下也要掛，
        # 否則沒東西可以測「單純背板會不會擋住滾輪穿透」。
        self._panel_root_zstack.set_mouse_hovered_fn(self._on_panel_hovered)
        self._panel_background.set_mouse_hovered_fn(self._on_panel_hovered)

    def _build_diagnostic_background_only(self) -> None:
        """⚠️ 診斷用，見 `_DIAGNOSTIC_BACKGROUND_ONLY` 的說明——只畫一塊填滿
        `self._panel_root_zstack`（viewport 右下角，1/4 寬 × 1/2 高）的
        背板，不建任何互動元件，用來排除「是不是內容本身造成滾輪穿透／
        視覺溢出」。`width=height=ui.Percent(100)` 是相對**直接父層**
        `self._panel_root_zstack`，不是直接相對 viewport——單純填滿父層
        留給它的空間。除錯歷程見 docs/CHANGELOG.md。
        """
        self._panel_background = ui.Rectangle(
            width=ui.Percent(100),
            height=ui.Percent(100),
            style={
                "background_color": _PANEL_BACKGROUND_COLOR,
                "border_radius": 6,
            },
        )
        self._panel_background.set_mouse_wheel_fn(self._on_panel_wheel)

    def _build_full_panel_content(self) -> None:
        """正式版面：背板 + 一個垂直可捲動、內含可收合內容本體的 ScrollingFrame。

        內容（圓形選擇器 120px + 俯瞰圖 256px + 幾個 24px 列 + 狀態文字）
        粗估總高度遠超過面板分配到的高度（viewport 高的 1/2），裝不下就用
        ScrollingFrame 捲動，不讓面板本身撐高溢出邊界——捲動是硬性需求，
        不能拿掉。標題列跟捲動內容放在同一個 ScrollingFrame 裡（不是釘在
        外面固定不動）——多層巢狀容器的高度分配在這個 Kit 版本一再證實不
        可靠（見 docs/CHANGELOG.md「面板定位方式的除錯歷程」），標題列滾出
        畫面外還能捲回來看，比再賭一次「剩餘空間怎麼分配」風險小。

        ⚠️ 俯瞰圖／圓形選擇器內部用 `ui.Placer(Pixel offset)` 定位的元件
        （Kitchen 矩形、袋口、球堆、母球標記、瞄準線）先前實測會偏移；已定位
        元兇是本方法內、包住 body 的 VStack 曾經帶的 `style={"margin": 8}`
        （與 ScrollingFrame 本身、與外層面板定位用的 Placer 都無關），改用
        `ui.Spacer()` 圍內距後確認修好，除錯歷程見 docs/CHANGELOG.md。
        """
        # width/height 明確給 Percent(100)：Rectangle 跟緊接著的 ScrollingFrame
        # 是同一層的手足元件，各自預設撐滿的方式不一定一致，不明講的話背板
        # 可能只跟著收合狀態縮成標題列大小，蓋不住展開後的內容區域。
        self._panel_background = ui.Rectangle(
            width=ui.Percent(100),
            height=ui.Percent(100),
            style={
                "background_color": _PANEL_BACKGROUND_COLOR,
                "border_radius": 6,
            },
        )
        # 見 _on_panel_wheel() docstring：掛上去但 2026-09-08 實測發現沒有
        # 真的擋住滾輪穿透，留著沒有壞處，真正的修法是 _on_panel_hovered()
        # （carb.settings 停用相機手勢）。
        self._panel_background.set_mouse_wheel_fn(self._on_panel_wheel)

        self._scroll_frame = ui.ScrollingFrame(
            height=ui.Percent(100),
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            # ScrollingFrame 疊在 _panel_background 正上方，若它自己的背景不是
            # 透明的就會整個蓋掉底下的半透明背板；明確清成透明，不依賴預設值。
            style={"ScrollingFrame": {"background_color": 0x0}},
        )
        # 見 _on_panel_hovered() docstring：多個 widget 疊在同一個位置時，
        # 誰會收到 hover 事件沒有文件可查，ScrollingFrame 是內容區域最上層
        # 的容器，額外掛一份確保滾動內容時相機縮放手勢也會被停用。
        self._scroll_frame.set_mouse_hovered_fn(self._on_panel_hovered)
        with self._scroll_frame:
            # height=0 是 ScrollingFrame 官方範例的寫法：讓 VStack 依內容
            # 自然撐高，捲動的空間才有意義（不是被截斷成固定高度）。
            #
            # ⚠️ 內距刻意不用 style={"margin": 8}——實測確認這個 Kit 版本裡，
            # 帶 margin style、又沒有明確 width 的 VStack，會讓巢狀在裡面的
            # `ui.Placer` 算出錯誤的座標基準（俯瞰圖 Kitchen 矩形等元件全部
            # 偏移，跟 ScrollingFrame 本身無關），見 docs/CHANGELOG.md 除錯
            # 歷程。改用四個固定像素的 `ui.Spacer()` 手動圍出同樣的 8px
            # 內距，不透過 margin style。
            with ui.VStack(height=0, spacing=6):
                ui.Spacer(height=8)
                with ui.HStack():
                    ui.Spacer(width=8)
                    with ui.VStack(spacing=6):
                        with ui.HStack(height=28):
                            ui.Label("Shot Control")
                            ui.Spacer()
                            # 三角形符號（▼/▶）在 Isaac Sim 內建 UI 字型裡
                            # 沒有字形，會顯示成 "?"，改用一定有字形的 ASCII
                            # 字元；順便加大點擊區域，24px 見方偏小不好點。
                            self._collapse_button = ui.Button(
                                "v",
                                width=28,
                                height=28,
                                clicked_fn=self._on_collapse_button_clicked,
                            )

                        self._collapsible_body = ui.VStack(spacing=6)
                        with self._collapsible_body:
                            self._build_body_ui()
                    ui.Spacer(width=8)
                ui.Spacer(height=8)

        # 見「這些屬性無論 root frame 拿不拿得到都要先設好」那條註解旁邊的
        # 說明：預設收合，這裡把實際 widget 狀態同步成 __init__ 設的
        # self._is_collapsed=True，不能只改常數不改 widget（widget 建構時
        # 沒有讀那個旗標，預設一律可見）。
        self._collapsible_body.visible = False
        self._collapse_button.text = ">"

    def _build_body_ui(self) -> None:
        with ui.HStack(height=24, spacing=6):
            ui.Label("Table", width=50)
            ui.ComboBox(self._table_combo_model, width=180, height=24)

        with ui.HStack(height=24, spacing=6):
            self._controller_mode_label = ui.Label("Mode: -", width=100)
            self._controller_mode_button = ui.Button(
                "Switch", clicked_fn=self._on_controller_mode_button_clicked
            )

        # 讀數疊在圓形選擇器下方（不是並排）：並排的 HStack 裡「固定像素
        # 圓形 + 沒給寬度的 VStack」寬度分配在這個 Kit 版本不可靠（跟上一輪
        # Stack 高度分配的問題同一類），改成疊放讓 Label 吃到面板全寬。
        with ui.VStack(spacing=4):
            self._build_offset_picker()
            self._offset_readout_label = ui.Label("", word_wrap=True)

        with ui.HStack(height=24, spacing=6):
            ui.Label("Speed", width=40)
            # 力道輸入框選 FloatField + SimpleFloatModel，不退回 StringField：
            # tech-design 1.7 已從官方文件確認兩者的類別本身存在（「文件
            # 確認」等級），只有 get_value_as_float()/set_value()/
            # add_value_changed_fn() 這三個方法是「推測，需實測」——但這三個
            # 方法是 AbstractValueModel 家族的共用介面，本專案已有
            # SimpleStringModel 的先例（table_combo_box_model.py），同一個
            # 基底類別家族的方法命名極可能一致，風險評估上比另外實作一套
            # StringField + float() 手動解析（要自己處理無效輸入、還是要用
            # 同一套 AbstractValueModel 事件機制）更小、程式碼也更精簡。
            self._speed_model = ui.SimpleFloatModel(CUE_BALL_SPEED[1])
            self._speed_model.add_value_changed_fn(self._on_speed_value_changed)
            ui.FloatField(model=self._speed_model, width=70, height=24, precision=3)
            ui.Label(
                # en-dash（U+2013）在內建字型裡沒有字形，改用 ASCII 減號。
                f"m/s ({CUE_BALL_SPEED[0]:.2f}-{CUE_BALL_SPEED[1]:.2f})",
                word_wrap=True,
            )

        # 讀數疊在俯瞰圖下方（不是並排），理由同上——並排的固定像素俯瞰圖
        # 128px + 沒給寬度的 VStack 在窄面板下會把文字壓縮到逐字換行。
        with ui.VStack(spacing=4):
            self._build_topview_map()
            self._angle_placement_readout_label = ui.Label("", word_wrap=True)
            self._feasibility_label = ui.Label(
                "", word_wrap=True, style={"color": _FEASIBILITY_TEXT_COLOR}
            )

        with ui.HStack(height=24, spacing=6):
            self._shot_button = ui.Button("Shoot", clicked_fn=self._on_shot_button_clicked)
            self._reset_button = ui.Button(
                "Reset Table", clicked_fn=self._on_reset_button_clicked
            )

        # 獨立一顆按鈕，不跟上面的 Reset Table 合併：Reset Table 是隨時可按
        # 的強制重開整局（ERROR 復原用），Next Rack 是球停下來後、確認要
        # 重擺才重擺的一般流程，兩者語意不同。
        with ui.HStack(height=24, spacing=6):
            self._next_rack_button = ui.Button(
                "Next Rack", clicked_fn=self._on_next_rack_button_clicked
            )

        self._status_label = ui.Label("", word_wrap=True)
        self._shot_result_label = ui.Label("", word_wrap=True)

    def _build_offset_picker(self) -> None:
        """圓形擊球點選擇器：母球大圓 + 十字準星 + 可拖曳標記 + 透明滑鼠捕手。

        互動走「主方案（統一走滑鼠事件）」：ZStack 最上層放透明 `Rectangle`
        當滑鼠捕手，標記用非 draggable 的 `Placer` 由程式定位，不用
        `Placer(draggable=True)`——tech-design 1.4 確認該組合的參數都存在，
        但「這個特定組合在目前專案用的 Kit 版本裡沒有 bug」仍未實測，主方案
        本來就不依賴它，只當作未來手感優化的選項。
        """
        with ui.ZStack(width=_CIRCLE_SIZE_PX, height=_CIRCLE_SIZE_PX):
            ui.Circle(
                width=_CIRCLE_SIZE_PX,
                height=_CIRCLE_SIZE_PX,
                radius=_CIRCLE_RADIUS_PX,
                style={"background_color": cl(0.75, 0.75, 0.78, 0.9)},
            )
            with ui.Placer(offset_x=ui.Pixel(_CIRCLE_CENTER_PX - 0.5), offset_y=ui.Pixel(0.0)):
                ui.Rectangle(
                    width=1,
                    height=_CIRCLE_SIZE_PX,
                    style={"background_color": cl(0.3, 0.3, 0.32, 0.6)},
                )
            with ui.Placer(offset_x=ui.Pixel(0.0), offset_y=ui.Pixel(_CIRCLE_CENTER_PX - 0.5)):
                ui.Rectangle(
                    width=_CIRCLE_SIZE_PX,
                    height=1,
                    style={"background_color": cl(0.3, 0.3, 0.32, 0.6)},
                )

            self._offset_marker_placer = ui.Placer(
                offset_x=ui.Pixel(_CIRCLE_CENTER_PX - _CIRCLE_MARKER_RADIUS_PX),
                offset_y=ui.Pixel(_CIRCLE_CENTER_PX - _CIRCLE_MARKER_RADIUS_PX),
            )
            with self._offset_marker_placer:
                ui.Circle(
                    width=_CIRCLE_MARKER_RADIUS_PX * 2,
                    height=_CIRCLE_MARKER_RADIUS_PX * 2,
                    radius=_CIRCLE_MARKER_RADIUS_PX,
                    style={"background_color": cl(0.9, 0.1, 0.1, 1.0)},
                )

            self._circle_catcher = ui.Rectangle(
                width=_CIRCLE_SIZE_PX,
                height=_CIRCLE_SIZE_PX,
                style={"background_color": cl(0.0, 0.0, 0.0, 0.0)},
            )
            self._circle_catcher.set_mouse_pressed_fn(self._on_circle_press)
            self._circle_catcher.set_mouse_moved_fn(self._on_circle_move)
            self._circle_catcher.set_mouse_wheel_fn(self._on_panel_wheel)

    def _build_topview_map(self) -> None:
        """球桌俯瞰圖：桌面底色 → Kitchen 合法區 → head string → 6 個袋口
        → 開球球堆 9 顆（靜態，狀態機只在 is_init_state 時才離開 IDLE，按下
        擊球那一刻球一定在開球位置）→ 母球標記（可拖）→ 瞄準線（`_AIM_DOT_COUNT`
        個小圓點，不用 `ui.Line`——軸對齊，畫斜線要 `FreeLine` + 兩個 anchor
        widget，是風險最高的未知 API，見計畫「俯瞰圖圖層」一節）→ 透明滑鼠
        捕手。
        """
        with ui.ZStack(width=_TOPVIEW_WIDTH_PX, height=_TOPVIEW_HEIGHT_PX):
            ui.Rectangle(
                width=_TOPVIEW_WIDTH_PX,
                height=_TOPVIEW_HEIGHT_PX,
                style={"background_color": cl(0.05, 0.20, 0.08, 1.0)},
            )
            self._draw_static_kitchen_region()
            self._draw_static_head_string()
            self._draw_static_pockets()
            self._draw_static_rack_balls()

            self._cue_marker_placer = ui.Placer(offset_x=ui.Pixel(0.0), offset_y=ui.Pixel(0.0))
            with self._cue_marker_placer:
                ui.Circle(
                    width=_CUE_BALL_MARKER_RADIUS_PX * 2,
                    height=_CUE_BALL_MARKER_RADIUS_PX * 2,
                    radius=_CUE_BALL_MARKER_RADIUS_PX,
                    style={"background_color": cl(1.0, 1.0, 1.0, 1.0)},
                )

            self._aim_dot_placers: list[tuple[ui.Placer, ui.Circle]] = []
            for _ in range(_AIM_DOT_COUNT):
                placer = ui.Placer(offset_x=ui.Pixel(0.0), offset_y=ui.Pixel(0.0))
                with placer:
                    dot = ui.Circle(
                        width=_AIM_DOT_RADIUS_PX * 2,
                        height=_AIM_DOT_RADIUS_PX * 2,
                        radius=_AIM_DOT_RADIUS_PX,
                        style={"background_color": _AIM_LINE_COLOR},
                    )
                self._aim_dot_placers.append((placer, dot))

            self._topview_catcher = ui.Rectangle(
                width=_TOPVIEW_WIDTH_PX,
                height=_TOPVIEW_HEIGHT_PX,
                style={"background_color": cl(0.0, 0.0, 0.0, 0.0)},
            )
            self._topview_catcher.set_mouse_pressed_fn(self._on_topview_press)
            self._topview_catcher.set_mouse_moved_fn(self._on_topview_move)
            self._topview_catcher.set_mouse_released_fn(self._on_topview_release)
            self._topview_catcher.set_mouse_wheel_fn(self._on_panel_wheel)

    def _draw_static_kitchen_region(self) -> None:
        # 畫的是真實撞球桌 Kitchen 線（球面到線的距離），不是母球球心的合法
        # 擺位範圍——兩者差一顆球半徑，見 mapper.kitchen_line_bounds()
        # docstring。拖曳限制仍然是 clamp_cue_ball_placement()／
        # CUE_BALL_PLACEMENT_X/Y，這裡只是畫給使用者看的參考線，兩者刻意
        # 不同一組數字。
        x_min, x_max, y_min, y_max = mapper.kitchen_line_bounds()
        x0, y0 = mapper.topview_pixels_from_table_xy(
            x_min, y_min, _TOPVIEW_WIDTH_PX, _TOPVIEW_HEIGHT_PX
        )
        x1, y1 = mapper.topview_pixels_from_table_xy(
            x_max, y_max, _TOPVIEW_WIDTH_PX, _TOPVIEW_HEIGHT_PX
        )
        left, right = min(x0, x1), max(x0, x1)
        top, bottom = min(y0, y1), max(y0, y1)
        with ui.Placer(offset_x=ui.Pixel(left), offset_y=ui.Pixel(top)):
            ui.Rectangle(
                width=right - left,
                height=bottom - top,
                style={"background_color": cl(0.3, 0.65, 0.3, 0.25)},
            )

    def _draw_static_head_string(self) -> None:
        # head string 是 Kitchen 真實邊界線，要跟 _draw_static_kitchen_region()
        # 畫的矩形同一組數字（kitchen_line_bounds() 的 y_max），不能用球心
        # 範圍的 CUE_BALL_PLACEMENT_Y[1]——否則矩形邊緣會超出這條線。
        _, _, _, y_max = mapper.kitchen_line_bounds()
        _, py = mapper.topview_pixels_from_table_xy(
            0.0, y_max, _TOPVIEW_WIDTH_PX, _TOPVIEW_HEIGHT_PX
        )
        with ui.Placer(offset_x=ui.Pixel(0.0), offset_y=ui.Pixel(py)):
            ui.Rectangle(
                width=_TOPVIEW_WIDTH_PX,
                height=1,
                style={"background_color": cl(0.8, 0.8, 0.8, 0.5)},
            )

    def _draw_static_pockets(self) -> None:
        for x, y in POCKET_POSITIONS.values():
            px, py = mapper.topview_pixels_from_table_xy(x, y, _TOPVIEW_WIDTH_PX, _TOPVIEW_HEIGHT_PX)
            with ui.Placer(
                offset_x=ui.Pixel(px - _POCKET_MARKER_RADIUS_PX),
                offset_y=ui.Pixel(py - _POCKET_MARKER_RADIUS_PX),
            ):
                ui.Circle(
                    width=_POCKET_MARKER_RADIUS_PX * 2,
                    height=_POCKET_MARKER_RADIUS_PX * 2,
                    radius=_POCKET_MARKER_RADIUS_PX,
                    style={"background_color": cl(0.02, 0.02, 0.02, 1.0)},
                )

    def _draw_static_rack_balls(self) -> None:
        for ball_id, (x, y) in BREAK_SHOT_POSITIONS.items():
            if ball_id == 0:
                continue  # 母球另外用可拖曳標記畫，不在這裡重複畫
            px, py = mapper.topview_pixels_from_table_xy(x, y, _TOPVIEW_WIDTH_PX, _TOPVIEW_HEIGHT_PX)
            with ui.Placer(
                offset_x=ui.Pixel(px - _RACK_BALL_MARKER_RADIUS_PX),
                offset_y=ui.Pixel(py - _RACK_BALL_MARKER_RADIUS_PX),
            ):
                ui.Circle(
                    width=_RACK_BALL_MARKER_RADIUS_PX * 2,
                    height=_RACK_BALL_MARKER_RADIUS_PX * 2,
                    radius=_RACK_BALL_MARKER_RADIUS_PX,
                    style={"background_color": cl(0.85, 0.85, 0.8, 1.0)},
                )

    # ------------------------------------------------------------------
    # 圓形擊球點選擇器互動
    # ------------------------------------------------------------------

    def _on_circle_press(self, x: float, y: float, button: int, modifier: int) -> None:
        self._handle_circle_pointer(x, y)

    def _on_circle_move(self, x: float, y: float, modifier: int, is_pressed: bool) -> None:
        # set_mouse_moved_fn 只在按著鍵拖曳時觸發（tech-design 1.5 文件確
        # 認），不需要另外判斷 is_pressed。
        self._handle_circle_pointer(x, y)

    def _handle_circle_pointer(self, x: float, y: float) -> None:
        if self._table_combo_model is None:
            return
        table_id = self._table_combo_model.get_selected_table_id()
        if table_id is None:
            return
        local_x, local_y = self._to_local(x, y, self._circle_catcher)
        new_offset = mapper.offset_from_circle_pixels(
            local_x,
            local_y,
            _CIRCLE_CENTER_PX,
            _CIRCLE_CENTER_PX,
            _CIRCLE_RADIUS_PX,
            _CIRCLE_MAX_OFFSET,
        )
        current = self._current_parameters(table_id)
        updated = dataclasses.replace(current, position_offset=tuple(new_offset))
        self._push_parameters(table_id, updated)

    def _reposition_offset_marker(self, position_offset) -> None:
        marker_px, marker_py = mapper.circle_pixels_from_offset(
            position_offset, _CIRCLE_CENTER_PX, _CIRCLE_CENTER_PX, _CIRCLE_RADIUS_PX
        )
        self._offset_marker_placer.offset_x = ui.Pixel(marker_px - _CIRCLE_MARKER_RADIUS_PX)
        self._offset_marker_placer.offset_y = ui.Pixel(marker_py - _CIRCLE_MARKER_RADIUS_PX)

    # ------------------------------------------------------------------
    # 力道輸入框
    # ------------------------------------------------------------------

    def _on_speed_value_changed(self, model: ui.SimpleFloatModel) -> None:
        if self._suppress_speed_callback:
            return
        if self._table_combo_model is None:
            return
        table_id = self._table_combo_model.get_selected_table_id()
        if table_id is None:
            return
        raw_value = model.get_value_as_float()
        clamped = min(max(raw_value, CUE_BALL_SPEED[0]), CUE_BALL_SPEED[1])
        current = self._current_parameters(table_id)
        updated = dataclasses.replace(current, cue_ball_speed=clamped)
        # 被夾過就回填讓使用者看到——_redraw_controls() 裡的 _speed_model.
        # set_value() 統一處理，不在這裡另外寫回，避免兩處各自防遞迴。
        self._push_parameters(table_id, updated)

    # ------------------------------------------------------------------
    # 俯瞰圖互動
    # ------------------------------------------------------------------

    def _on_topview_press(self, x: float, y: float, button: int, modifier: int) -> None:
        if self._table_combo_model is None:
            return
        table_id = self._table_combo_model.get_selected_table_id()
        if table_id is None:
            return
        local_x, local_y = self._to_local(x, y, self._topview_catcher)
        current = self._current_parameters(table_id)
        cue_px, cue_py = mapper.topview_pixels_from_table_xy(
            current.cue_ball_placement[0],
            current.cue_ball_placement[1],
            _TOPVIEW_WIDTH_PX,
            _TOPVIEW_HEIGHT_PX,
        )
        if mapper.is_within_radius(local_x, local_y, cue_px, cue_py, _CUE_BALL_HIT_RADIUS_PX):
            self._topview_drag_mode = "placement"
        else:
            self._topview_drag_mode = "angle"
        self._handle_topview_pointer(table_id, local_x, local_y)

    def _on_topview_move(self, x: float, y: float, modifier: int, is_pressed: bool) -> None:
        if self._topview_drag_mode is None:
            return
        if self._table_combo_model is None:
            return
        table_id = self._table_combo_model.get_selected_table_id()
        if table_id is None:
            return
        local_x, local_y = self._to_local(x, y, self._topview_catcher)
        self._handle_topview_pointer(table_id, local_x, local_y)

    def _on_topview_release(self, x: float, y: float, button: int, modifier: int) -> None:
        self._topview_drag_mode = None

    # ------------------------------------------------------------------
    # 滾輪事件：擋掉往下傳給 viewport 相機縮放
    # ------------------------------------------------------------------

    def _on_panel_wheel(self, x: float, y: float, modifier: int) -> None:
        """滑鼠滾輪 no-op handler，掛在面板背板 `_panel_background`（覆蓋
        整個面板本體）與兩個互動畫布的透明滑鼠捕手（`_circle_catcher`／
        `_topview_catcher`）上。

        ⚠️ **2026-09-08 實測結果：這個假設不成立。** 使用者在 GUI 下實際
        滾動，確認掛了這個 no-op handler 之後滾輪還是會讓 viewport 相機
        縮放——不是「事件送到這裡、又繼續往下傳」的猜測，是真的擋不住。

        推論：`omni.kit.manipulator.camera` 的相機操作走的是獨立於
        `omni.ui` widget 事件樹的輸入路徑（官方文件用 `carb.settings` 的
        手勢綁定表描述相機互動，不是用 widget 的事件消費機制），也就是說
        在 `ui.Widget` 上「認領」一個事件，並不會讓底層相機操作的輸入監聽
        跟著停下來——這兩層是分開的。真正的修法是 `_on_panel_hovered()`
        （懸停時透過 `carb.settings` 暫時停用相機的 `ZoomScrollGesture`
        手勢），這個 handler 留著沒有壞處但已知無效，之後若要精簡可以直接
        移除，這裡先保留紀錄。
        """

    def _on_panel_hovered(self, hovered: bool) -> None:
        """滑鼠懸停在面板範圍內／離開時，暫時停用／還原
        `ZoomScrollGesture`（滾輪縮放）這個相機手勢——見
        `_CAMERA_BINDINGS_SETTING_PATH` 常數旁的官方文件依據與推導。

        懸停進入：讀目前的 `/exts/omni.kit.viewport.window/bindings/camera`
        整包設定值存進 `self._saved_camera_bindings`，再寫回一份拿掉
        `ZoomScrollGesture` 的副本。懸停離開：把存起來的原始值寫回去，並把
        `self._saved_camera_bindings` 清成 `None`。`self._saved_camera_bindings
        is not None` 同時是「目前已經停用中」的判斷依據，避免同一次懸停
        期間（多個 widget 各自觸發 hover）重複存值蓋掉更早存的原始值。

        ⚠️ **這裡有兩個尚未在 GUI 下驗證的假設**：
        1. `set_mouse_hovered_fn()` 的觸發範圍——多個 widget（外層
           `_panel_root_zstack`、`_panel_background`）疊在同一個位置，
           `hovered=True/False` 是照「游標是否落在這個 widget 的幾何範圍
           內」判斷（跟 z-order／有沒有被其他 widget 蓋住無關），還是照
           「這個 widget 是不是滑鼠事件的目標」判斷（跟滾輪穿透一樣，只有
           最上層的 widget 才會觸發），官方文件沒有陳述。若是後者，懸停在
           被其他子 widget（Label/Button/ComboBox/FloatField）蓋住的位置
           時，這兩個 widget 都不會觸發 hover，滾輪縮放在那些位置一樣不會
           被停用——這正是 `_DIAGNOSTIC_BACKGROUND_ONLY` 診斷分支要排除
           的第一件事：拿掉所有子 widget、只留純背板，如果懸停在純背板上
           滾輪縮放確實被擋住了，就能確定這個機制本身有效，殘留的問題
           只在「哪些位置沒有掛到 hover」。
        2. **這個設定是全域的，不是 per-viewport 的**——路徑
           `/exts/omni.kit.viewport.window/bindings/camera` 沒有任何
           viewport id 或 window 限定詞。本專案目前只會同時顯示一個
           Demo 桌的 viewport，這個限制暫時不影響使用，但如果未來場景
           變成多視窗，懸停在其中一個面板會連帶停用所有視窗的滾輪縮放。

        目前只停用滾輪縮放（`_DISABLED_CAMERA_GESTURES`）。右鍵環景
        （`LookGesture`）、Alt+左鍵翻轉（`TumbleGesture`）、Alt+右鍵縮放
        （`ZoomGesture`）理論上可能有同一種穿透，但面板互動沒有用到那些
        按鍵組合，使用者也還沒回報那幾個手勢的問題，這次不主動處理。
        """
        settings = carb.settings.get_settings()
        if hovered:
            if self._saved_camera_bindings is not None:
                return
            current = settings.get(_CAMERA_BINDINGS_SETTING_PATH)
            current = dict(current) if current else {}
            self._saved_camera_bindings = current
            modified = {
                key: value
                for key, value in current.items()
                if key not in _DISABLED_CAMERA_GESTURES
            }
            settings.set(_CAMERA_BINDINGS_SETTING_PATH, modified)
        else:
            if self._saved_camera_bindings is None:
                return
            settings.set(_CAMERA_BINDINGS_SETTING_PATH, self._saved_camera_bindings)
            self._saved_camera_bindings = None

    def _handle_topview_pointer(self, table_id: str, local_x: float, local_y: float) -> None:
        current = self._current_parameters(table_id)
        if self._topview_drag_mode == "placement":
            table_x, table_y = mapper.table_xy_from_topview_pixels(
                local_x, local_y, _TOPVIEW_WIDTH_PX, _TOPVIEW_HEIGHT_PX
            )
            clamped_xy = mapper.clamp_cue_ball_placement(table_x, table_y)
            updated = dataclasses.replace(current, cue_ball_placement=clamped_xy)
        elif self._topview_drag_mode == "angle":
            target_xy = mapper.table_xy_from_topview_pixels(
                local_x, local_y, _TOPVIEW_WIDTH_PX, _TOPVIEW_HEIGHT_PX
            )
            try:
                angle = mapper.shot_angle_from_points(current.cue_ball_placement, target_xy)
            except ValueError:
                # 按太靠近母球本身（距離小於一顆球半徑），方向未定義，忽略
                # 這次事件，不更動角度。
                return
            clamped_angle = mapper.clamp_manual_shot_angle(angle)
            updated = dataclasses.replace(current, shot_angle=clamped_angle)
        else:
            return
        self._push_parameters(table_id, updated)

    def _redraw_topview(self, parameters: ManualShotParameters, feasibility: ManualShotFeasibility) -> None:
        cue_x, cue_y = parameters.cue_ball_placement
        cue_px, cue_py = mapper.topview_pixels_from_table_xy(
            cue_x, cue_y, _TOPVIEW_WIDTH_PX, _TOPVIEW_HEIGHT_PX
        )
        self._cue_marker_placer.offset_x = ui.Pixel(cue_px - _CUE_BALL_MARKER_RADIUS_PX)
        self._cue_marker_placer.offset_y = ui.Pixel(cue_py - _CUE_BALL_MARKER_RADIUS_PX)

        end_x, end_y = mapper.aim_line_endpoint((cue_x, cue_y), parameters.shot_angle)
        dot_color = _AIM_LINE_COLOR if feasibility.is_feasible else _AIM_LINE_INFEASIBLE_COLOR
        dot_count = len(self._aim_dot_placers)
        for index, (placer, dot) in enumerate(self._aim_dot_placers):
            # 點等距排列在母球與瞄準線終點之間（不含母球本身這一端），
            # 視覺上就是撞球軟體常見的虛線瞄準線。
            t = (index + 1) / (dot_count + 1)
            dot_x = cue_x + (end_x - cue_x) * t
            dot_y = cue_y + (end_y - cue_y) * t
            dot_px, dot_py = mapper.topview_pixels_from_table_xy(
                dot_x, dot_y, _TOPVIEW_WIDTH_PX, _TOPVIEW_HEIGHT_PX
            )
            placer.offset_x = ui.Pixel(dot_px - _AIM_DOT_RADIUS_PX)
            placer.offset_y = ui.Pixel(dot_py - _AIM_DOT_RADIUS_PX)
            dot.style = {"background_color": dot_color}

    # ------------------------------------------------------------------
    # 按鈕
    # ------------------------------------------------------------------

    def _on_shot_button_clicked(self) -> None:
        if self._table_combo_model is None:
            return
        table_id = self._table_combo_model.get_selected_table_id()
        if table_id is None:
            return
        self._on_shot_requested(table_id)

    def _on_reset_button_clicked(self) -> None:
        if self._table_combo_model is None:
            return
        table_id = self._table_combo_model.get_selected_table_id()
        if table_id is None:
            return
        self._on_reset_requested(table_id)

    def _on_controller_mode_button_clicked(self) -> None:
        if self._table_combo_model is None:
            return
        table_id = self._table_combo_model.get_selected_table_id()
        if table_id is None:
            return
        self._on_toggle_controller_mode(table_id)

    def _on_next_rack_button_clicked(self) -> None:
        if self._table_combo_model is None:
            return
        table_id = self._table_combo_model.get_selected_table_id()
        if table_id is None:
            return
        self._confirm_reset(table_id)

    def _on_collapse_button_clicked(self) -> None:
        self._is_collapsed = not self._is_collapsed
        self._collapsible_body.visible = not self._is_collapsed
        self._collapse_button.text = ">" if self._is_collapsed else "v"

    # ------------------------------------------------------------------
    # 資料流：讀取現值／推送更新／整批重繪
    # ------------------------------------------------------------------

    def _current_parameters(self, table_id: "str | None") -> ManualShotParameters:
        """回傳 `table_id` 目前的參數；查無 table_id 或常駐 controller 尚未
        回報任何值（`get_parameters()` 回 None）一律回傳
        `ManualShotParameters.default()` 當作畫面上的安全預設值。"""
        if table_id is None:
            return ManualShotParameters.default()
        parameters = self._get_parameters(table_id)
        if parameters is None:
            return ManualShotParameters.default()
        return parameters

    def _push_parameters(self, table_id: str, parameters: ManualShotParameters) -> None:
        """使用者互動（拖曳/輸入）產生一組新參數後的統一出口：先推給常駐
        controller，再用同一份物件重繪畫面——不重新呼叫 `get_parameters()`
        讀回來，省一趟往返，也避免「controller 端還沒來得及套用」的競態。"""
        self._on_parameters_changed(table_id, parameters)
        self._redraw_controls(table_id, parameters)

    def _evaluate_feasibility(
        self, table_id: "str | None", parameters: ManualShotParameters
    ) -> ManualShotFeasibility:
        if table_id is None:
            return ManualShotFeasibility(is_feasible=True, reason="")
        geometry = self._get_table_geometry(table_id)
        if geometry is None:
            return ManualShotFeasibility(is_feasible=True, reason="")
        table_z, ball_radius = geometry
        return evaluate_manual_shot_parameters(parameters, table_z, ball_radius)

    def _redraw_controls(self, table_id: "str | None", parameters: ManualShotParameters) -> None:
        """整批重繪：圓形選擇器標記、力道欄、俯瞰圖（母球標記＋瞄準線＋
        可行性紅線）、讀數 Label、按鈕可用狀態。只在互動（拖曳/輸入/切桌）
        時呼叫，不在 `_on_update()` 裡每 frame 呼叫。"""
        self._reposition_offset_marker(parameters.position_offset)
        offset_v, offset_h = parameters.position_offset
        # ↑/→ 箭頭符號在 Isaac Sim 內建 UI 字型裡沒有字形，換成一定有字形
        # 的 ASCII 前綴（跟 v/> 收合鈕、力道範圍標籤同一類修法）。
        self._offset_readout_label.text = f"V:{offset_v:+.3f}  H:{offset_h:+.3f}"

        self._suppress_speed_callback = True
        self._speed_model.set_value(parameters.cue_ball_speed)
        self._suppress_speed_callback = False

        feasibility = self._evaluate_feasibility(table_id, parameters)
        self._redraw_topview(parameters, feasibility)

        placement_x, placement_y = parameters.cue_ball_placement
        self._angle_placement_readout_label.text = (
            f"Angle {parameters.shot_angle:+.1f}°  Placement ({placement_x:+.3f}, {placement_y:+.3f})"
        )
        self._feasibility_label.text = "" if feasibility.is_feasible else f"Infeasible: {feasibility.reason}"

        self._shot_button.enabled = table_id is not None and feasibility.is_feasible
        self._reset_button.enabled = table_id is not None

    def set_available_tables(self, table_ids: list[str]) -> None:
        """給呼叫端（`billiard_digital_twin.py`）在 Demo 桌增刪後 push 選桌
        清單，跟 `DebugMenu.set_available_tables()` 同樣寫法／同樣的呼叫時機
        （桌子增刪當下），不再由面板每 frame 輪詢。面板未成功建構
        （headless，`self._table_combo_model` 為 None）時安靜 no-op，跟
        `destroy()` 之後任何外部呼叫都應該安全的既定慣例一致。"""
        if self._table_combo_model is None:
            return
        self._table_combo_model.set_items(table_ids)

    def _refresh_controls_for_selected_table(self) -> None:
        # 診斷模式下沒有任何 body widget（讀數 Label、offset marker、俯瞰圖
        # 圖層……）可以刷新，一定要在碰它們之前擋掉，否則對不存在的屬性
        # 直接 AttributeError，整個 extension startup 就失敗。
        if _DIAGNOSTIC_BACKGROUND_ONLY:
            return
        if self._table_combo_model is None:
            return
        table_id = self._table_combo_model.get_selected_table_id()
        parameters = self._current_parameters(table_id)
        self._redraw_controls(table_id, parameters)

    # ------------------------------------------------------------------
    # 每 frame 輪詢
    # ------------------------------------------------------------------

    def _on_update(self, event) -> None:
        """沿用 debug_menu 的每 frame 輪詢寫法，但只更新狀態列 Label（輕量的
        文字操作），不重繪畫布——畫布只在互動或切桌時才重繪（見
        `_redraw_controls()` docstring）。桌子清單改用 push（見
        `set_available_tables()`），這裡不再輪詢查表。

        診斷模式（`_DIAGNOSTIC_BACKGROUND_ONLY`）下 `self._status_label`
        根本不存在，整個狀態列更新跳過——純背板沒有東西可以顯示狀態。
        """
        if _DIAGNOSTIC_BACKGROUND_ONLY:
            return
        if self._table_combo_model is None:
            return
        table_id = self._table_combo_model.get_selected_table_id()
        if table_id != self._last_selected_table_id:
            self._last_selected_table_id = table_id
            self._refresh_controls_for_selected_table()
        self._status_label.text = self._get_shot_status_text(table_id) if table_id is not None else ""

        # 模式顯示跟狀態列一樣走每 frame 輪詢，不是只在互動時重繪——Debug
        # Menu 也能改這個狀態，面板必須反映外部變化，不能只信任自己上次
        # 按鈕點擊後的畫面。
        mode_text = self._get_controller_mode_text(table_id) if table_id is not None else ""
        self._controller_mode_label.text = f"Mode: {mode_text}" if mode_text else "Mode: -"
        if mode_text == "AI":
            self._controller_mode_button.text = "Switch to Manual"
        elif mode_text == "Manual":
            self._controller_mode_button.text = "Switch to AI"
        else:
            self._controller_mode_button.text = "Switch"
        self._controller_mode_button.enabled = table_id is not None

        self._shot_result_label.text = (
            self._get_shot_result_text(table_id) if table_id is not None else ""
        )
        self._next_rack_button.enabled = (
            self._is_ready_to_reset(table_id) if table_id is not None else False
        )

    # ------------------------------------------------------------------
    # 生命週期
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        # 安全網：若面板在游標還停在上面、camera bindings 還處於暫停狀態
        # 時被銷毀（例如 Demo toggle 關閉、extension 重載），不能讓使用者
        # 的滾輪縮放永遠停在停用狀態——見 _on_panel_hovered() docstring。
        if self._saved_camera_bindings is not None:
            carb.settings.get_settings().set(
                _CAMERA_BINDINGS_SETTING_PATH, self._saved_camera_bindings
            )
            self._saved_camera_bindings = None
        self._update_sub = None
        if self._root_frame is not None:
            self._root_frame.clear()
            self._root_frame = None
        self._table_combo_model = None
