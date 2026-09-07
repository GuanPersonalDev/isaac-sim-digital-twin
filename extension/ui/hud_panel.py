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

⚠️ **本檔兩處明確隔離了 spike 未證實的假設**，實測後若需要調整，只需要改
對應的那一個方法，其餘部分不動：

1. `_create_root_frame()` — overlay 容器怎麼拿到（`get_frame()` vs 退回
   獨立 `ui.Window`）。
2. `_to_local()` — `set_mouse_*_fn` 收到的座標怎麼換算成 widget local 座標。

兩者的細節與可信度標示見
`docs/tech-design/hud-shot-control-panel-tech-design.md` 第 1 節。

其餘實作細節（`Placer.offset_x` 是否接受 `ui.Pixel`、`FloatField`/
`SimpleFloatModel` 的讀寫方法、overlay 上的滑鼠事件會不會被 viewport 相機
操作吃掉……）沒有被抽成獨立方法，因為 tech-design 已在文件層級對每一項給
出可信度標示與備援方案；改動這些假設通常只影響單一個 widget 的建構程式碼，
不需要額外的間接層。實作時的判斷與理由見各私有方法的 docstring。
"""

import dataclasses
from typing import Callable

import omni.kit.app
import omni.ui as ui
from omni.ui import color as cl

from .table_combo_box_model import TableComboBoxModel

from core.models.action_bounds import CUE_BALL_SPEED
from core.models.manual_shot_bounds import CUE_BALL_PLACEMENT_X, CUE_BALL_PLACEMENT_Y
from core.models.manual_shot_parameters import ManualShotParameters
from core.services import shot_panel_input_mapper as mapper
from core.services.break_shot_position_provider import BREAK_SHOT_POSITIONS
from core.services.manual_shot_feasibility import (
    ManualShotFeasibility,
    evaluate_manual_shot_parameters,
)
from core.services.pocket_geometry import POCKET_POSITIONS

# 面板本體寬度（含背板），viewport overlay 用 Spacer 相對定位推到左下角，
# 不寫死絕對座標，理由見計畫「HUD 嵌在 Viewport 內」一節。
_PANEL_WIDTH = 330.0

# 圓形擊球點選擇器：120x120，半徑 60px 填滿整個方框，標記半徑 6px。數值全部
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
_CUE_BALL_MARKER_RADIUS_PX = 5.0
# 命中判定半徑刻意大於視覺半徑（球心對應像素半徑約 2.9px 太小難點擊）——這是
# UX 上的選擇，不是物理量，跟 evaluate_manual_shot 用的 ball_radius 無關。
_CUE_BALL_HIT_RADIUS_PX = 10.0
_RACK_BALL_MARKER_RADIUS_PX = 3.0
_POCKET_MARKER_RADIUS_PX = 4.0
_AIM_DOT_COUNT = 12
_AIM_DOT_RADIUS_PX = 1.5

_PANEL_BACKGROUND_COLOR = cl(0.08, 0.08, 0.10, 0.55)
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
    ) -> None:
        self._ext_id = ext_id
        self._get_parameters = get_parameters
        self._on_parameters_changed = on_parameters_changed
        self._on_shot_requested = on_shot_requested
        self._on_reset_requested = on_reset_requested
        self._get_shot_status_text = get_shot_status_text
        self._get_table_geometry = get_table_geometry

        # 這些屬性無論 root frame 拿不拿得到都要先設好，destroy() 才能在
        # headless（root frame 為 None）下安全呼叫。
        self._update_sub = None
        self._table_combo_model: TableComboBoxModel | None = None
        self._is_collapsed = False
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
            with ui.VStack():
                ui.Spacer()  # 把面板推到 viewport 底部
                with ui.HStack():
                    with ui.ZStack(width=_PANEL_WIDTH):
                        ui.Rectangle(
                            style={
                                "background_color": _PANEL_BACKGROUND_COLOR,
                                "border_radius": 6,
                            }
                        )
                        with ui.VStack(spacing=6, style={"margin": 8}):
                            with ui.HStack(height=24):
                                ui.Label("Shot Control")
                                ui.Spacer()
                                self._collapse_button = ui.Button(
                                    "▼",
                                    width=24,
                                    height=24,
                                    clicked_fn=self._on_collapse_button_clicked,
                                )

                            self._collapsible_body = ui.VStack(spacing=6)
                            with self._collapsible_body:
                                self._build_body_ui()
                    ui.Spacer()  # 把面板推到 viewport 左側

    def _build_body_ui(self) -> None:
        with ui.HStack(height=24, spacing=6):
            ui.Label("Table", width=50)
            ui.ComboBox(self._table_combo_model, width=180, height=24)

        with ui.HStack(spacing=6):
            self._build_offset_picker()
            with ui.VStack():
                self._offset_readout_label = ui.Label("", word_wrap=True)

        with ui.HStack(height=24, spacing=6):
            ui.Label("力道", width=40)
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
                f"m/s ({CUE_BALL_SPEED[0]:.2f}–{CUE_BALL_SPEED[1]:.2f})",
                word_wrap=True,
            )

        with ui.HStack(spacing=6):
            self._build_topview_map()
            with ui.VStack():
                self._angle_placement_readout_label = ui.Label("", word_wrap=True)
                self._feasibility_label = ui.Label(
                    "", word_wrap=True, style={"color": _FEASIBILITY_TEXT_COLOR}
                )

        with ui.HStack(height=24, spacing=6):
            self._shot_button = ui.Button("擊球", clicked_fn=self._on_shot_button_clicked)
            self._reset_button = ui.Button(
                "重設球局", clicked_fn=self._on_reset_button_clicked
            )

        self._status_label = ui.Label("", word_wrap=True)

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
                    style={"background_color": cl(0.95, 0.35, 0.15, 1.0)},
                )

            self._circle_catcher = ui.Rectangle(
                width=_CIRCLE_SIZE_PX,
                height=_CIRCLE_SIZE_PX,
                style={"background_color": cl(0.0, 0.0, 0.0, 0.0)},
            )
            self._circle_catcher.set_mouse_pressed_fn(self._on_circle_press)
            self._circle_catcher.set_mouse_moved_fn(self._on_circle_move)

    def _build_topview_map(self) -> None:
        """球桌俯瞰圖：桌面底色 → Kitchen 合法區 → head string → 6 個袋口
        → 開球球堆 9 顆（靜態，狀態機只在 is_init_state 時才離開 IDLE，按下
        擊球那一刻球一定在開球位置）→ 母球標記（可拖）→ 瞄準線（12 個小
        圓點，不用 `ui.Line`——軸對齊，畫斜線要 `FreeLine` + 兩個 anchor
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
                    style={"background_color": cl(0.95, 0.95, 0.92, 1.0)},
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

    def _draw_static_kitchen_region(self) -> None:
        x0, y0 = mapper.topview_pixels_from_table_xy(
            CUE_BALL_PLACEMENT_X[0], CUE_BALL_PLACEMENT_Y[0], _TOPVIEW_WIDTH_PX, _TOPVIEW_HEIGHT_PX
        )
        x1, y1 = mapper.topview_pixels_from_table_xy(
            CUE_BALL_PLACEMENT_X[1], CUE_BALL_PLACEMENT_Y[1], _TOPVIEW_WIDTH_PX, _TOPVIEW_HEIGHT_PX
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
        _, py = mapper.topview_pixels_from_table_xy(
            0.0, CUE_BALL_PLACEMENT_Y[1], _TOPVIEW_WIDTH_PX, _TOPVIEW_HEIGHT_PX
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
            # 12 個點等距排列在母球與瞄準線終點之間（不含母球本身這一端），
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

    def _on_collapse_button_clicked(self) -> None:
        self._is_collapsed = not self._is_collapsed
        self._collapsible_body.visible = not self._is_collapsed
        self._collapse_button.text = "▶" if self._is_collapsed else "▼"

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
        self._offset_readout_label.text = f"↑{offset_v:+.3f}  →{offset_h:+.3f}"

        self._suppress_speed_callback = True
        self._speed_model.set_value(parameters.cue_ball_speed)
        self._suppress_speed_callback = False

        feasibility = self._evaluate_feasibility(table_id, parameters)
        self._redraw_topview(parameters, feasibility)

        placement_x, placement_y = parameters.cue_ball_placement
        self._angle_placement_readout_label.text = (
            f"角度 {parameters.shot_angle:+.1f}°  擺位 ({placement_x:+.3f}, {placement_y:+.3f})"
        )
        self._feasibility_label.text = "" if feasibility.is_feasible else f"不可行：{feasibility.reason}"

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
        `set_available_tables()`），這裡不再輪詢查表。"""
        if self._table_combo_model is None:
            return
        table_id = self._table_combo_model.get_selected_table_id()
        if table_id != self._last_selected_table_id:
            self._last_selected_table_id = table_id
            self._refresh_controls_for_selected_table()
        self._status_label.text = self._get_shot_status_text(table_id) if table_id is not None else ""

    # ------------------------------------------------------------------
    # 生命週期
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        self._update_sub = None
        if self._root_frame is not None:
            self._root_frame.clear()
            self._root_frame = None
        self._table_combo_model = None
