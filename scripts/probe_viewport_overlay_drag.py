"""
scripts/probe_viewport_overlay_drag.py — #115 階段 0 API spike：**這支腳本
不是 headless**，是給使用者在有畫面的 GUI 下親手拖曳的最小實驗。

唯一目的：確認「疊在 viewport 內的 overlay 上拖曳滑鼠」會不會同時把
viewport 的相機操作（tumble/pan/zoom）一起吃走。這是整個 #115 計畫最大的
未知（見 `docs/tech-design/hud-shot-control-panel-tech-design.md` 第一節與
技術計畫 `115-hud-peppy-harbor.md` 的「風險」表），headless 環境沒有真實
滑鼠事件，原理上量不出來，必須用眼睛看。

跟 `scripts/probe_omni_ui_shot_panel_widgets.py` 的分工：那支是 headless-
safe 的批次 API 探測（widget 存不存在、型別、數值），本檔只做一件事——搭一
個最小可運行的 overlay（半透明背板 + 一個可拖曳的小圓標記），把每個滑鼠
事件收到的座標印出來，剩下的判斷交給使用者的眼睛和手。

============================================================
使用者該怎麼判斷結果（照這個順序做，每一步都對照 console 輸出）
============================================================

1. **在標記（亮色小圓點）上按住拖曳** → 標記應該跟著游標移動，
   console 會印出一連串 `[probe] mouse_moved ... category=標記`；
   **相機不應該轉動、平移或縮放**。如果相機動了，代表 overlay 沒有真的
   吃掉這個 drag gesture，計畫書列的備援方案要啟動。

2. **在面板背板的空白處（標記以外、半透明底色範圍內）按住拖曳** →
   觀察相機會不會轉。console 會印出 `category=背板空白處`。這一步是為了
   分辨「只有標記本身特殊」還是「整塊 overlay 都能吃事件」——如果第 1 步
   OK 但這一步相機還是轉了，代表要在架構上把整塊背板都掛上滑鼠 catcher
   （計畫書的主方案就是這樣設計的，這裡剛好可以驗證catcher 有沒有確實蓋滿
   整個面板）。

3. **在面板外的 viewport 區域拖曳** → 這裡**不會**印出任何 `[probe]` 訊
   息（我們的 catcher 只蓋住面板範圍，面板外的滑鼠事件本來就不會進到這支
   腳本），相機應該正常轉動/平移/縮放——這是預期的正常行為，用來確認
   catcher 沒有不小心蓋住整個 viewport（那樣的話面板外也會操作不了相機，
   是另一種要修的 bug）。

4. **對照 console 印出的 x/y 跟標記實際位置的關係**：每次事件都會印出
   「原始 x/y」（`set_mouse_pressed_fn` 等 callback 收到的原始參數）與
   「換算後的 widget local x/y」（減去 `screen_position_x/y` 之後）。標記
   一開始畫在面板正中央，如果 local 座標在按住標記時約等於面板尺寸的一
   半，代表 local 座標系原點在面板左上角、跟 `shot_panel_input_mapper.py`
   docstring 假設的「widget local，左上原點，y 往下增加」一致；如果對不
   上，代表座標系假設要修正，這正是要在這裡實測出來的答案之一。

============================================================
相機互動停用開關（可切換旗標，用來比較「有停用」與「沒停用」）
============================================================

`_DISABLE_CAMERA_INTERACTION_WHILE_DRAGGING` 預設 `False`。任務 A 查到
Omniverse 官方文件（`omni.kit.viewport.docs` 的 Camera Manipulator 頁）
確實有 `model.set_ints('disable_tumble'/'disable_look'/'disable_pan'/
'disable_zoom', [1])` 這組 API，但**官方文件沒有寫「怎麼從一個已經存在的
viewport window 拿到這個 model」**——這條路徑本身也是要在這支腳本裡實測
的未知之一。`_try_get_camera_manipulator_model()` 用專案既有慣例（跟
`verify_controller_mode_switch.py` 用 `gc.get_objects()` 撈 `BilliardExtension`
同一招）掃目前存活的物件，找型別名稱以 `CameraManipulator` 結尾、且有
`model.set_ints` 可呼叫的實例——**這個掃法本身未經實測驗證，只是目前找得
到的最合理猜測**，找不到就會印出「找不到，備援方案要改用②③」。

把旗標改成 `True` 後重新跑一次，在標記或背板上按住拖曳時相機應該完全不
動；跟旗標 `False` 時的結果並排比較，就能判斷① 這條停用相機互動的路徑是
不是必要的備援。

跑法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    PYTHONIOENCODING=utf-8 \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/probe_viewport_overlay_drag.py

**不要加 `headless: True`**——這支腳本量的就是滑鼠拖曳跟相機操作的視覺
互動，headless 沒有畫面也沒有真實滑鼠事件，跑起來沒有意義。

也可以在已經開著的 Kit session 裡，透過 Tools 選單
「Tools > Billiard/Probe Viewport Overlay Drag (#115 Stage 0 - GUI)」直接
掛到目前的 viewport（billiard_digital_twin 場景已經在跑的那個），不用另
外開一個 SimulationApp。這樣還能同時看到桌台場景，跟計畫書設想的「疊在
撞球場景上」的真實情境比較接近。
"""

import math
import os
import sys
import traceback

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ui.tool_menu_registry import tool_menu_item

_PROBE_EXT_ID = "billiard_digital_twin.probe_viewport_overlay_drag"

# 面板尺寸（像素，widget local）——寫死常數，不依賴 computed_width（見
# probe_omni_ui_shot_panel_widgets.py 區塊 8 的第一 frame 為 0 疑慮）。
_PANEL_WIDTH_PX = 300.0
_PANEL_HEIGHT_PX = 300.0
_MARKER_RADIUS_PX = 16.0

# 可切換旗標：True 時在按住拖曳期間嘗試停用相機互動，放開後恢復。見檔案
# 級 docstring 的「相機互動停用開關」一節。
_DISABLE_CAMERA_INTERACTION_WHILE_DRAGGING = False


def _try_get_camera_manipulator_model():
    """未經官方文件證實的猜測性做法：掃目前存活物件，找型別名稱以
    `CameraManipulator` 結尾、且有 `model.set_ints` 可呼叫的實例。找不到就
    回傳 `None`（呼叫端要能安靜處理，不能讓整支腳本因此掛掉）。
    """
    import gc

    for obj in gc.get_objects():
        type_name = type(obj).__name__
        if not type_name.endswith("CameraManipulator"):
            continue
        model = getattr(obj, "model", None)
        if model is not None and hasattr(model, "set_ints"):
            print(f"[probe] 找到候選相機操作器：{type_name}（透過 gc.get_objects() 掃描）")
            return model
    return None


def _set_camera_interaction_disabled(model, disabled: bool) -> None:
    if model is None:
        return
    value = [1] if disabled else [0]
    try:
        model.set_ints("disable_tumble", value)
        model.set_ints("disable_look", value)
        model.set_ints("disable_pan", value)
        model.set_ints("disable_zoom", value)
        print(f"[probe] 相機互動停用旗標已設為 disabled={disabled}")
    except Exception:
        print("[probe] 呼叫 model.set_ints() 失敗：")
        traceback.print_exc()


class _OverlayDragProbe:
    """搭一個最小的 viewport overlay：半透明背板 + 一個非 draggable、由
    程式手動定位的圓形標記，掛上三個滑鼠 callback 把收到的座標印出來。

    座標換算 `_to_local()` 沿用計畫書「主方案」的作法：拿 callback 收到的
    螢幕座標，減去 catcher widget 的 `screen_position_x/y`。這個換算本身
    對不對，就是這支腳本要驗證的項目之一（見檔案級 docstring 第 4 點）。
    """

    def __init__(self, viewport_window) -> None:
        import omni.ui as ui
        from omni.ui import color as cl

        self._ui = ui
        self._cl = cl
        self._marker_x = _PANEL_WIDTH_PX / 2.0
        self._marker_y = _PANEL_HEIGHT_PX / 2.0
        self._dragging = False
        self._camera_manipulator_model = None

        frame = viewport_window.get_frame(_PROBE_EXT_ID)
        with frame:
            with ui.VStack():
                ui.Spacer()
                with ui.HStack(height=0):
                    with ui.ZStack(width=_PANEL_WIDTH_PX, height=_PANEL_HEIGHT_PX):
                        ui.Rectangle(
                            style={
                                "background_color": cl(0.08, 0.08, 0.10, 0.55),
                                "border_radius": 6,
                            }
                        )
                        self._marker_placer = ui.Placer(
                            offset_x=self._marker_x - _MARKER_RADIUS_PX,
                            offset_y=self._marker_y - _MARKER_RADIUS_PX,
                        )
                        with self._marker_placer:
                            ui.Circle(
                                radius=_MARKER_RADIUS_PX,
                                style={"background_color": cl(1.0, 0.35, 0.0, 1.0)},
                            )
                        self._catcher = ui.Rectangle(
                            style={"background_color": cl(0.0, 0.0, 0.0, 0.0)}
                        )
                    ui.Spacer()

        self._catcher.set_mouse_pressed_fn(self._on_press)
        self._catcher.set_mouse_moved_fn(self._on_move)
        self._catcher.set_mouse_released_fn(self._on_release)

        print(
            f"[probe] overlay 已建構：面板 {_PANEL_WIDTH_PX:.0f}x{_PANEL_HEIGHT_PX:.0f}px，"
            f"標記初始位置=({self._marker_x:.1f}, {self._marker_y:.1f})，"
            f"疊在 viewport 左下角（VStack+Spacer 相對定位）"
        )
        print(
            "[probe] 請照檔案開頭 docstring 的四個步驟親手拖曳，"
            "每個滑鼠事件都會印出 [probe] mouse_* 開頭的訊息"
        )

    def _to_local(self, x: float, y: float) -> tuple[float, float]:
        return x - self._catcher.screen_position_x, y - self._catcher.screen_position_y

    def _category(self, local_x: float, local_y: float) -> str:
        distance = math.hypot(local_x - self._marker_x, local_y - self._marker_y)
        if distance <= _MARKER_RADIUS_PX:
            return "標記"
        return "背板空白處"

    def _on_press(self, x: float, y: float, button: int, modifier: int) -> None:
        local_x, local_y = self._to_local(x, y)
        category = self._category(local_x, local_y)
        self._dragging = True
        print(
            f"[probe] mouse_pressed 原始=({x:.1f}, {y:.1f}) "
            f"local=({local_x:.1f}, {local_y:.1f}) category={category} button={button}"
        )
        if _DISABLE_CAMERA_INTERACTION_WHILE_DRAGGING:
            if self._camera_manipulator_model is None:
                self._camera_manipulator_model = _try_get_camera_manipulator_model()
                if self._camera_manipulator_model is None:
                    print(
                        "[probe] 找不到相機操作器 model，"
                        "此旗標這次按下無法生效——備援方案要改用②（吃事件的透明層）或③（獨立視窗）"
                    )
            _set_camera_interaction_disabled(self._camera_manipulator_model, disabled=True)

    def _on_move(self, x: float, y: float, modifier: int, is_pressed: bool) -> None:
        if not self._dragging:
            return
        local_x, local_y = self._to_local(x, y)
        clamped_x = min(max(local_x, 0.0), _PANEL_WIDTH_PX)
        clamped_y = min(max(local_y, 0.0), _PANEL_HEIGHT_PX)
        self._marker_x, self._marker_y = clamped_x, clamped_y
        self._marker_placer.offset_x = clamped_x - _MARKER_RADIUS_PX
        self._marker_placer.offset_y = clamped_y - _MARKER_RADIUS_PX
        print(
            f"[probe] mouse_moved 原始=({x:.1f}, {y:.1f}) "
            f"local=({local_x:.1f}, {local_y:.1f}) 標記已移到=({clamped_x:.1f}, {clamped_y:.1f})"
        )

    def _on_release(self, x: float, y: float, button: int, modifier: int) -> None:
        local_x, local_y = self._to_local(x, y)
        print(
            f"[probe] mouse_released 原始=({x:.1f}, {y:.1f}) "
            f"local=({local_x:.1f}, {local_y:.1f})"
        )
        self._dragging = False
        if _DISABLE_CAMERA_INTERACTION_WHILE_DRAGGING:
            _set_camera_interaction_disabled(self._camera_manipulator_model, disabled=False)


# 模組層保留一個參照，避免 overlay 物件被 GC 回收後滑鼠 callback 失效
# （omni.ui 的 callback 只保留弱參照的情況在專案裡沒有先例可查，保守起見
# 顯式持有；探測腳本本來就不追求乾淨的生命週期管理）。
_ACTIVE_PROBE = None


def _run_probe() -> None:
    global _ACTIVE_PROBE
    from omni.kit.viewport.utility import get_active_viewport_window

    viewport_window = get_active_viewport_window()
    if viewport_window is None:
        print(
            "[probe] FAIL：get_active_viewport_window() 回傳 None——"
            "這支腳本需要一個真正的 viewport 視窗，確認目前是不是用 headless 模式在跑"
            "（不應該是，見檔案開頭 docstring 的跑法說明）"
        )
        return

    _ACTIVE_PROBE = _OverlayDragProbe(viewport_window)


@tool_menu_item("Billiard/Probe Viewport Overlay Drag (#115 Stage 0 - GUI)")
def run_from_tool_menu() -> None:
    """在已經啟用 billiard_digital_twin 的 Kit session（有真實 viewport 畫面）
    裡，從 Tools 選單直接掛上這個 overlay。不另開 SimulationApp。"""
    _run_probe()


def _run(simulation_app) -> None:
    import omni.kit.app
    import omni.timeline

    manager = omni.kit.app.get_app().get_extension_manager()
    manager.add_path(_EXT_DIR)
    for _ in range(10):
        simulation_app.update()
    manager.set_extension_enabled_immediate("billiard_digital_twin", True)
    print("[probe] billiard_digital_twin 已啟用，等待場景建立…")
    for _ in range(120):
        simulation_app.update()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(60):
        simulation_app.update()

    _run_probe()

    print("[probe] overlay 已掛上，腳本會保持視窗開著等待手動操作，"
          "拖曳測試完畢後直接關閉視窗即可結束")
    # 保持 App 存活，讓使用者能在 GUI 裡操作；不像 verify_*.py 那樣量完
    # 就結束——這支腳本的驗證動作是使用者的手，不是程式本身。
    while simulation_app.is_running():
        simulation_app.update()


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": False})
    try:
        _run(simulation_app)
    except Exception:
        print("[probe] _run() 拋出例外：")
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        simulation_app.close()
