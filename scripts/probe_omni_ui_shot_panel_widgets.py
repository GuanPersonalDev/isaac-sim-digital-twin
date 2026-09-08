"""
scripts/probe_omni_ui_shot_panel_widgets.py — #115 階段 0 API spike：headless-safe
地探測 `hud_panel.py`（階段 4）會用到的 omni.ui / omni.kit.viewport.utility API，
把「文件查不到、只能實測才知道」的問題逐項問出實際答案。

跟 `scripts/probe_viewport_overlay_drag.py` 的分工：
    本檔（headless-safe）  — 能在 `SimulationApp({"headless": True})` 下跑完的
        項目：widget 是否存在、`Placer.offset_x` 型別與讀寫往返、
        `Placer(draggable=True, drag_axis=...)` 能不能建構、`omni.ui.color`
        各種呼叫形式的實際數值、`get_active_viewport_window()`/`get_frame()`
        的型別與可用性、2D widget 能不能放進 `get_frame()`、
        `computed_width`/`screen_position_x` 在第幾個 `app.update()` 後才有
        非零值、滑鼠 callback 能不能掛上。
    `probe_viewport_overlay_drag.py`（**不是** headless，需要在 GUI 下手動
        拖曳）— 驗證 overlay 上的拖曳會不會被 viewport 相機操作吃掉，這是
        headless 環境原理上量不出來的東西（沒有真實滑鼠事件），本檔只能
        驗證「callback 掛得上去」，掛上去之後 x/y 座標系與是否會跟相機互動
        衝突一律留給那支腳本。

輸出格式：每一項獨立 try/except 包起來，某一項不支援或炸掉不能讓整支腳本
掛掉；每一行前綴 `[probe]`，方便使用者把整份 console 輸出貼回來對答案。

**這個環境沒有安裝 Isaac Sim**（只有未解壓的安裝檔，`import isaacsim` 會
`ModuleNotFoundError`），本檔在撰寫當下**沒有被執行過**，所有輸出行為都是
根據 omni.ui / omni.kit.viewport.utility 官方文件與程式碼慣例撰寫，真正的
數值要等使用者在有 Isaac Sim 的環境跑過才知道——這正是這支探測腳本存在的
意義。

跑法（跟 `verify_manual_controller_wiring.py` 同一種雙模式）：

  1) 獨立執行，自己開一個 headless SimulationApp：
     ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
     PYTHONIOENCODING=utf-8 \\
     "C:/Other/OmniverseProjects/isaac/python.bat" scripts/probe_omni_ui_shot_panel_widgets.py

     ⚠️ 這是獨立安裝的 Isaac Sim（`python.bat`，不是 pip venv 的
     `Scripts/python.exe`）——路徑因環境而異，2026-09-08 實測確認的路徑是
     `C:/Other/OmniverseProjects/isaac`。

  2) 透過 Tool Menu Registry：billiard_digital_twin 啟用後，Kit 主選單
     「Tools > Billiard/Probe Omni UI Shot Panel Widgets」直接點擊執行，
     沿用目前 session 已經開著的 Kit（不另開 SimulationApp）。

兩種跑法都是 headless-safe，但**用真正的 GUI 視窗（不加 `headless: True`）
跑一次的參考價值更高**——`get_active_viewport_window()` 在純 headless 模式
下大機率回傳 `None`（見腳本內第 5 項），那樣的話第 6、7 項（`get_frame()`
相關）會被安靜跳過，測不到最關鍵的「2D widget 能不能放進 get_frame()」。
"""

import os
import sys
import traceback

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from ui.tool_menu_registry import tool_menu_item
except ImportError:
    # 獨立執行（python.bat scripts/...py）時，這一行在 SimulationApp 建構
    # 之前就會被執行到——Kit 的擴充功能系統（含 omni.kit.menu.utils，
    # tool_menu_registry.py 依賴它）此時還沒載入，import 不到。實測踩過：
    # `ModuleNotFoundError: No module named 'omni.kit.menu'`。獨立執行模式
    # 本來就不需要 Tool Menu 註冊（見檔案最下方 __main__ 區塊，直接呼叫
    # `_run()`），這裡給一個 no-op decorator 讓模組照常載入完畢即可；只有
    # 「透過 Tool Menu Registry 點擊執行」那個模式（discover_and_register()
    # 在 BilliardExtension.on_startup() 之後才呼叫，那時 Kit 早就跑起來
    # 了）才用得到真正的 tool_menu_item，那個情境下這個 try 會成功。
    def tool_menu_item(menu_path: str):
        def decorator(func):
            return func

        return decorator

# 內部固定的 ext_id key：get_frame() 只把它當成 dict key 用來識別同一個
# frame，不要求一定是真的 extension id 字串，探測不依賴 BilliardExtension
# 是否已啟用。
_PROBE_EXT_ID = "billiard_digital_twin.probe_omni_ui_shot_panel_widgets"

_WIDGET_CLASS_NAMES = [
    "Frame",
    "VStack",
    "HStack",
    "ZStack",
    "Placer",
    "Rectangle",
    "Circle",
    "Button",
    "Label",
    "FloatField",
    "StringField",
    "ComboBox",
    "SimpleFloatModel",
    "SimpleStringModel",
    "AbstractValueModel",
    "CollapsableFrame",
    "Axis",
    "color",
]


def _probe(label: str, fn) -> None:
    """統一的探測外殼：印出 label，呼叫 fn()，把回傳值（若非 None）印出來；
    fn() 拋例外時印出完整 traceback 但不中斷後續探測項目。"""
    print(f"[probe] {label}")
    try:
        result = fn()
        if result is not None:
            print(f"[probe]   -> {result}")
    except Exception:
        print("[probe]   -> 失敗，例外如下：")
        for line in traceback.format_exc().splitlines():
            print(f"[probe]        {line}")


def _extract_color_int(color_obj) -> int:
    """`omni.ui.color` 物件轉成 32-bit 整數的嘗試——文件沒有明確給出
    Python 端要用哪個轉換方式，依序嘗試 `int()`、`.value`、`float()`
    （四捨五入前的 packed 表示法有時是用 float 承載），第一個成功的就回傳。
    """
    try:
        return int(color_obj)
    except Exception:
        pass
    if hasattr(color_obj, "value"):
        return int(color_obj.value)
    return int(float(color_obj))


def _run_probes() -> None:
    import omni.kit.app
    import omni.ui as ui
    from omni.ui import color as cl

    print("[probe] ========== 區塊 1：widget / model 類別是否存在 ==========")
    for name in _WIDGET_CLASS_NAMES:
        _probe(f"hasattr(omni.ui, {name!r})", lambda n=name: hasattr(ui, n))

    print("[probe] ========== 區塊 2：Placer.offset_x 型別與讀寫往返 ==========")

    # ui.Placer 必須在一個 build context（Window/Frame 的 with 區塊）裡建構，
    # 這裡開一個不可見的浮動 Window 純粹當容器，不依賴 viewport。
    probe_window = ui.Window(
        "Billiard HUD Probe (headless-safe)", width=400, height=400, visible=False
    )

    placer_holder = {}

    def _build_placer_with_offset() -> str:
        with probe_window.frame:
            with ui.ZStack():
                placer = ui.Placer(offset_x=10, offset_y=20)
                with placer:
                    ui.Rectangle(width=20, height=20)
        placer_holder["placer"] = placer
        return f"建構成功；type(offset_x)={type(placer.offset_x)}，offset_x={placer.offset_x!r}"

    _probe("建構 ui.Placer(offset_x=10, offset_y=20) 並讀回 offset_x 型別/數值", _build_placer_with_offset)

    def _roundtrip_offset_x() -> str:
        placer = placer_holder.get("placer")
        if placer is None:
            return "跳過（上一步建構失敗，沒有 placer 可用）"
        before = placer.offset_x
        placer.offset_x = 25.5
        after = placer.offset_x
        return (
            f"寫入前 type={type(before)} value={before!r}；"
            f"寫入 25.5 (float) 後 type={type(after)} value={after!r}"
        )

    _probe("寫入 placer.offset_x = 25.5（純 float）後讀回，確認讀寫往返一致", _roundtrip_offset_x)

    print("[probe] ========== 區塊 3：Placer(draggable=True, drag_axis=...) ==========")

    def _build_draggable_placer() -> str:
        if not hasattr(ui, "Axis"):
            return "跳過：omni.ui 沒有 Axis 這個 enum（見區塊 1 結果）"
        with probe_window.frame:
            with ui.ZStack():
                draggable_placer = ui.Placer(
                    offset_x=0, offset_y=0, draggable=True, drag_axis=ui.Axis.XY
                )
                with draggable_placer:
                    ui.Circle(radius=10)
        return "建構成功：Placer(draggable=True, drag_axis=ui.Axis.XY) 在這個 Kit 版本可用"

    _probe("建構 ui.Placer(draggable=True, drag_axis=ui.Axis.XY)", _build_draggable_placer)

    def _set_offset_x_changed_fn() -> str:
        if "placer" not in placer_holder:
            return "跳過（沒有可用的 placer）"
        placer_holder["placer"].set_offset_x_changed_fn(lambda new_offset: None)
        return "set_offset_x_changed_fn() 掛載成功"

    _probe("Placer.set_offset_x_changed_fn() 是否可掛載", _set_offset_x_changed_fn)

    print("[probe] ========== 區塊 4：omni.ui.color 各種呼叫形式 ==========")

    color_call_forms = [
        ("cl(1.0, 0.0, 0.0, 1.0) — 純紅、不透明", lambda: cl(1.0, 0.0, 0.0, 1.0)),
        ("cl(0.0, 1.0, 0.0, 1.0) — 純綠、不透明", lambda: cl(0.0, 1.0, 0.0, 1.0)),
        ("cl(0.0, 0.0, 1.0, 1.0) — 純藍、不透明", lambda: cl(0.0, 0.0, 1.0, 1.0)),
        ("cl(0.5) — 單一 float，灰階", lambda: cl(0.5)),
        ("cl(0.0, 0.0, 0.0, 0.0) — 全透明黑", lambda: cl(0.0, 0.0, 0.0, 0.0)),
        ('cl("#FF0000") — 帶 # 的十六進位字串', lambda: cl("#FF0000")),
        ('cl("1F2123") — 不帶 # 的十六進位字串（RGB）', lambda: cl("1F2123")),
        ('cl("CCCCCCCC") — 8 碼含 Alpha 的十六進位字串', lambda: cl("CCCCCCCC")),
    ]
    for description, factory in color_call_forms:
        def _do(factory=factory, description=description):
            color_obj = factory()
            try:
                as_int = _extract_color_int(color_obj)
                return f"type={type(color_obj)}，repr={color_obj!r}，轉成整數後 hex={hex(as_int)}"
            except Exception as exc:
                return f"type={type(color_obj)}，repr={color_obj!r}，無法轉成整數（{exc!r}）"

        _probe(description, _do)

    def _verify_abgr_theory() -> str:
        """官方文件（`OMNIVERSE KIT UI STYLE BEST PRACTICE`）明講：直接寫十六進位
        整數字面量（如 0xFF23211F）是 ABGR 順序，`cl()` 才是符合直覺的 ARGB
        順序。用純紅不透明色反推：cl(r=1,g=0,b=0,a=1) 若真的照 ABGR 存放，轉成
        整數後應該等於 0xFF0000FF（A=FF, B=00, G=00, R=FF，由高位到低位）。
        這裡直接算出來讓使用者一眼比對，不用自己心算。
        """
        pure_red_opaque = cl(1.0, 0.0, 0.0, 1.0)
        actual = _extract_color_int(pure_red_opaque)
        expected_if_abgr = 0xFF0000FF
        matches = actual == expected_if_abgr
        return (
            f"cl(1,0,0,1) 轉整數後 = {hex(actual)}；"
            f"若官方文件的 ABGR 說法成立應該等於 {hex(expected_if_abgr)}；"
            f"是否吻合：{matches}"
        )

    _probe("反推驗證：cl(r,g,b,a) 的內部整數表示法是否真的是 ABGR", _verify_abgr_theory)

    def _raw_hex_literal_on_style() -> str:
        rect = ui.Rectangle(width=10, height=10)
        rect.set_style({"background_color": 0xFF0000FF})
        return "0xFF0000FF 直接指派給 style['background_color'] 沒有拋例外（純紅不透明，依 ABGR 理論）"

    _probe("直接把十六進位整數字面量指派給 style['background_color']", _raw_hex_literal_on_style)

    print("[probe] ========== 區塊 5：get_active_viewport_window() ==========")

    viewport_window_holder = {}

    def _get_viewport_window() -> str:
        from omni.kit.viewport.utility import get_active_viewport_window

        window = get_active_viewport_window()
        viewport_window_holder["window"] = window
        if window is None:
            return "回傳 None（headless 環境下的預期行為，見計畫文件）"
        return f"回傳非 None：type={type(window)}"

    _probe("get_active_viewport_window()", _get_viewport_window)

    print("[probe] ========== 區塊 6：viewport_window.get_frame(ext_id) ==========")

    frame_holder = {}

    def _get_frame_first_call() -> str:
        window = viewport_window_holder.get("window")
        if window is None:
            return "跳過：上一步拿到的 viewport window 是 None（headless 環境預期行為）"
        frame = window.get_frame(_PROBE_EXT_ID)
        frame_holder["frame"] = frame
        return f"type={type(frame)}（預期是一般 omni.ui.Frame，不是 sc.Frame）"

    _probe("window.get_frame(ext_id) 第一次呼叫", _get_frame_first_call)

    def _get_frame_second_call_identity() -> str:
        window = viewport_window_holder.get("window")
        first_frame = frame_holder.get("frame")
        if window is None or first_frame is None:
            return "跳過：沒有可用的 viewport window / frame"
        second_frame = window.get_frame(_PROBE_EXT_ID)
        return f"同一個 ext_id 重複呼叫是否回傳同一個物件（identity）：{second_frame is first_frame}"

    _probe("window.get_frame(ext_id) 第二次呼叫，驗證同一個 ext_id 是否回傳同一個 Frame", _get_frame_second_call_identity)

    print("[probe] ========== 區塊 7：在 get_frame() 裡放一般 2D widget ==========")

    def _build_2d_widget_in_viewport_frame() -> str:
        frame = frame_holder.get("frame")
        if frame is None:
            return "跳過：沒有可用的 viewport frame（headless 環境預期行為）"
        with frame:
            with ui.VStack():
                ui.Spacer()
                with ui.HStack(height=0):
                    with ui.ZStack(width=330, height=200):
                        ui.Rectangle(style={"background_color": cl(0.08, 0.08, 0.10, 0.55), "border_radius": 6})
                        with ui.VStack(spacing=6):
                            ui.Label("Shot Control (probe)")
                            ui.Button("Fire (probe)", clicked_fn=lambda: None)
                    ui.Spacer()
        return "在 viewport_window.get_frame(ext_id) 裡建構 VStack+Rectangle+Button 成功，沒有拋例外"

    _probe("在 get_frame() 回傳的 Frame 裡建構 VStack + Rectangle + Button", _build_2d_widget_in_viewport_frame)

    print("[probe] ========== 區塊 8：computed_width / screen_position_x 的更新時機 ==========")

    app = omni.kit.app.get_app()
    sized_rect_holder = {}

    def _build_sized_rect() -> str:
        with probe_window.frame:
            pass
        # 用獨立的第二個 Window，跟區塊 2/3 共用的 probe_window 分開，避免
        # frame 內容互相覆蓋。
        sized_window = ui.Window(
            "Billiard HUD Probe - Sized Rect", width=200, height=200, visible=False
        )
        with sized_window.frame:
            rect = ui.Rectangle(width=50, height=50)
        sized_rect_holder["rect"] = rect
        sized_rect_holder["window"] = sized_window
        return "已建構固定 50x50 的 Rectangle，準備量測 computed_width / screen_position_x"

    _probe("建構待量測的固定尺寸 Rectangle", _build_sized_rect)

    def _measure_after_updates(update_count: int) -> str:
        rect = sized_rect_holder.get("rect")
        if rect is None:
            return "跳過：沒有可用的 rect"
        for _ in range(update_count):
            app.update()
        return (
            f"第 {update_count} 個 app.update() 後："
            f"computed_width={rect.computed_width}, computed_height={rect.computed_height}, "
            f"screen_position_x={rect.screen_position_x}, screen_position_y={rect.screen_position_y}"
        )

    # 累計呼叫次數：1、再 1（累計 2）、再 3（累計 5）、再 5（累計 10）。
    _probe("量測時機 #1（累計 1 個 app.update()）", lambda: _measure_after_updates(1))
    _probe("量測時機 #2（累計 2 個 app.update()）", lambda: _measure_after_updates(1))
    _probe("量測時機 #3（累計 5 個 app.update()）", lambda: _measure_after_updates(3))
    _probe("量測時機 #4（累計 10 個 app.update()）", lambda: _measure_after_updates(5))

    print("[probe] ========== 區塊 9：滑鼠 callback 掛載（不驗證真實事件） ==========")

    def _attach_mouse_callbacks() -> str:
        rect = sized_rect_holder.get("rect")
        if rect is None:
            return "跳過：沒有可用的 rect"
        received = []
        rect.set_mouse_pressed_fn(lambda x, y, button, modifier: received.append(("pressed", x, y)))
        rect.set_mouse_moved_fn(lambda x, y, modifier, is_pressed: received.append(("moved", x, y)))
        rect.set_mouse_released_fn(lambda x, y, button, modifier: received.append(("released", x, y)))
        return (
            "set_mouse_pressed_fn / set_mouse_moved_fn / set_mouse_released_fn 三個都掛載成功；"
            "headless 環境無法產生真實滑鼠事件，這裡只證明「掛得上去」，"
            "x/y 是螢幕座標還是 widget local 座標、以及會不會被 viewport 相機操作吃掉，"
            "需要在 GUI 下用 scripts/probe_viewport_overlay_drag.py 親手拖曳才能驗證"
        )

    _probe("在 Rectangle 上掛載三個滑鼠事件 callback", _attach_mouse_callbacks)

    print("[probe] ========== 區塊 10：FloatField / SimpleFloatModel ==========")

    def _build_float_field_with_simple_model() -> str:
        if not hasattr(ui, "FloatField") or not hasattr(ui, "SimpleFloatModel"):
            return "跳過：FloatField 或 SimpleFloatModel 不存在（見區塊 1 結果），面板要退回 StringField + float() 解析"
        model = ui.SimpleFloatModel(1.5)
        with probe_window.frame:
            ui.FloatField(model=model)
        as_float = None
        for accessor_name in ("get_value_as_float", "as_float"):
            if hasattr(model, accessor_name):
                as_float = getattr(model, accessor_name)()
                break
        model.set_value(2.75)
        after_set = None
        for accessor_name in ("get_value_as_float", "as_float"):
            if hasattr(model, accessor_name):
                after_set = getattr(model, accessor_name)()
                break
        return (
            f"SimpleFloatModel(1.5) 建構成功，FloatField(model=model) 建構成功；"
            f"初始讀值={as_float}；set_value(2.75) 後讀值={after_set}"
        )

    _probe("建構 ui.FloatField(model=ui.SimpleFloatModel(1.5))，讀寫往返", _build_float_field_with_simple_model)

    print("[probe] ========== 區塊 11：ui.Circle 在 Placer 內的 size_policy / alignment ==========")

    def _circle_in_placer_size_policy_alignment() -> str:
        notes = []
        with probe_window.frame:
            with ui.ZStack():
                with ui.Placer(offset_x=0, offset_y=0):
                    circle = ui.Circle(radius=20)
        notes.append("baseline：ui.Placer 內放 ui.Circle(radius=20) 建構成功")

        for enum_name in ("Alignment", "FillPolicy", "CircleSizePolicy"):
            notes.append(f"hasattr(omni.ui, {enum_name!r}) = {hasattr(ui, enum_name)}")

        if hasattr(ui, "Alignment"):
            try:
                circle.alignment = ui.Alignment.CENTER
                notes.append("circle.alignment = ui.Alignment.CENTER 賦值成功")
            except Exception as exc:
                notes.append(f"circle.alignment 賦值失敗：{exc!r}")

        for enum_name in ("FillPolicy", "CircleSizePolicy"):
            if hasattr(ui, enum_name):
                enum_cls = getattr(ui, enum_name)
                try:
                    first_member = list(enum_cls.__members__.values())[0]
                    circle.size_policy = first_member
                    notes.append(f"circle.size_policy = {enum_name}.{first_member.name} 賦值成功")
                except Exception as exc:
                    notes.append(f"circle.size_policy（用 {enum_name}）賦值失敗：{exc!r}")

        return "；".join(notes)

    _probe("ui.Circle 建構於 Placer 內，並嘗試設定 alignment / size_policy", _circle_in_placer_size_policy_alignment)

    print("[probe] 全部區塊跑完。請把以上輸出整份貼回去，用來把「文件查不到」的項目補齊。")


@tool_menu_item("Billiard/Probe Omni UI Shot Panel Widgets (#115 Stage 0)")
def run_from_tool_menu() -> None:
    """在已經啟用 billiard_digital_twin 的 Kit session 裡，從 Tools 選單直接
    跑這支探測腳本。不另開 SimulationApp、也不重新 enable 一次 extension。"""
    _run_probes()


def _run(simulation_app) -> None:
    _run_probes()


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run(simulation_app)
    except Exception:
        print("[probe] _run() 拋出例外：")
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        simulation_app.close()
