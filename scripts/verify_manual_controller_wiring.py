"""
scripts/verify_manual_controller_wiring.py — 驗證 #115 階段 3「Extension 接線」
的實際接線（不含 UI；`hud_panel.py` 是階段 4），走真實路徑，不 mock 任何
Isaac Sim / core 元件。

驗證鏈路：
  BilliardExtension._demo_manual_controllers（常駐字典，與 Demo session 同生
  同滅）→ _on_demo_controller_mode_changed()（切到手動模式）→
  TableSession.request_controller_swap() → TableRuntime（下一個 tick 生效，
  順帶強制 full_reset()）→ TableOrchestrator.set_controller()；以及
  BilliardExtension 四個 HUD DI 方法（get/set_manual_shot_parameters、
  request_manual_shot、request_manual_reset、get_manual_shot_status_text）
  → 常駐 ManualController 本身。

驗證項目（對應技術計畫「Headless 驗證」一節）：
  1. `_demo_manual_controllers` 有 Demo 桌的項目
  2. 切手動模式後，orchestrator 實際持有的 controller `is` 字典裡那個物件
     （identity，證明是常駐實例、沒有被重建——重建的話會觸發
     `full_reset()`，見 `billiard_digital_twin.py` `_build_controller_for_
     mode()` 的 docstring）
  3. 不呼叫 `request_manual_shot()`，跑 120 tick：狀態恆為 IDLE、母球世界
     座標零位移（證明不會自動循環，呼應 `test_manual_controller.py` 的
     `test_stays_in_idle_forever_without_shot_request` 錨點，但這裡走的是
     Extension 接線，不是單元測試）
  4. 推參數 `(0.3, -0.8) / 12° / 2.0 / (0.2, -0.1)` → `request_manual_shot()`
     → tick 到 WAITING：母球世界座標 ≈ 桌台 XY + (0.3, -0.8)、全程
     `has_error` 恆為 False
  5. 連續呼叫 `set_manual_shot_parameters()` 20 次不觸發重擺球（記錄某顆
     目標球的位置，推 20 次參數後斷言沒被 teleport 回開球位，證明沒有經過
     controller swap 那條會 `full_reset()` 的路）
  6. 未知 table_id 呼叫五個方法全部不拋例外
  7. `HudPanel`（#115 階段 4）在 headless（拿不到 viewport）環境安靜不建立
     內部 widget、extension 啟動全程不拋例外——這支腳本本身跑到這裡沒有
     崩潰，就已經間接證明了「extension 啟動不拋例外」，這裡額外白箱檢查
     `extension._hud_panel` 存在（`_billiard_init()` 無條件建構它，不因
     headless 就整個跳過）但 `_hud_panel._root_frame is None`（`_create_
     root_frame()` 拿不到 viewport 時安靜回傳 None，見 `hud_panel.py`
     docstring）。保護的是「面板不會害死所有 headless 腳本」這條硬性
     要求——`scripts/` 底下十幾支既有 headless 驗證腳本都要靠這條路徑
     安全

跑法（獨立執行，會自己開一個 headless SimulationApp）：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
    PYTHONIOENCODING=utf-8 \
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/verify_manual_controller_wiring.py

也可透過 Tool Menu Registry（extension/ui/tool_menu_registry.py）在 Kit 主
選單「Tools > Billiard/...」點擊執行——此時 billiard_digital_twin 已經在
目前 Kit session 啟用（Tool Menu 項目本來就是它自己在 on_startup() 註冊
的），直接對現有的 BilliardExtension 實例跑驗證，不另開 SimulationApp、也
不重新 enable 一次 extension。
"""

import gc
import os
import sys
import traceback

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")

# 必須跟 extension/billiard_digital_twin/billiard_digital_twin.py 用同一種 import
# 路徑（把 extension/ 本身加進 sys.path，import 成 "ui.tool_menu_registry"），
# 否則同一支檔案會被當成兩個不同模組載入，各自有獨立的 _REGISTERED_TOOLS 清單，
# decorator 註冊的內容跟 discover_and_register 讀到的會對不上（見
# measure_swing_speed.py 同一段說明）。
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ui.tool_menu_registry import tool_menu_item

_IDLE_HOLD_TICKS = 120  # 項目 3：不按按鈕的情況下跑幾個 tick 確認恆為 IDLE
_MAX_TICKS_TO_IDLE = 8000  # controller swap 觸發的 full_reset() 收斂回 IDLE 的預算（手臂歸位）
_MAX_TICKS_TO_WAITING = 30000  # 項目 4：AIM（RMPflow 可能上萬 tick）+ STRIKE 給足預算，
# 數量級參照 scripts/test_ur10e_table_flat.py 的 _MAX_STEPS_PER_AIM_ACTION=20000。
_MAX_TICKS_BACK_TO_IDLE = 8000  # 項目 4 打完之後，等 WAITING -> RESET -> IDLE 走完才能開始項目 5
_PARAMETER_PUSH_COUNT = 20  # 項目 5：連續推參數次數
_WORLD_POSITION_TOLERANCE_M = 0.05  # STRIKING -> WAITING 之間母球可能已經開始滾動，容許誤差
_STATIC_POSITION_TOLERANCE_M = 1e-3  # 球理論上完全靜止，容許浮點/物理求解器的極小抖動


def _find_extension():
    return next(
        (o for o in gc.get_objects() if type(o).__name__ == "BilliardExtension"),
        None,
    )


def _wait_for_state(session, target_state_name: str, max_ticks: int, app) -> int | None:
    """跑到 `max_ticks` 為止，回傳達到目標狀態所花的 tick 數；逾時回傳 None。"""
    for step in range(max_ticks):
        app.update()
        if session.get_current_state().name == target_state_name:
            return step + 1
    return None


def _verify(extension) -> bool:
    import omni.kit.app

    from core.models.manual_shot_parameters import ManualShotParameters

    app = omni.kit.app.get_app()

    demo_sessions = extension._demo_sessions
    if not demo_sessions:
        print("[verify] FAIL：沒有任何 Demo 桌 session（Demo toggle 沒開？）")
        return False
    session = demo_sessions[0]
    table_id = session.get_table_id()
    print(f"[verify] table_id={table_id}")

    # 項目 1：常駐字典有這張 Demo 桌的項目
    manual_controller = extension._demo_manual_controllers.get(table_id)
    has_entry = manual_controller is not None
    print(f"[verify] _demo_manual_controllers 有 {table_id} 的項目：{has_entry}")
    if not has_entry:
        print("[verify] FAIL：缺少常駐 ManualController，後續驗證無法進行")
        return False

    def _current_controller():
        # 白箱查看，跟 verify_controller_mode_switch.py 同一條路徑：
        # TableSession -> TableRuntime -> TableOrchestrator._script_controller
        return session._runtime._orchestrator._script_controller

    # 切到手動模式（走跟 DebugMenu 完全相同的呼叫路徑）
    extension._on_demo_controller_mode_changed(table_id, False)
    app.update()
    app.update()

    # 項目 2：orchestrator 實際持有的 controller 必須 is 常駐字典裡那個物件
    is_same_instance = _current_controller() is manual_controller
    print(f"[verify] orchestrator 持有的 controller 與常駐字典裡的實例相同（identity）：{is_same_instance}")

    # controller swap 會強制 full_reset()（重擺球＋手臂歸位），要等它收斂回
    # IDLE，項目 3 的「恆為 IDLE」檢查才有意義（否則一開始量到的會是 RESET）。
    ticks_to_idle = _wait_for_state(session, "IDLE", _MAX_TICKS_TO_IDLE, app)
    reached_idle_after_switch = ticks_to_idle is not None
    print(f"[verify] 切手動模式後在 {_MAX_TICKS_TO_IDLE} tick 預算內回到 IDLE："
          f"{reached_idle_after_switch}（花費 {ticks_to_idle} tick）")

    # 項目 3：不呼叫 request_manual_shot()，跑 120 tick，狀態恆為 IDLE、母球零位移
    cue_ball_prim_path = session._table.get_table_ball_set().get_ball_prim_paths()[0]
    position_before_idle_hold = list(extension._rigid_body_api.get_position(cue_ball_prim_path))
    states_seen: set[str] = set()
    for _ in range(_IDLE_HOLD_TICKS):
        app.update()
        states_seen.add(session.get_current_state().name)
    position_after_idle_hold = list(extension._rigid_body_api.get_position(cue_ball_prim_path))
    stayed_idle = states_seen == {"IDLE"}
    stayed_still = all(
        abs(a - b) < _STATIC_POSITION_TOLERANCE_M
        for a, b in zip(position_before_idle_hold, position_after_idle_hold)
    )
    print(f"[verify] 不按按鈕跑 {_IDLE_HOLD_TICKS} tick 狀態恆為 IDLE：{stayed_idle}"
          f"（觀察到的狀態集合={states_seen}）")
    print(f"[verify] 不按按鈕跑 {_IDLE_HOLD_TICKS} tick 母球世界座標零位移：{stayed_still}"
          f"（before={position_before_idle_hold}，after={position_after_idle_hold}）")

    # 項目 4：推參數 + 擊球 -> tick 到 WAITING
    table_x, table_y = session._table.get_table_ball_set().get_table_x_y()
    shot_parameters = ManualShotParameters(
        cue_ball_placement=(0.3, -0.8),
        shot_angle=12.0,
        cue_ball_speed=2.0,
        position_offset=(0.2, -0.1),
    )
    extension.set_manual_shot_parameters(table_id, shot_parameters)
    extension.request_manual_shot(table_id)

    reached_waiting = False
    had_error = False
    steps_to_waiting = 0
    for step in range(_MAX_TICKS_TO_WAITING):
        app.update()
        steps_to_waiting = step + 1
        observation = session.get_last_observation()
        if observation is not None and observation.has_error:
            had_error = True
            break
        if session.get_current_state().name == "WAITING":
            reached_waiting = True
            break
    print(f"[verify] 推參數＋擊球後在 {steps_to_waiting} tick 內到達 WAITING：{reached_waiting}"
          f"（目前狀態={session.get_current_state().name}）")
    print(f"[verify] 全程 has_error 恆為 False：{not had_error}")

    placement_ok = False
    if reached_waiting:
        observation = session.get_last_observation()
        expected_xy = [table_x + 0.3, table_y + (-0.8)]
        actual_xy = list(observation.cue_ball_position[:2])
        placement_ok = all(
            abs(a - b) < _WORLD_POSITION_TOLERANCE_M for a, b in zip(actual_xy, expected_xy)
        )
        print(f"[verify] 母球世界座標 ≈ 桌台 XY + (0.3, -0.8)：{placement_ok}"
              f"（預期≈{expected_xy}，實際={actual_xy}）")
    else:
        print("[verify] 未到達 WAITING，跳過母球擺位斷言")

    # 項目 5：連續推 20 次參數不觸發重擺球。要先等這一局自然走完
    # WAITING -> RESET -> IDLE（正常擊球流程本身就會 _reset_balls()，跟
    # controller swap 觸發的 full_reset() 是兩回事），再開始記錄基準位置，
    # 否則量到的位移會是這一局本身造成的，不是要驗的重擺球問題。
    ticks_back_to_idle = _wait_for_state(session, "IDLE", _MAX_TICKS_BACK_TO_IDLE, app)
    back_to_idle = ticks_back_to_idle is not None
    print(f"[verify] 這一局結束後在 {_MAX_TICKS_BACK_TO_IDLE} tick 預算內回到 IDLE："
          f"{back_to_idle}（花費 {ticks_back_to_idle} tick）")

    other_ball_prim_path = session._table.get_table_ball_set().get_ball_prim_paths()[1]
    position_before_pushes = list(extension._rigid_body_api.get_position(other_ball_prim_path))
    for i in range(_PARAMETER_PUSH_COUNT):
        extension.set_manual_shot_parameters(
            table_id,
            ManualShotParameters(
                cue_ball_placement=(0.3, -0.8),
                shot_angle=float(i % 10),
                cue_ball_speed=1.0 + (i % 5) * 0.1,
                position_offset=(0.0, 0.0),
            ),
        )
        app.update()
    position_after_pushes = list(extension._rigid_body_api.get_position(other_ball_prim_path))
    no_rerack = all(
        abs(a - b) < _STATIC_POSITION_TOLERANCE_M
        for a, b in zip(position_before_pushes, position_after_pushes)
    )
    print(f"[verify] 連續推 {_PARAMETER_PUSH_COUNT} 次參數不觸發重擺球：{no_rerack}"
          f"（before={position_before_pushes}，after={position_after_pushes}）")

    # 項目 7：HudPanel 在拿不到 viewport 的環境安靜不建立內部 widget、
    # extension 啟動全程不拋例外。這支腳本執行到這裡本身沒有崩潰，就已經
    # 證明了「extension 啟動不拋例外」；這裡額外白箱檢查 _billiard_init()
    # 無條件建構 self._hud_panel（不是 None），而 HudPanel._create_root_
    # frame() 只有在拿得到 viewport 時才會回傳非 None 的 Frame——用實際
    # 查到的 viewport 可用性反推期望值，而不是寫死「一定是 headless」：
    # 這支腳本主要跑法是 headless（見檔案開頭「跑法」），但也支援透過
    # Tool Menu 在已有畫面的 GUI session 裡執行（見同一段說明），兩種情況
    # 這裡都要給出正確答案，不能只驗證其中一種。
    try:
        from omni.kit.viewport.utility import get_active_viewport_window

        viewport_available = get_active_viewport_window() is not None
    except ImportError:
        viewport_available = False

    hud_panel = getattr(extension, "_hud_panel", None)
    hud_panel_exists = hud_panel is not None
    hud_panel_has_root_frame = hud_panel_exists and hud_panel._root_frame is not None
    hud_panel_consistent_with_viewport = hud_panel_exists and (
        hud_panel_has_root_frame == viewport_available
    )
    print(f"[verify] extension._hud_panel 存在（_billiard_init() 有建構它）：{hud_panel_exists}")
    print(f"[verify] 目前環境拿得到 viewport：{viewport_available}"
          f"（headless 跑法下應為 False）")
    print(f"[verify] HudPanel 的 _root_frame 是否建立與 viewport 可用性一致："
          f"{hud_panel_consistent_with_viewport}"
          f"（headless 下應該是「拿不到 viewport → 安靜不建立 → _root_frame is None」）")

    # 項目 6：未知 table_id 呼叫五個方法全部不拋例外
    unknown_table_id = "/World/NotATable"
    no_crash_on_unknown_table = True
    try:
        extension.get_manual_shot_parameters(unknown_table_id)
        extension.set_manual_shot_parameters(unknown_table_id, ManualShotParameters.default())
        extension.request_manual_shot(unknown_table_id)
        extension.request_manual_reset(unknown_table_id)
        extension.get_manual_shot_status_text(unknown_table_id)
    except Exception:
        no_crash_on_unknown_table = False
        traceback.print_exc()
    print(f"[verify] 未知 table_id 呼叫五個 HUD DI 方法皆不拋例外：{no_crash_on_unknown_table}")

    all_pass = (
        has_entry
        and is_same_instance
        and reached_idle_after_switch
        and stayed_idle
        and stayed_still
        and reached_waiting
        and not had_error
        and placement_ok
        and back_to_idle
        and no_rerack
        and no_crash_on_unknown_table
        and hud_panel_exists
        and hud_panel_consistent_with_viewport
    )
    print(f"[verify] {'PASS' if all_pass else 'FAIL'}：ManualController 常駐接線"
          f"{'運作正常' if all_pass else '有問題，見上方個別項目'}")
    return all_pass


@tool_menu_item("Billiard/Verify Manual Controller Wiring")
def run_from_tool_menu() -> None:
    """在已經啟用 billiard_digital_twin 的 Kit session 裡，從 Tools 選單直接
    對現有的 BilliardExtension 實例跑驗證。不另開 SimulationApp、也不重新
    enable 一次 extension——那兩件事都已經由目前這個 session 做過了（Tool
    Menu 項目本來就是 billiard_digital_twin 自己在 on_startup() 註冊的，
    能點到這個選項就代表 extension 已經在跑）。"""
    extension = _find_extension()
    if extension is None:
        print("[verify] FAIL：找不到 BilliardExtension 實例（billiard_digital_twin 沒有啟用？）")
        return
    _verify(extension)


def _run(simulation_app) -> None:
    import omni.kit.app
    import omni.timeline

    manager = omni.kit.app.get_app().get_extension_manager()
    manager.add_path(_EXT_DIR)
    for _ in range(10):
        simulation_app.update()
    manager.set_extension_enabled_immediate("billiard_digital_twin", True)
    print("[verify] billiard_digital_twin 已啟用，等待場景建立…")
    for _ in range(120):
        simulation_app.update()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(60):
        simulation_app.update()

    extension = _find_extension()
    if extension is None:
        print("[verify] FAIL：找不到 BilliardExtension 實例")
        return

    _verify(extension)


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run(simulation_app)
    except Exception:
        print("[verify] _run() 拋出例外：")
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        simulation_app.close()
