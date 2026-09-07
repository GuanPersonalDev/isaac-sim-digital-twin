"""
scripts/verify_controller_mode_switch.py — 驗證 Debug Menu 切換手臂操作策略
的實際接線（不只是語法對，是真的能換、換完狀態機能正常跑）。

驗證鏈路：DebugMenu 的 callback 簽章 → BilliardExtension.
_on_demo_controller_mode_changed() → TableSession.request_controller_swap()
→ TableRuntime（下一個 tick 生效）→ TableOrchestrator.set_controller()。

跑法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
    PYTHONIOENCODING=utf-8 \
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/verify_controller_mode_switch.py
"""

import os
import sys
import traceback

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _run(simulation_app) -> None:
    import gc

    import omni.kit.app
    import omni.timeline

    from core.controllers.model_controller import ModelController
    from core.controllers.script_controller import ScriptController

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

    extension = next(
        (o for o in gc.get_objects() if type(o).__name__ == "BilliardExtension"),
        None,
    )
    if extension is None:
        print("[verify] FAIL：找不到 BilliardExtension 實例")
        return

    demo_sessions = extension._demo_sessions
    if not demo_sessions:
        print("[verify] FAIL：沒有任何 Demo 桌 session（Demo toggle 沒開？）")
        return
    table_id = demo_sessions[0].get_table_id()
    print(f"[verify] table_id={table_id}")

    def _current_controller():
        # 白箱查看：TableSession -> TableRuntime -> TableOrchestrator._script_controller
        return demo_sessions[0]._runtime._orchestrator._script_controller

    initial = type(_current_controller()).__name__
    print(f"[verify] 初始 controller={initial}")
    ok = initial == "ModelController"
    print(f"[verify] 初始應為 ModelController：{ok}")

    # 切到 Script 模式（走跟 DebugMenu 完全相同的呼叫路徑）
    extension._on_demo_controller_mode_changed(table_id, False)
    before_apply = type(_current_controller()).__name__
    same_before_tick = before_apply == initial
    print(f"[verify] 呼叫當下不應立刻套用（要等下一個 tick）：{same_before_tick}"
          f"（目前={before_apply}）")

    for _ in range(5):
        simulation_app.update()

    after_switch = type(_current_controller()).__name__
    switched_to_script = after_switch == "ScriptController"
    print(f"[verify] 幾個 tick 後應換成 ScriptController：{switched_to_script}"
          f"（目前={after_switch}）")

    # 跑一段確認狀態機沒有卡死或拋例外（error_state 不能被標記）
    state_before_run = demo_sessions[0].get_current_state()
    for _ in range(60):
        simulation_app.update()
    observation = demo_sessions[0].get_last_observation()
    no_error = observation is not None and not observation.has_error
    print(f"[verify] Script 模式跑 60 tick 沒有進入 ERROR：{no_error}"
          f"（切換前 state={state_before_run.name}，"
          f"目前 state={demo_sessions[0].get_current_state().name}）")

    # 切回 AI 模式
    extension._on_demo_controller_mode_changed(table_id, True)
    for _ in range(5):
        simulation_app.update()
    back_to_ai = type(_current_controller()).__name__ == "ModelController"
    print(f"[verify] 切回 AI 應變回 ModelController：{back_to_ai}"
          f"（目前={type(_current_controller()).__name__}）")

    # 對不存在的 table_id 呼叫不應拋例外（Debug 工具的安靜 no-op 保證）
    try:
        extension._on_demo_controller_mode_changed("/World/NotATable", False)
        no_crash_on_unknown_table = True
    except Exception:
        no_crash_on_unknown_table = False
    print(f"[verify] 對不存在的 table_id 呼叫不拋例外：{no_crash_on_unknown_table}")

    all_pass = (
        ok and same_before_tick and switched_to_script and no_error
        and back_to_ai and no_crash_on_unknown_table
    )
    print(f"[verify] {'PASS' if all_pass else 'FAIL'}：操作策略注入機制"
          f"{'運作正常' if all_pass else '有問題，見上方個別項目'}")


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
