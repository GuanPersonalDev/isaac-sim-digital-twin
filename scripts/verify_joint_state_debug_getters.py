"""
scripts/verify_joint_state_debug_getters.py — 確認能不能取得每個關節的
角度與目前速度（`ArticulationAPIImpl.get_dof_positions_for_debug()`／
`get_dof_velocities_for_debug()`／`get_dof_names_for_debug()`），數值要
真的有意義（名稱/角度/速度三者長度一致，不是空陣列或全 0）。

跑法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
    PYTHONIOENCODING=utf-8 \
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/verify_joint_state_debug_getters.py
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

    manager = omni.kit.app.get_app().get_extension_manager()
    manager.add_path(_EXT_DIR)
    for _ in range(10):
        simulation_app.update()
    manager.set_extension_enabled_immediate("billiard_digital_twin", True)
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
        print("[verify] FAIL：沒有 Demo 桌")
        return
    table_id = demo_sessions[0].get_table_id()
    articulation_api = extension._demo_articulation_apis.get(table_id)
    if articulation_api is None:
        print("[verify] FAIL：找不到對應的 ArticulationAPIImpl")
        return

    names = articulation_api.get_dof_names_for_debug()
    positions = articulation_api.get_dof_positions_for_debug()
    velocities = articulation_api.get_dof_velocities_for_debug()

    print(f"[verify] 關節數量：names={len(names)} positions={len(positions)} "
          f"velocities={len(velocities)}")
    lengths_match = len(names) == len(positions) == len(velocities) and len(names) > 0
    print(f"[verify] 三者長度一致且非空：{lengths_match}")
    for name, position, velocity in zip(names, positions, velocities):
        print(f"    {name:<20} q={position:>9.4f}  qd={velocity:>9.4f}")

    # RESET 剛完成、手臂應該靜止在 HOME，速度應接近 0（不是判斷「取得到非 0」
    # 這種容易флаky 的條件，而是驗證讀到的是真實物理量、量級合理）
    velocities_are_small = all(abs(v) < 0.5 for v in velocities)
    print(f"[verify] RESET 完成後靜止姿態，速度量級合理（<0.5）：{velocities_are_small}")

    # 讓手臂真的動起來一段（進入 AIMING），確認速度不是恆為 0（陣列接錯/永遠讀
    # 到快取值的話，這裡會露餡）
    max_abs_velocity_seen = 0.0
    for _ in range(120):
        simulation_app.update()
        current_velocities = articulation_api.get_dof_velocities_for_debug()
        if current_velocities:
            max_abs_velocity_seen = max(max_abs_velocity_seen, max(abs(v) for v in current_velocities))
    moved = max_abs_velocity_seen > 1e-4
    print(f"[verify] 手臂移動過程中量到非零速度（max|v|={max_abs_velocity_seen:.6f}）："
          f"{moved}")

    # DebugMenu 實際會呼叫的入口：BilliardExtension.get_joint_state_text()
    joint_state_text = extension.get_joint_state_text(table_id)
    text_has_all_joint_names = bool(joint_state_text) and all(
        name in joint_state_text for name in names
    )
    print(f"[verify] get_joint_state_text() 產出的文字含全部關節名稱："
          f"{text_has_all_joint_names}")
    print("---")
    print(joint_state_text)
    print("---")

    all_pass = lengths_match and velocities_are_small and moved and text_has_all_joint_names
    print(f"[verify] {'PASS' if all_pass else 'FAIL'}：關節角度/速度讀取"
          f"{'正常，可以加進 Debug Menu' if all_pass else '有問題，見上方個別項目'}")


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
