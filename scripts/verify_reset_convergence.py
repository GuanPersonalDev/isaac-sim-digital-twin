"""
scripts/verify_reset_convergence.py — 驗證 RESET 狀態不再卡死：
move_to_home() 收斂後 is_motion_complete() 是否真的會變成 True。

背景：DemoTableOrchestrator 進入 RESET 時呼叫 robot_arm.reset()，內部走
ArticulationAPIImpl.move_to_home()，比對桿尖目前位置跟 _home_position。原本
_default_joint_positions 在 initialize() 裡同步呼叫 get_dof_positions()（physics
可能一步都還沒跑），跟隔了至少一個 PHYSICS_POST_STEP 才擷取的 _home_position
不是同一個瞬間的姿態，導致 move_to_home() 永遠碰不到 _home_position、
is_motion_complete() 恆為 False。修法是把兩者搬到同一個 callback 一起擷取
（見 articulation_api_impl.py _capture_home_position_once）。

這支腳本走跟正式流程一樣的 TableRobotManager（含 CueStick／FixedJoint），
呼叫 initialize() 後立刻呼叫 move_to_home()，監看 is_motion_complete() 是否
在合理步數內變成 True。

用法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/verify_reset_convergence.py
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")

for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.articulation_api_impl import ArticulationAPIImpl

    from core.models.barrett_wam_robot import BarrettWamRobot
    from core.models.table_robot_manager import TableRobotManager

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    base_path = "/World/ResetConvergenceProbe"
    robot_prim_path = BarrettWamRobot.get_prim_path(base_path)
    end_effector_prim_path = BarrettWamRobot.get_end_effector_prim_path(base_path)
    articulation_api = ArticulationAPIImpl(robot_prim_path, end_effector_prim_path)

    robot_manager = TableRobotManager(
        (0.0, 0.0, 0.0), base_path, stage_api, articulation_api, BarrettWamRobot
    )

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    articulation_api.initialize()
    # initialize() 只註冊 home-capture callback，實際擷取要等下一個
    # PHYSICS_POST_STEP，這裡先跑幾步讓它觸發。
    for _ in range(5):
        simulation_app.update()

    print(f"default_joint_positions={np.asarray(articulation_api._default_joint_positions).tolist()}")
    print(f"home_position={np.asarray(articulation_api._home_position).tolist()}")

    # 先模擬真正的 Demo 流程：AIMING 會把手臂移動到 CANONICAL_REST_JOINTS
    # 附近的姿態，不是一開始就停在 home，這樣才是有意義的收斂測試。
    from core.services.base_placement_calculator import CANONICAL_REST_JOINTS

    print("\n--- 先移動到瞄準姿態 ---")
    joint_targets = [0.5, *CANONICAL_REST_JOINTS]
    # target_end_effector_position 只是給 is_motion_complete() 用，這裡不需要
    # 精確算出來，直接跑固定步數讓它自然穩定，不靠 is_motion_complete() 判斷。
    articulation_api.move_to_joint_position(joint_targets, articulation_api.get_end_effector_position())
    for _ in range(300):
        simulation_app.update()
    print(f"  position_after_aim={articulation_api.get_end_effector_position()}")

    print("\n--- 再呼叫 reset() 回 home ---")
    robot = robot_manager.get_robot()
    robot.reset()  # 呼叫 move_to_home()

    max_steps = 500
    converged = False
    for step in range(max_steps):
        simulation_app.update()
        if robot.is_reset_complete():
            converged = True
            print(f"converged at step={step}")
            break

    if not converged:
        print(f"NOT CONVERGED after {max_steps} steps")

    final_position = articulation_api.get_end_effector_position()
    print(f"final_end_effector_position={final_position}")
    if articulation_api._home_position is not None:
        error = float(
            np.linalg.norm(np.array(final_position) - np.asarray(articulation_api._home_position))
        )
        print(f"final_position_error={error:.5f} m (tolerance={ArticulationAPIImpl.POSITION_TOLERANCE})")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
