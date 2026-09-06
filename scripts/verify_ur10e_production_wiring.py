"""
scripts/verify_ur10e_production_wiring.py — UR10e 重新設計計畫步驟 9：
驗證**正式路徑**的接線，不是驗證演算法。

為什麼需要這支：`scripts/test_ur10e_table_flat.py`／`test_ur10e_table_bridge.py`
驗的是演算法（AIM 準不準、STRIKE 打不打得到），但它們把兩件正式路徑該做的
事**自己在腳本裡手動做掉了**，因此這兩件事從來沒有被真實驗證過：

1. `ArticulationAPI.set_robot_base_pose()` —— 第一次 RESET 之前要先讓
   RMPflow 知道底座在世界座標的哪裡。沒有的話 RMPflow 會當底座在原點，
   `move_to_home()` 的起點（實際量到的末端世界位姿）跟目標（RMPflow 內部
   算出來的 HOME 末端位姿）分屬兩個座標系。
2. `register_static_box_obstacle()`／`register_dynamic_sphere_obstacle()`
   —— 決策 6 的第一層防護。查證過生產路徑從來沒呼叫過。

兩者現在都搬進 `DemoTableSession.initialize_articulation()`（core/services/
table_session.py）。這支腳本走的就是那個正式入口，然後用 `RobotArm.reset()`
（正式路徑 `DemoTableOrchestrator._reset_downstream()` 呼叫的同一個方法）
把手臂帶回 HOME，確認真的收斂。

不涵蓋 AIM／STRIKE——那兩段已經由 flat／bridge 兩支驗收腳本覆蓋。

跑法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/verify_ur10e_production_wiring.py
"""

import os
import sys
import traceback
from unittest.mock import MagicMock

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_MAX_RESET_STEPS = 4000


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl
    from isaac_sim_impl_6_0.articulation_api_impl import ArticulationAPIImpl

    from core.models.billiard_table import BilliardTable
    from core.models.table_robot_manager import TableRobotManager
    from core.services.table_session import DemoTableSession

    # 從正式 extension 讀生產路徑實際掛的手臂類別——這支腳本的重點就是
    # 「正式設定長什麼樣」，不能自己另外指定一個。
    from billiard_digital_twin.billiard_digital_twin import _ROBOT_ARM_CLASS

    print(f"[wiring] 生產路徑的 _ROBOT_ARM_CLASS={_ROBOT_ARM_CLASS.__name__}")

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()

    table_base_path = "/World/TestUr10eProductionWiring"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))

    robot_prim_path = _ROBOT_ARM_CLASS.get_prim_path(table_base_path)
    end_effector_prim_path = _ROBOT_ARM_CLASS.get_end_effector_prim_path(table_base_path)
    articulation_api = ArticulationAPIImpl(robot_prim_path, end_effector_prim_path)

    robot_manager = TableRobotManager(
        table.get_table_center(), table_base_path, stage_api, articulation_api, _ROBOT_ARM_CLASS,
    )
    robot_arm = robot_manager.get_robot()

    # runtime／pocket_handler 跟這支要驗的接線無關，用 mock 佔位即可——
    # 重點是走 DemoTableSession 這個正式入口，不是重建整條 Demo 流程。
    session = DemoTableSession(
        table_base_path, table, MagicMock(), MagicMock(), rigid_body_api,
        robot_manager, articulation_api,
    )

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    expected_base = robot_manager.get_initial_robot_base_position()
    print(f"[wiring] TableRobotManager 回報的初始底座位置={list(expected_base)}")

    # === 正式入口：Timeline PLAY 事件會呼叫的就是這一個方法 ===
    session.initialize_articulation()
    for _ in range(5):
        simulation_app.update()
    print("[wiring] DemoTableSession.initialize_articulation() 完成")

    rmpflow_controller = articulation_api._ur10e_rmpflow_controller
    if rmpflow_controller is None:
        print("[wiring] FAIL：UR10e 模式沒有啟用（articulation 裡找不到 CueSlideJoint）")
        return

    synced_base = rmpflow_controller._base_position
    base_ok = synced_base is not None and np.allclose(synced_base, np.asarray(expected_base))
    print(f"[wiring] RMPflow 內部記錄的底座位置={None if synced_base is None else synced_base.tolist()}"
          f"  {'一致' if base_ok else '⚠️ 不一致／未同步'}")

    obstacle_paths = articulation_api._ur10e_registered_obstacle_paths
    dynamic_obstacles = rmpflow_controller._dynamic_obstacle_sources
    obstacles_ok = len(obstacle_paths) == 1 and len(dynamic_obstacles) == 1
    print(f"[wiring] 已註冊靜態障礙物 {len(obstacle_paths)} 個={obstacle_paths}，"
          f"動態障礙物 {len(dynamic_obstacles)} 個  {'符合預期' if obstacles_ok else '⚠️ 不符預期（各應為 1）'}")

    # === 正式路徑的 RESET：DemoTableOrchestrator._reset_downstream() 呼叫
    # 的就是 RobotArm.reset() ===
    print("[wiring] 呼叫 RobotArm.reset()（＝正式路徑的 RESET）...")
    sys.stdout.flush()
    robot_arm.reset()
    steps = 0
    while not robot_arm.is_reset_complete() and steps < _MAX_RESET_STEPS:
        simulation_app.update()
        steps += 1
    timed_out = articulation_api.did_last_motion_timeout()
    print(f"[wiring] RESET 完成，steps={steps} did_last_motion_timeout={timed_out}")

    achieved_joints = np.asarray(articulation_api.get_dof_positions_for_debug(), dtype=float)
    home_joints = np.asarray(rmpflow_controller._HOME_JOINT_POSITIONS, dtype=float)
    active_indices = rmpflow_controller._active_dof_indices
    joint_error = float(np.max(np.abs(achieved_joints[active_indices] - home_joints)))
    print(f"[wiring] HOME 關節角誤差（最大分量）={joint_error:.6f} rad")

    reset_ok = (not timed_out) and steps < _MAX_RESET_STEPS and joint_error <= 0.01
    if base_ok and obstacles_ok and reset_ok:
        print("[wiring] PASS：正式入口有同步底座位姿、有註冊避障物，RESET 也正常收斂到 HOME")
    else:
        reasons = []
        if not base_ok:
            reasons.append("底座位姿沒有在 initialize_articulation() 裡同步給 RMPflow")
        if not obstacles_ok:
            reasons.append("避障物沒有註冊齊全")
        if not reset_ok:
            reasons.append(f"RESET 未正常收斂（steps={steps} timeout={timed_out} 關節誤差={joint_error:.6f}）")
        print(f"[wiring] FAIL：{'；'.join(reasons)}")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    except Exception:
        print("[wiring] _run() 拋出例外：")
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        simulation_app.close()
