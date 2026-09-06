"""
scripts/profile_ur10e_waypoint_tolerance.py — 驗證「中繼 waypoint 的收斂
容許值太緊，是手臂動作看起來很慢的主因」這個假設。

背景：`Ur10eRmpflowController` 把一段移動切成多個 waypoint
（`_MAX_WAYPOINT_STEP_M=0.08`），每個 waypoint 都要收斂到
`_POSITION_TOLERANCE_M=0.005`（5mm）／`_ORIENTATION_TOLERANCE_RAD=0.02`
才會前進到下一個。RMPflow 是漸近收斂，要壓到 5mm 等於手臂在每個中繼點
都幾乎完全停下來、再重新加速——實測 RESET 要 902 tick（模擬時間 15 秒）
才把手臂移到 HOME，真實 UR10e 這種動作 2~3 秒就夠了。

中繼點其實不需要這個精度：最終精度是由後面的收尾階段
（`_start_finishing_phase()` → 解析 IK／joint-space finish）負責的，
waypoint chain 只是「大致沿著這條路走過去、順便避障」。

這支腳本掃描幾組中繼容許值，記錄：
1. RESET（move_to_home）需要幾個 tick
2. 收尾後最終落點的實際誤差——確認放寬中繼容許值**不會**犧牲最終精度

不改生產程式碼，只在實例上覆寫類別屬性。

跑法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/profile_ur10e_waypoint_tolerance.py
"""

import os
import sys
import time
import traceback

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# (位置容許值 m, 方向容許值 rad)；第一組是現況
_TRIALS = (
    (0.005, 0.02),
    (0.02, 0.05),
    (0.05, 0.10),
    (0.10, 0.20),
)
_MAX_RESET_TICKS = 3000


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
    from core.models.ur10e_robot import UR10eRobot
    from core.services.spread_score_calculator import TABLE_LENGTH, TABLE_WIDTH

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()

    table_base_path = "/World/ToleranceUr10eTable"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))
    table_ball_set = table.get_table_ball_set()
    table_z = table_ball_set.get_table_z()
    ball_radius = table_ball_set.DEFAULT_BALL_RADIUS

    robot_prim_path = UR10eRobot.get_prim_path(table_base_path)
    end_effector_prim_path = UR10eRobot.get_end_effector_prim_path(table_base_path)
    articulation_api = ArticulationAPIImpl(robot_prim_path, end_effector_prim_path)
    TableRobotManager(
        table.get_table_center(), table_base_path, stage_api, articulation_api, UR10eRobot,
    )
    ball_prim_path = table_ball_set.get_ball_prim_paths()[0]

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()
    table_ball_set.place_ball(0, 0.0, 0.5)
    for _ in range(5):
        simulation_app.update()
    articulation_api.initialize()
    for _ in range(5):
        simulation_app.update()

    _TABLE_OBSTACLE_HEIGHT_M = 0.15
    table_center = table.get_table_center()
    articulation_api.register_static_box_obstacle(
        [table_center[0], table_center[1], table_z - _TABLE_OBSTACLE_HEIGHT_M / 2.0],
        [TABLE_WIDTH, TABLE_LENGTH, _TABLE_OBSTACLE_HEIGHT_M],
    )
    articulation_api.register_dynamic_sphere_obstacle(ball_prim_path, ball_radius)

    # ⚠️ 一定要先同步底座位姿，否則 RMPflow 會以為底座在原點、實際在
    # table_center+(1.5,0,0)，move_to_home() 追的是錯的世界座標，永遠不會
    # 收斂（第一版沒加，四組容許值全部跑滿 3000 tick、關節誤差 1.07 rad，
    # 量到的完全是假的）。跟 DemoTableSession._sync_initial_robot_base_pose()
    # 同一件事。
    initial_base_offset = TableRobotManager._ROBOT_OFFSET_FROM_TABLE_CENTER
    initial_base_position = [table_center[i] + initial_base_offset[i] for i in range(3)]
    articulation_api.set_robot_base_pose(initial_base_position, [1.0, 0.0, 0.0, 0.0])
    for _ in range(5):
        simulation_app.update()

    rmp_ctrl = articulation_api._ur10e_rmpflow_controller
    home_joints = np.asarray(rmp_ctrl._HOME_JOINT_POSITIONS, dtype=float)
    active_indices = rmp_ctrl._active_dof_indices
    articulation = articulation_api._articulation

    # 每個 trial 前先把手臂送回同一個起始姿態，讓比較有意義
    start_joints = np.asarray(articulation.get_dof_positions())[0].copy()

    def _reset_to_start() -> None:
        articulation.switch_dof_control_mode("position")
        articulation.set_dof_position_targets(start_joints[None, :])
        for _ in range(240):
            simulation_app.update()

    print(f"[tolerance] HOME 關節角={np.round(home_joints, 4).tolist()}")
    print(f"[tolerance] {'中繼容許值':<26} {'RESET tick':>11} {'模擬秒數':>9} "
          f"{'HOME 關節誤差(rad)':>18} {'逾時':>6}")

    for position_tolerance, orientation_tolerance in _TRIALS:
        _reset_to_start()

        # 覆寫實例屬性（類別屬性讀取會先找實例），不動生產程式碼。
        # 這兩個常數只被 _is_current_waypoint_converged() 使用，收尾階段的
        # 精度由 _POSITION_TOLERANCE_M／_FINAL_ORIENTATION_TOLERANCE_RAD
        # 另外把關，不受這裡影響。
        rmp_ctrl._WAYPOINT_POSITION_TOLERANCE_M = position_tolerance
        rmp_ctrl._WAYPOINT_ORIENTATION_TOLERANCE_RAD = orientation_tolerance

        articulation_api.move_to_home()
        start = time.perf_counter()
        ticks = 0
        while not articulation_api.is_motion_complete() and ticks < _MAX_RESET_TICKS:
            simulation_app.update()
            ticks += 1
        elapsed = time.perf_counter() - start

        achieved = np.asarray(articulation.get_dof_positions())[0][active_indices]
        joint_error = float(np.max(np.abs(achieved - home_joints)))
        timed_out = articulation_api.did_last_motion_timeout()

        label = f"pos={position_tolerance:.3f}m ori={orientation_tolerance:.2f}rad"
        print(f"[tolerance] {label:<26} {ticks:>11} {ticks / 60.0:>8.1f}s "
              f"{joint_error:>18.6f} {str(timed_out):>6}   "
              f"（實際耗時 {elapsed:.1f}s）")
        sys.stdout.flush()

    # 還原
    del rmp_ctrl._WAYPOINT_POSITION_TOLERANCE_M
    del rmp_ctrl._WAYPOINT_ORIENTATION_TOLERANCE_RAD
    print("[tolerance] 完成。HOME 關節誤差的驗收門檻是 _FINAL_ORIENTATION_TOLERANCE_RAD=0.005 rad，"
          "只要放寬中繼容許值之後這個值沒有變差，就代表最終精度由收尾階段守住、沒有被犧牲。")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    except Exception:
        print("[tolerance] _run() 拋出例外：")
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        simulation_app.close()
