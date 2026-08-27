"""
scripts/debug_ported_aim_regression.py — 對照測試：用「已經驗證過 100% 成功」
的 scan_elevated_bridge_approach.py 網格點（(0.0, 0.4)，flat 案例），分別用
(a) 舊研究腳本本身的邏輯 (b) 新搬進 core/services/cue_pose_calculator.py +
table_orchestrator 風格的呼叫順序，看新版本是不是真的複現了舊版本的行為，
藉此判斷 verify_swing_trajectory.py 的全滅是新程式碼的迴歸，還是純粹測試
網格範圍本身比較難。

用法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/debug_ported_aim_regression.py
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")

for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import math


def _shot_angle_deg(cue_ball, target):
    return math.degrees(math.atan2(cue_ball[0] - target[0], target[1] - cue_ball[1]))


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.articulation_api_impl import ArticulationAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl

    from core.models.barrett_wam_robot import BarrettWamRobot
    from core.models.billiard_table import BilliardTable
    from core.models.table_robot_manager import TableRobotManager
    from core.services.base_placement_calculator import (
        CANONICAL_FLAT_ORIENTATION, CANONICAL_REST_JOINTS, compute_base_pose,
        compute_canonical_wrist_position, required_grip_position,
    )
    from core.services import cue_pose_calculator

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()

    table_base_path = "/World/DebugPortedAimTable"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))
    table_z = 0.0
    ball_radius = 0.028575

    robot_prim_path = BarrettWamRobot.get_prim_path(table_base_path)
    end_effector_prim_path = BarrettWamRobot.get_end_effector_prim_path(table_base_path)
    articulation_api = ArticulationAPIImpl(robot_prim_path, end_effector_prim_path)
    robot_manager = TableRobotManager(
        table.get_table_center(), table_base_path, stage_api, articulation_api, BarrettWamRobot
    )
    robot = robot_manager.get_robot()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()
    articulation_api.initialize()
    for _ in range(5):
        simulation_app.update()

    def _run_case(label, cue_ball, shot_angle_deg, max_steps=1200):
        base_position, base_yaw_rad = compute_base_pose(cue_ball[0], cue_ball[1], shot_angle_deg, table_z)
        robot.reposition(base_position)
        for _ in range(30):
            simulation_app.update()

        wrist, orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
            cue_ball, shot_angle_deg, table_z, ball_radius, [0.0, 0.0]
        )
        print(f"  [{label}] tilt_rad={tilt_rad}")
        if tilt_rad is None:
            print(f"  [{label}] GEOMETRICALLY_INFEASIBLE crossing={crossing}")
            return

        if tilt_rad <= 1e-6:
            grip_position = required_grip_position(cue_ball[0], cue_ball[1], shot_angle_deg)
            joint_targets = [base_yaw_rad, *CANONICAL_REST_JOINTS]
            articulation_api.move_to_joint_position(
                joint_targets, [grip_position[0], grip_position[1], table_z + ball_radius]
            )
        else:
            safe_joint_targets = [0.0, *CANONICAL_REST_JOINTS]
            safe_target_position = list(compute_canonical_wrist_position(base_position, 0.0))
            roll_rad = cue_pose_calculator.lookup_roll_rad(cue_ball)
            bridge_waypoints = cue_pose_calculator.compute_elevated_bridge_waypoints(
                safe_target_position, list(CANONICAL_FLAT_ORIENTATION),
                cue_ball, shot_angle_deg, table_z, ball_radius, position_offset=[0.0, 0.0],
                roll_rad=roll_rad,
            )
            articulation_api.move_through_poses(
                bridge_waypoints, preceding_joint_targets=(safe_joint_targets, safe_target_position)
            )

        settled_step = None
        for step in range(max_steps):
            simulation_app.update()
            if step % 100 == 0:
                joints = np.asarray(articulation_api._articulation.get_dof_positions())[0]
                err = float(np.linalg.norm(np.array(articulation_api.get_end_effector_position()) - articulation_api._target_position)) if articulation_api._target_position is not None else -1
                print(f"  [{label}] step={step} err={err:.5f} joints={np.round(joints,4).tolist()} is_joint_space={articulation_api._is_joint_space_motion}")
            if articulation_api.is_motion_complete():
                settled_step = step
                break
        timed_out = articulation_api.did_last_motion_timeout()
        print(f"  [{label}] FINAL settled_step={settled_step} timed_out={timed_out}")

    # 這個案例在 scan_elevated_bridge_approach.py 的原始 25 點網格測試裡
    # 已經驗證是 flat（tilt=0）且成功收斂的案例。
    _run_case("flat_(0.0,0.4)", (0.0, 0.4), _shot_angle_deg((0.0, 0.4), (0.0, 0.635)))

    # 這個案例已知需要抬高（tilt>0），且原本高架橋研究驗證成功。
    _run_case("bridge_(-0.5,-1.1)", (-0.5, -1.1), _shot_angle_deg((-0.5, -1.1), (0.0, 0.635)))

    # action_bounds.py 的 Kitchen 母球位置範圍（CUE_BALL_PLACEMENT_Y=
    # (-1.241425,-0.635)），shot_angle=0（瞄準桌台 +Y，遠離這個母球最近的
    # 那面庫邊）——verify_swing_trajectory.py 對這個範圍全滅，這裡單獨隔離
    # 出來看詳細收斂過程。
    _run_case("kitchen_(0.0,-0.9382125)", (0.0, -0.9382125), 0.0)


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
