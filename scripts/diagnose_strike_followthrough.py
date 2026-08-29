"""
scripts/diagnose_strike_followthrough.py — 對單一已知 AIM_OK 案例
（(0.0,-0.635) roll=150deg）跑 STRIKE，逐步印出位置誤差/關節角/關節速度，
找出隨揮終點（follow-through waypoint，需要追蹤 ~1.5 m/s 桿尖速度）逾時的
確切原因：是卡在關節限位、關節速度飽和、還是單純沒收斂但還在慢慢逼近。

背景：scripts/verify_new_roll_table.py 用真實物理模擬證實 AIM 已經被新
roll 查表修好（0/20 -> 5/6），但 STRIKE 全部逾時；
scripts/search_backswing_ik.py 用數值 IK 證實後擺點本身是可達的（有健康
關節餘裕），所以問題不是靜態可達性，可能是
`ArticulationAPIImpl._step_motion()` 的速度追蹤（P 控制器 + feedforward
疊加、qdot clip 到 `_dof_limits`）在這個構型下無法在 1000 步內把末端帶到
5mm/0.02rad 容許值內。這支腳本補上直接觀測，不用再猜。

用法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/diagnose_strike_followthrough.py
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")

for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import math

_TABLE_Z = 0.0
_BALL_RADIUS = 0.028575
_CUE_BALL = (0.0, -0.635)
_SHOT_ANGLE_DEG = 0.0
_ROLL_DEG = 150
_AIM_MAX_STEPS = 4000
_STRIKE_MAX_STEPS = 2500
_CUE_BALL_SPEED = float(os.environ.get("DIAG_CUE_BALL_SPEED", "1.995"))


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.articulation_api_impl import ArticulationAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl
    from isaac_sim_impl_6_0.physics_api_impl import PhysicsAPIImpl

    from core.models.barrett_wam_robot import BarrettWamRobot
    from core.models.billiard_table import BilliardTable
    from core.models.table_robot_manager import TableRobotManager
    from core.models.contact_event import ContactEvent
    from core.services.base_placement_calculator import (
        CANONICAL_FLAT_ORIENTATION, CANONICAL_REST_JOINTS, compute_base_pose,
        compute_canonical_wrist_position, required_grip_position,
    )
    from core.services import cue_pose_calculator, swing_trajectory_calculator

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()
    physics_api = PhysicsAPIImpl()

    table_base_path = "/World/DiagnoseStrikeTable"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))

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

    print(f"[diag] dof_max_velocities={articulation_api._dof_limits.tolist()}")

    _RESET_JOINTS = np.array([[0.0, *CANONICAL_REST_JOINTS]])
    articulation_api._articulation.set_dof_positions(_RESET_JOINTS)
    articulation_api._articulation.set_dof_velocities(np.zeros((1, 7)))
    for _ in range(10):
        simulation_app.update()

    roll_rad = math.radians(_ROLL_DEG)
    base_position, base_yaw_rad = compute_base_pose(_CUE_BALL[0], _CUE_BALL[1], _SHOT_ANGLE_DEG, _TABLE_Z)
    robot.reposition(base_position)
    for _ in range(30):
        simulation_app.update()

    safe_joint_targets = [0.0, *CANONICAL_REST_JOINTS]
    safe_target_position = list(compute_canonical_wrist_position(base_position, 0.0))
    bridge_waypoints = cue_pose_calculator.compute_elevated_bridge_waypoints(
        safe_target_position, list(CANONICAL_FLAT_ORIENTATION),
        _CUE_BALL, _SHOT_ANGLE_DEG, _TABLE_Z, _BALL_RADIUS, roll_rad=roll_rad, rotate_steps=8,
    )
    articulation_api.move_through_poses(
        bridge_waypoints, preceding_joint_targets=(safe_joint_targets, safe_target_position)
    )
    wrist, orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
        _CUE_BALL, _SHOT_ANGLE_DEG, _TABLE_Z, _BALL_RADIUS, [0.0, 0.0], roll_rad=roll_rad
    )
    for step in range(_AIM_MAX_STEPS):
        simulation_app.update()
        if articulation_api.is_motion_complete():
            break
    print(f"[diag] AIM done at step={step}  timed_out={articulation_api.did_last_motion_timeout()}")

    direction_unit = cue_pose_calculator.compute_tilted_direction(_SHOT_ANGLE_DEG, tilt_rad)
    print(f"[diag] cue_ball_speed={_CUE_BALL_SPEED}")
    waypoints = swing_trajectory_calculator.compute_swing_waypoints(
        contact_position=wrist.tolist(), contact_orientation=orientation.tolist(),
        direction_unit=direction_unit.tolist(), cue_ball_speed=_CUE_BALL_SPEED,
        backswing_distance=0.15,
    )
    print(f"[diag] backswing target={waypoints[0].position}  velocity={waypoints[0].linear_velocity}")
    print(f"[diag] follow_through target={waypoints[1].position}  velocity={waypoints[1].linear_velocity}")
    articulation_api.move_through_poses(waypoints)

    names = ["base_yaw", "shoulder_pitch", "shoulder_yaw", "elbow_pitch", "wrist_yaw", "wrist_pitch", "palm_yaw"]
    lowers = np.array([-2.6, -1.985, -2.8, -0.9, -4.55, -1.5707, -3.0])
    uppers = np.array([2.6, 1.985, 2.8, math.pi, 1.25, 1.5707, 3.0])

    prev_waypoint_index = -1
    for step in range(_STRIKE_MAX_STEPS):
        simulation_app.update()
        joints = np.asarray(articulation_api._articulation.get_dof_positions())[0]
        velocities = np.asarray(articulation_api._articulation.get_dof_velocities())[0]
        wp_idx = articulation_api._waypoint_index
        if wp_idx != prev_waypoint_index:
            print(f"[diag] step={step}  切到 waypoint_index={wp_idx}")
            prev_waypoint_index = wp_idx
        if step % 20 == 0 and wp_idx == 1 and step < 400:
            current_position = np.array(articulation_api.get_end_effector_position())
            target = articulation_api._target_position
            pos_err = float(np.linalg.norm(current_position - target)) if target is not None else -1
            margins = np.minimum(joints - lowers, uppers - joints)
            min_margin_idx = int(np.argmin(margins))
            p_twist = articulation_api._compute_pose_tracking_twist()
            ff_twist = articulation_api._feedforward_twist
            print(
                f"[diag] step={step} wp={wp_idx} pos_err={pos_err:.5f} target={target.tolist()} current={current_position.tolist()} "
                f"min_margin={margins[min_margin_idx]:.4f}({names[min_margin_idx]}) "
                f"p_twist={np.round(p_twist,3).tolist()} ff_twist={np.round(ff_twist,3).tolist()} "
                f"vel={np.round(velocities,3).tolist()} vel_norm={float(np.linalg.norm(velocities)):.3f}"
            )
        elif step % 200 == 0 or step >= _STRIKE_MAX_STEPS - 5:
            current_position = np.array(articulation_api.get_end_effector_position())
            target = articulation_api._target_position
            pos_err = float(np.linalg.norm(current_position - target)) if target is not None else -1
            margins = np.minimum(joints - lowers, uppers - joints)
            min_margin_idx = int(np.argmin(margins))
            print(
                f"[diag] step={step} wp={wp_idx} pos_err={pos_err:.5f} "
                f"min_margin={margins[min_margin_idx]:.4f}({names[min_margin_idx]}) "
                f"vel={np.round(velocities,3).tolist()} vel_norm={float(np.linalg.norm(velocities)):.3f} "
                f"dof_limits={articulation_api._dof_limits.tolist()}"
            )
        if articulation_api.is_motion_complete():
            print(f"[diag] BREAK at step={step} waypoint_index={articulation_api._waypoint_index}")
            break
    else:
        print(f"[diag] EXHAUSTED at {_STRIKE_MAX_STEPS}")
    print(f"[diag] timed_out={articulation_api.did_last_motion_timeout()}")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
