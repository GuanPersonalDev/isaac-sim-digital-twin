"""
scripts/diagnose_ball_impact.py — 直接量測母球本身的真實物理速度（不是
桿尖的估計速度），驗證「桿子真的打到球了嗎」這個最根本的問題，繞開
scripts/verify_swing_trajectory.py 的 `actual_speed` 量測時機可能量錯
（收斂後才量，而非最接近球的瞬間）這個潛在測量問題。

背景：docs/issue-180-reachability-analysis.md 第十五節記錄 STRIKE 隨揮
終點有結構性穩態誤差，但 `DemoTableOrchestrator._execute_strike()` 正式
程式碼本身只是丟出 `move_through_poses()` 就返回，不等待收斂、不自己量測
速度——球是否真的被打到，完全取決於桿頭幾何有沒有在模擬過程中真的碰到
球的真實 PhysX 碰撞。`verify_swing_trajectory.py` 找到桿尖離球最近距離
只有 4.67mm（遠小於球半徑 28.6mm），代表桿尖軌跡很可能真的穿過了球——
這支腳本直接追蹤 `RigidPrim(paths=ball_prim_path).get_velocities()`，
看球在整個 STRIKE 步驟範圍內有沒有真的獲得速度，不管軟體的完成判定
說了什麼。

用法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/diagnose_ball_impact.py
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")

for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_TABLE_Z = 0.0
_BALL_RADIUS = 0.028575
_CUE_BALL = (0.0, -0.635)
_SHOT_ANGLE_DEG = 0.0
_CUE_BALL_SPEED = 1.995
_AIM_MAX_STEPS = 4000
_STRIKE_MAX_STEPS = 2500


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf
    from isaacsim.core.experimental.prims import RigidPrim

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.articulation_api_impl import ArticulationAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl
    from isaac_sim_impl_6_0.physics_api_impl import PhysicsAPIImpl

    from core.models.barrett_wam_robot import BarrettWamRobot
    from core.models.billiard_table import BilliardTable
    from core.models.table_robot_manager import TableRobotManager
    from core.services.base_placement_calculator import (
        CANONICAL_FLAT_ORIENTATION, CANONICAL_REST_JOINTS, compute_base_pose,
        compute_canonical_wrist_position,
    )
    from core.services import cue_pose_calculator, swing_trajectory_calculator

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()
    physics_api = PhysicsAPIImpl()

    table_base_path = "/World/DiagnoseBallImpactTable"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))
    # 其餘 9 顆球放到遠離桌台的地方，避免干擾這次只關心母球的碰撞測試；
    # TableBallSet.build() 要求 key 0-9 齊全。
    ball_positions = {i: (5.0 + i * 0.2, 5.0) for i in range(10)}
    ball_positions[0] = _CUE_BALL
    table.get_table_ball_set().build(ball_positions)

    robot_prim_path = BarrettWamRobot.get_prim_path(table_base_path)
    end_effector_prim_path = BarrettWamRobot.get_end_effector_prim_path(table_base_path)
    articulation_api = ArticulationAPIImpl(robot_prim_path, end_effector_prim_path)
    robot_manager = TableRobotManager(
        table.get_table_center(), table_base_path, stage_api, articulation_api, BarrettWamRobot
    )
    robot = robot_manager.get_robot()

    ball_prim_path = table.get_table_ball_set().get_ball_prim_paths()[0]
    ball_rigid_prim = RigidPrim(paths=ball_prim_path)
    print(f"[diag] ball_prim_path={ball_prim_path}")

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()
    articulation_api.initialize()
    for _ in range(5):
        simulation_app.update()

    _RESET_JOINTS = np.array([[0.0, *CANONICAL_REST_JOINTS]])
    articulation_api._articulation.set_dof_positions(_RESET_JOINTS)
    articulation_api._articulation.set_dof_velocities(np.zeros((1, 7)))
    for _ in range(10):
        simulation_app.update()

    base_position, base_yaw_rad = compute_base_pose(_CUE_BALL[0], _CUE_BALL[1], _SHOT_ANGLE_DEG, _TABLE_Z)
    robot.reposition(base_position)
    for _ in range(30):
        simulation_app.update()

    roll_rad = cue_pose_calculator.lookup_roll_rad(_CUE_BALL)
    safe_joint_targets = [0.0, *CANONICAL_REST_JOINTS]
    safe_target_position = list(compute_canonical_wrist_position(base_position, 0.0))
    bridge_waypoints = cue_pose_calculator.compute_elevated_bridge_waypoints(
        safe_target_position, list(CANONICAL_FLAT_ORIENTATION),
        _CUE_BALL, _SHOT_ANGLE_DEG, _TABLE_Z, _BALL_RADIUS, roll_rad=roll_rad,
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
    waypoints = swing_trajectory_calculator.compute_swing_waypoints(
        contact_position=wrist.tolist(), contact_orientation=orientation.tolist(),
        direction_unit=direction_unit.tolist(), cue_ball_speed=_CUE_BALL_SPEED,
    )
    articulation_api.move_through_poses(waypoints)

    max_ball_speed = 0.0
    max_ball_speed_step = -1
    initial_ball_pos = np.array(ball_rigid_prim.get_world_poses()[0][0])
    for step in range(_STRIKE_MAX_STEPS):
        simulation_app.update()
        linear_vel, _angular_vel = ball_rigid_prim.get_velocities()
        ball_speed = float(np.linalg.norm(np.asarray(linear_vel)[0]))
        if ball_speed > max_ball_speed:
            max_ball_speed = ball_speed
            max_ball_speed_step = step
        if step % 200 == 0:
            print(f"[diag] step={step} waypoint_index={articulation_api._waypoint_index} ball_speed={ball_speed:.4f} is_complete={articulation_api.is_motion_complete()}")
        # 不要在 is_motion_complete() 就提早跳出——要看完整過程球有沒有被打到。

    final_ball_pos = np.array(ball_rigid_prim.get_world_poses()[0][0])
    ball_displacement = float(np.linalg.norm(final_ball_pos - initial_ball_pos))
    print(f"[diag] max_ball_speed={max_ball_speed:.4f} m/s at step={max_ball_speed_step}")
    print(f"[diag] ball_displacement={ball_displacement:.4f} m (initial={initial_ball_pos.tolist()} final={final_ball_pos.tolist()})")
    print(f"[diag] required_tip_speed={swing_trajectory_calculator.compute_required_tip_speed(_CUE_BALL_SPEED):.4f}")
    print(f"[diag] expected cue_ball_speed(action)={_CUE_BALL_SPEED}")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
