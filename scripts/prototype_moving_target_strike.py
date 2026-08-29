"""
scripts/prototype_moving_target_strike.py — 驗證 STRIKE 隨揮終點的候選修法：
「移動目標點」（moving target）取代目前的「靜態目標點 + feedforward 速度」。

背景：docs/issue-180-reachability-analysis.md 第十五節證實目前的設計（
`compute_swing_waypoints()` 的隨揮終點是靜態位置＋固定 feedforward 速度）
會在 P 控制器（POSITION_GAIN=5.0）+ feedforward 疊加的控制律下产生結構性
穩態誤差——目標一旦被通過（因為 feedforward 持續往前推），P 項就會反向
煞車，最終在 |feedforward|/POSITION_GAIN 這個距離外的一點跟 feedforward
打平，桿尖停在那裡幾乎不動。直接量測母球真實物理速度
（scripts/diagnose_ball_impact.py）證實這不是測量時機問題——球真的沒被
好好打到（max_ball_speed=0.24m/s，該有 1.51m/s）。

這支原型腳本測試的修法：不要求 P 控制器去「收斂」到一個固定點，而是每個
物理步都呼叫 `move_to_pose()` 把目標往 direction 方向前進
`required_tip_speed × PHYSICS_DT`（`core/services/rolling_resistance_
service.py` 的 `PHYSICS_DT=1/60`，跟 `SimulationManager.setup_simulation
(dt=1/60)` 一致）——目標永遠在桿尖前方一小步，P 項的角色只剩「修正微小
的追蹤誤差」，不會反向煞車，桿尖應該能維持接近 feedforward 全速通過整段
路徑。這是在呼叫端（不是共用的 ArticulationAPIImpl 控制律本身）實作，
`_activate_pose_target()` 本來就設計成可以重複呼叫更新目標，不需要改
共用類別。

用 scripts/diagnose_ball_impact.py 同一套母球真實速度量測方式驗收：
`max_ball_speed` 有沒有真正接近 `required_tip_speed`。

用法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/prototype_moving_target_strike.py
"""

import math
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")

for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_TABLE_Z = 0.0
_BALL_RADIUS = 0.028575
_CUE_BALL = (0.0, -0.9382125)
_SHOT_ANGLE_DEG = 0.0
_CUE_BALL_SPEED = 1.995
_AIM_MAX_STEPS = 4000
_PHYSICS_DT = 1.0 / 60.0
_BACKSWING_DISTANCE_M = 0.15
_EXTRA_SAFETY_STEPS = 30  # 多跑幾步確保桿尖真正走完全程，不要卡在最後一步前
_ORIENTATION_GAIN_OVERRIDE = float(os.environ.get("PROTO_ORIENTATION_GAIN", "5.0"))
# 揮桿階段用調低的 ORIENTATION_GAIN，減少姿態修正跟平移速度搶 qdot 額度
# （見 docs/issue-180-reachability-analysis.md 第十六節線性規劃分析：完全
# 鎖死姿態時 y=-0.9382125 案例最高只能到 1.33m/s，放寬到允許小幅姿態
# 漂移能顯著提高可達速度）。只在這支原型腳本用 monkeypatch 測試假說，
# 還沒決定要不要正式改共用類別。


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

    table_base_path = "/World/PrototypeMovingTargetTable"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))
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
    print(f"[proto] AIM done at step={step}  timed_out={articulation_api.did_last_motion_timeout()}")

    direction_unit = np.array(cue_pose_calculator.compute_tilted_direction(_SHOT_ANGLE_DEG, tilt_rad))
    required_tip_speed = swing_trajectory_calculator.compute_required_tip_speed(_CUE_BALL_SPEED)
    follow_through_distance = swing_trajectory_calculator.compute_follow_through_distance(required_tip_speed)
    contact_position = np.array(wrist)
    backswing_position = swing_trajectory_calculator.compute_backswing_position(
        contact_position, direction_unit, _BACKSWING_DISTANCE_M
    )
    follow_through_position = contact_position + follow_through_distance * direction_unit

    print(f"[proto] required_tip_speed={required_tip_speed:.4f}  backswing={backswing_position.tolist()}  follow_through={follow_through_position.tolist()}")

    # Phase 1：後擺（靜態點，跟現有設計一樣，v=0，等真正收斂）。
    articulation_api.move_to_pose(backswing_position.tolist(), orientation.tolist(), [0.0, 0.0, 0.0])
    for step in range(2500):
        simulation_app.update()
        if articulation_api.is_motion_complete():
            break
    print(f"[proto] backswing done at step={step}  timed_out={articulation_api.did_last_motion_timeout()}")

    # Phase 2：移動目標點——每個物理步把目標往 direction 前進
    # required_tip_speed*PHYSICS_DT，全程帶 feedforward=required_tip_speed*
    # direction，不等待/不檢查 is_motion_complete()，跑固定步數走完全程。
    # 揮桿階段調低 ORIENTATION_GAIN，減少姿態修正跟平移速度搶 qdot 額度
    # （見文件第十六節：完全鎖死姿態時 y=-0.9382125 案例最高只能到
    # 1.33m/s，放寬姿態容許能顯著提高可達速度）。
    original_orientation_gain = ArticulationAPIImpl.ORIENTATION_GAIN
    ArticulationAPIImpl.ORIENTATION_GAIN = _ORIENTATION_GAIN_OVERRIDE
    print(f"[proto] ORIENTATION_GAIN override: {original_orientation_gain} -> {_ORIENTATION_GAIN_OVERRIDE}")

    total_distance = _BACKSWING_DISTANCE_M + follow_through_distance
    step_advance = required_tip_speed * _PHYSICS_DT
    num_steps = int(math.ceil(total_distance / step_advance)) + _EXTRA_SAFETY_STEPS
    print(f"[proto] total_distance={total_distance:.4f}  step_advance={step_advance:.5f}  num_steps={num_steps}")

    max_ball_speed = 0.0
    max_ball_speed_step = -1
    feedforward = (required_tip_speed * direction_unit).tolist()
    for i in range(num_steps):
        traveled = min((i + 1) * step_advance, total_distance)
        moving_target = backswing_position + traveled * direction_unit
        articulation_api.move_to_pose(moving_target.tolist(), orientation.tolist(), feedforward)
        simulation_app.update()
        linear_vel, _angular_vel = ball_rigid_prim.get_velocities()
        ball_speed = float(np.linalg.norm(np.asarray(linear_vel)[0]))
        if ball_speed > max_ball_speed:
            max_ball_speed = ball_speed
            max_ball_speed_step = i
        if i % 5 == 0:
            actual_pos = np.array(articulation_api.get_end_effector_position())
            pos_err = float(np.linalg.norm(actual_pos - moving_target))
            actual_orientation = articulation_api._get_end_effector_world_orientation()
            q_error = articulation_api._quat_error(actual_orientation, np.asarray(orientation))
            orient_err_deg = math.degrees(2.0 * np.linalg.norm(q_error[1:]))
            velocities = np.asarray(articulation_api._articulation.get_dof_velocities())[0]
            print(f"[proto] i={i} pos_err={pos_err:.5f} orient_err_deg={orient_err_deg:.2f} ball_speed={ball_speed:.4f} vel_norm={float(np.linalg.norm(velocities)):.3f}")

    # 收尾：目標停在 follow_through_position，再跑幾步讓球的碰撞完全結算，
    # 用真實球速做最終驗收。
    for _ in range(60):
        simulation_app.update()
        linear_vel, _angular_vel = ball_rigid_prim.get_velocities()
        ball_speed = float(np.linalg.norm(np.asarray(linear_vel)[0]))
        if ball_speed > max_ball_speed:
            max_ball_speed = ball_speed
            max_ball_speed_step = -2  # 收尾階段

    print(f"[proto] max_ball_speed={max_ball_speed:.4f} m/s at step={max_ball_speed_step}  required={required_tip_speed:.4f}")
    print(f"[proto] speed_error_ratio={abs(max_ball_speed*0 + max_ball_speed - required_tip_speed)/required_tip_speed:.4f}（僅供參考，母球速度跟桿尖速度不會 1:1，重點看有沒有數量級接近）")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
