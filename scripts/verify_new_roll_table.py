"""
scripts/verify_new_roll_table.py — 用真實 Isaac Sim 物理模擬驗證
scripts/build_roll_lookup_table.py 用數值 IK（不含碰撞模型）搜尋出的新 roll
候選表，是否也能通過真實差動 IK 收斂＋沒有碰撞。

背景：scripts/search_ik_reachability.py 證實舊的 9 點 _ROLL_LOOKUP_GRID 選
的 roll 值（0°/15°/45° 這種小角度）會逼 shoulder_pitch／wrist_pitch／
palm_yaw 同時頂死限位，這是先前 20 案例 STRIKE 全滅的根因；改用掃描找出的
roll（±120°~180° 附近）之後，數值 IK 顯示所有七軸關節都有 >0.23 rad 的健康
餘裕。但數值 IK 沒有建模手臂本體碰撞（C1 旋轉時可能掃過庫邊/袋口），這支
腳本補上真正的物理模擬驗證：對 6 個新查表候選點跑完整 AIM（B1→B2→8 段 C1
NLERP→C2），確認差動 IK 真的收斂到位、且沒有卡在碰撞。

沿用 scripts/search_backswing_distance.py 已經除錯過的 `_run_aim()`（含
false-positive 防護：exhaustion 偵測＋最終位置 sanity check）。

用法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/verify_new_roll_table.py
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

# scripts/build_roll_lookup_table.py 搜尋出的新候選（roll_deg），對
# verify_swing_trajectory.py 實際測試的 3x3 網格（扣掉 3 個 y=-1.241425 純
# 幾何無解點）。
_NEW_ROLL_CANDIDATES_DEG = [
    ((-0.606425, -0.9382125), 165),
    ((-0.606425, -0.635), -165),
    ((0.0, -0.9382125), -120),
    ((0.0, -0.635), 150),
    ((0.606425, -0.9382125), -135),
    ((0.606425, -0.635), -165),
]

_AIM_MAX_STEPS = 4000
_STRIKE_MAX_STEPS = 2500


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

    table_base_path = "/World/VerifyRollTable"
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

    _RESET_JOINTS = np.array([[0.0, *CANONICAL_REST_JOINTS]])

    def _hard_reset_joints():
        articulation_api._articulation.set_dof_positions(_RESET_JOINTS)
        articulation_api._articulation.set_dof_velocities(np.zeros((1, 7)))
        for _ in range(10):
            simulation_app.update()

    def _run_aim(cue_ball, shot_angle_deg, rotate_steps=8):
        _hard_reset_joints()
        base_position, base_yaw_rad = compute_base_pose(cue_ball[0], cue_ball[1], shot_angle_deg, _TABLE_Z)
        robot.reposition(base_position)
        for _ in range(30):
            simulation_app.update()

        wrist, orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
            cue_ball, shot_angle_deg, _TABLE_Z, _BALL_RADIUS, [0.0, 0.0]
        )
        if tilt_rad is None:
            return None

        if tilt_rad <= 1e-6:
            grip_position = required_grip_position(cue_ball[0], cue_ball[1], shot_angle_deg)
            joint_targets = [base_yaw_rad, *CANONICAL_REST_JOINTS]
            articulation_api.move_to_joint_position(
                joint_targets, [grip_position[0], grip_position[1], _TABLE_Z + _BALL_RADIUS]
            )
        else:
            safe_joint_targets = [0.0, *CANONICAL_REST_JOINTS]
            safe_target_position = list(compute_canonical_wrist_position(base_position, 0.0))
            roll_rad = cue_pose_calculator.lookup_roll_rad(cue_ball)
            bridge_waypoints = cue_pose_calculator.compute_elevated_bridge_waypoints(
                safe_target_position, list(CANONICAL_FLAT_ORIENTATION),
                cue_ball, shot_angle_deg, _TABLE_Z, _BALL_RADIUS, roll_rad=roll_rad,
                rotate_steps=rotate_steps,
            )
            if bridge_waypoints is None:
                return None
            articulation_api.move_through_poses(
                bridge_waypoints, preceding_joint_targets=(safe_joint_targets, safe_target_position)
            )
            wrist, orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
                cue_ball, shot_angle_deg, _TABLE_Z, _BALL_RADIUS, [0.0, 0.0], roll_rad=roll_rad
            )

        for step in range(_AIM_MAX_STEPS):
            simulation_app.update()
            if articulation_api.is_motion_complete():
                print(f"    [aim] BREAK at step={step} waypoint_index={articulation_api._waypoint_index}")
                break
        else:
            print(f"    [aim] EXHAUSTED _AIM_MAX_STEPS={_AIM_MAX_STEPS} without converging or timing out")
            return None
        if articulation_api.did_last_motion_timeout():
            print("    [aim] TIMEOUT")
            return None
        actual_position = np.array(articulation_api.get_end_effector_position())
        actual_orientation = articulation_api._get_end_effector_world_orientation()
        pos_diff = float(np.linalg.norm(actual_position - wrist))
        q_error = articulation_api._quat_error(actual_orientation, orientation)
        orient_diff = float(2.0 * np.linalg.norm(q_error[1:]))
        print(f"    [aim] pos_diff={pos_diff:.5f}  orient_diff={orient_diff:.5f}")
        if pos_diff > 0.05:
            print(f"    [aim] REJECTED: pos_diff={pos_diff:.5f} > 0.05")
            return None
        return wrist, orientation, tilt_rad

    def _run_strike(wrist, orientation, tilt_rad, shot_angle_deg, backswing_distance=0.15):
        direction_unit = cue_pose_calculator.compute_tilted_direction(shot_angle_deg, tilt_rad)
        waypoints = swing_trajectory_calculator.compute_swing_waypoints(
            contact_position=wrist.tolist(), contact_orientation=orientation.tolist(),
            direction_unit=direction_unit.tolist(), cue_ball_speed=1.995,
            backswing_distance=backswing_distance,
        )
        articulation_api.move_through_poses(waypoints)
        for step in range(_STRIKE_MAX_STEPS):
            simulation_app.update()
            if articulation_api.is_motion_complete():
                print(f"      [strike] BREAK at step={step} waypoint_index={articulation_api._waypoint_index}")
                break
        else:
            print(f"      [strike] EXHAUSTED _STRIKE_MAX_STEPS={_STRIKE_MAX_STEPS}")
            return "INCONCLUSIVE"
        if articulation_api.did_last_motion_timeout():
            return "TIMEOUT"
        final_position = np.array(articulation_api.get_end_effector_position())
        final_target = waypoints[-1].position
        pos_diff = float(np.linalg.norm(final_position - np.array(final_target)))
        # 隨揮終點帶 feedforward 速度，_is_current_target_converged() 現在
        # 允許的有效容許值是 POSITION_TOLERANCE + |feedforward|/POSITION_GAIN
        # （見 docs/issue-180-reachability-analysis.md 第十五節）——沿用同一
        # 公式重算合理門檻，不能再用舊的 0.05 死值，那是修法之前用來抓
        # 假陽性的門檻，現在會誤判正常的新行為。
        feedforward_speed = float(np.linalg.norm(waypoints[-1].linear_velocity))
        expected_tolerance = articulation_api.POSITION_TOLERANCE + feedforward_speed / articulation_api.POSITION_GAIN
        print(f"      [strike] pos_diff={pos_diff:.5f}  expected_tolerance={expected_tolerance:.5f}")
        return "OK" if pos_diff <= expected_tolerance * 1.1 else f"REJECTED(pos_diff={pos_diff:.5f})"

    results = []
    for cue_ball, _old_roll_deg in _NEW_ROLL_CANDIDATES_DEG:
        shot_angle_deg = 0.0
        roll_deg = math.degrees(cue_pose_calculator.lookup_roll_rad(cue_ball))
        print("=" * 100)
        print(f"[{cue_ball}] roll(查表)={roll_deg}deg")
        aim_result = _run_aim(cue_ball, shot_angle_deg)
        if aim_result is None:
            print(f"  AIM_FAILED")
            results.append((cue_ball, roll_deg, "AIM_FAILED", None))
            continue
        print(f"  AIM_OK")
        wrist, orientation, tilt_rad = aim_result
        strike_status = _run_strike(wrist, orientation, tilt_rad, shot_angle_deg)
        print(f"  STRIKE={strike_status}")
        results.append((cue_ball, roll_deg, "AIM_OK", strike_status))

    print("=" * 100)
    print("彙總：")
    for cue_ball, roll_deg, aim_status, strike_status in results:
        print(f"  {cue_ball}  roll={roll_deg}deg  aim={aim_status}  strike={strike_status}")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
