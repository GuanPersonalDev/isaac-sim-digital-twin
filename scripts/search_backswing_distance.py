"""
scripts/search_backswing_distance.py — 調查 STRIKE 階段 0/20 失敗的根因：
`DEFAULT_BACKSWING_DISTANCE_M=0.15` 沿桿身方向退開會讓 `base_yaw`／
`shoulder_pitch`／`palm_yaw` 同時撞死限位（見
docs/issue-180-reachability-analysis.md 第十三節「新發現、尚未解決：後擺
（backswing）距離超出可達範圍」）。這支腳本對已知 AIM 成功的 Kitchen 代表
案例，跑完整 AIM 後測試不同的 `backswing_distance`，找出實際可行的範圍。

用法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/search_backswing_distance.py
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")

for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import math

_TARGET_BALL = (0.0, 0.635)
_TABLE_Z = 0.0
_BALL_RADIUS = 0.028575
_CUE_BALL_SPEED = 1.995  # action_bounds.CUE_BALL_SPEED 中點附近，跟 verify_swing_trajectory.py 稀疏網格的中點案例一致

# 已知 AIM 成功的 Kitchen 代表案例（見上一輪 verify_swing_trajectory.py 20
# 案例結果），(cue_ball_xy, shot_angle_deg)。
_CASES = [
    ((0.0, -0.9382125), 0.0),
]

# 候選後擺距離：0.15 是目前的預設值（已知失敗），逐步縮小找出可行邊界，
# 0.0 當下限對照組（沒有後擺，等同直接從接觸點開始隨揮，驗證揮桿本身沒有
# 問題，只是後擺距離的問題）。
_BACKSWING_CANDIDATES_M = (0.0,)  # 診斷用：先只測 0，找出 AIM/STRIKE 目標不一致的根因

_AIM_MAX_STEPS = 4000  # 11 段 waypoint（B1+B2+8×C1+C2）＋每段最壞情況要跑到自己的
                        # MOTION_TIMEOUT_STEPS=1000 才會真的標記逾時，1200 太短，
                        # 會在真相揭曉前就先把測試迴圈跑完，誤判成「沒逾時=成功」。
_STRIKE_MAX_STEPS = 2500  # 2 段 waypoint，同樣道理，1200 可能不夠讓內部逾時真正發生。
_SHOULDER_PITCH_LIMIT = 1.985
_BASE_YAW_LIMIT = 2.6
_PALM_YAW_LIMIT = 3.0


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

    table_base_path = "/World/BackswingSearchTable"
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
        """瞬間把關節「傳送」回 CANONICAL_REST_JOINTS（不是下 position target
        讓 PD 慢慢追，是直接寫入模擬狀態），並把速度歸零，確保下一次
        _run_aim() 不會被前一次測試殘留的姿態/速度污染。之前的 roll 掃描
        因為同一個 session 連續測試沒有重置，其中幾次的失敗其實是污染造成
        的假結果，見 docs/issue-180-reachability-analysis.md 第十四節。"""
        articulation_api._articulation.set_dof_positions(_RESET_JOINTS)
        articulation_api._articulation.set_dof_velocities(np.zeros((1, 7)))
        for _ in range(10):
            simulation_app.update()

    def _run_aim_custom_start(cue_ball, shot_angle_deg, safe_joints_6dof, rotate_steps=24, roll_rad_override=None, max_steps=4000):
        """探索用：Phase 0 不用 CANONICAL_REST_JOINTS（那組是為了 flat 案例
        的桿身指向精度校正過的，不是為了給高架橋差動 IK 一個好收斂的起點），
        改用任意指定的 6 元組 `safe_joints_6dof`（shoulder_pitch, shoulder_
        yaw, elbow_pitch, wrist_yaw, wrist_pitch, palm_yaw）當安全起點。用
        真實量測的 Phase 0 收斂後位置/朝向（不是分析算出的佔位符）當 B1/B2
        的 current_position/current_orientation，探索階段不需要事先量測
        任何常數。"""
        _RESET = np.array([[0.0, *safe_joints_6dof]])
        articulation_api._articulation.set_dof_positions(_RESET)
        articulation_api._articulation.set_dof_velocities(np.zeros((1, 7)))
        for _ in range(10):
            simulation_app.update()

        base_position, base_yaw_rad = compute_base_pose(cue_ball[0], cue_ball[1], shot_angle_deg, _TABLE_Z)
        robot.reposition(base_position)
        for _ in range(30):
            simulation_app.update()

        roll_rad = cue_pose_calculator.lookup_roll_rad(cue_ball) if roll_rad_override is None else roll_rad_override
        wrist, orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
            cue_ball, shot_angle_deg, _TABLE_Z, _BALL_RADIUS, [0.0, 0.0], roll_rad=roll_rad
        )
        if tilt_rad is None:
            return None

        safe_joint_targets = [0.0, *safe_joints_6dof]
        articulation_api.move_to_joint_position(safe_joint_targets, articulation_api.get_end_effector_position())
        for _ in range(300):
            simulation_app.update()
        if articulation_api.did_last_motion_timeout():
            print("    [custom] Phase 0 自己就逾時，這組安全起點本身不合法")
            return None
        start_position = np.array(articulation_api.get_end_effector_position()).tolist()
        start_orientation = np.array(articulation_api._get_end_effector_world_orientation()).tolist()

        waypoints = cue_pose_calculator.compute_elevated_bridge_waypoints(
            start_position, start_orientation, cue_ball, shot_angle_deg, _TABLE_Z, _BALL_RADIUS,
            roll_rad=roll_rad, rotate_steps=rotate_steps,
        )
        if waypoints is None:
            return None
        articulation_api.move_through_poses(waypoints)

        for step in range(max_steps):
            simulation_app.update()
            if step % 500 == 0:
                joints = np.asarray(articulation_api._articulation.get_dof_positions())[0]
                print(f"    [custom step] step={step} waypoint_index={articulation_api._waypoint_index} joints={np.round(joints,4).tolist()}")
            if articulation_api.is_motion_complete():
                print(f"    [custom step] BREAK at step={step} waypoint_index={articulation_api._waypoint_index}")
                break
        else:
            print(f"    [custom step] EXHAUSTED max_steps={max_steps}")
            return None
        if articulation_api.did_last_motion_timeout():
            return None
        actual_position = np.array(articulation_api.get_end_effector_position())
        pos_diff = float(np.linalg.norm(actual_position - wrist))
        print(f"    [custom diag] pos_diff={pos_diff:.5f}")
        if pos_diff > 0.05:
            return None
        return wrist, orientation, tilt_rad

    def _run_aim(cue_ball, shot_angle_deg, rotate_steps=8, roll_rad_override=None):
        """跟 table_orchestrator._execute_aim() 完全一致的邏輯，跑到 AIM 完成
        （不管 flat 或 bridge）。回傳 (wrist, orientation, tilt_rad) 供後面
        STRIKE 用同一組值。"""
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
            roll_rad = cue_pose_calculator.lookup_roll_rad(cue_ball) if roll_rad_override is None else roll_rad_override
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
            # 高架橋案例的最終 wrist/orientation 也要用同一個 roll_rad 重算，
            # 跟 _execute_strike() 的做法一致。
            wrist, orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
                cue_ball, shot_angle_deg, _TABLE_Z, _BALL_RADIUS, [0.0, 0.0], roll_rad=roll_rad
            )

        for step in range(_AIM_MAX_STEPS):
            simulation_app.update()
            if step % 500 == 0:
                err = (
                    float(np.linalg.norm(np.array(articulation_api.get_end_effector_position()) - articulation_api._target_position))
                    if articulation_api._target_position is not None else -1
                )
                orient_err = -1.0
                if articulation_api._target_position is not None and not articulation_api._is_joint_space_motion:
                    current_orientation = articulation_api._get_end_effector_world_orientation()
                    q_error = articulation_api._quat_error(current_orientation, articulation_api._target_orientation)
                    orient_err = float(2.0 * np.linalg.norm(q_error[1:]))
                current_joints = np.asarray(articulation_api._articulation.get_dof_positions())[0]
                print(
                    f"    [aim step] step={step} waypoint_index={articulation_api._waypoint_index} "
                    f"is_joint_space={articulation_api._is_joint_space_motion} pos_err={err:.5f} orient_err={orient_err:.5f} "
                    f"is_complete={articulation_api.is_motion_complete()} timed_out={articulation_api.did_last_motion_timeout()} "
                    f"joints={np.round(current_joints, 4).tolist()}"
                )
            if articulation_api.is_motion_complete():
                print(f"    [aim step] BREAK at step={step} waypoint_index={articulation_api._waypoint_index}")
                break
        else:
            # 迴圈跑完 _AIM_MAX_STEPS 都沒 break：既沒收斂也沒等到內部逾時，
            # 是「預算不夠、真相還沒揭曉」的不確定狀態，不能當成功回傳
            # （這正是之前假陽性的成因）。
            print(f"    [aim step] EXHAUSTED _AIM_MAX_STEPS={_AIM_MAX_STEPS} without converging or timing out")
            return None
        if articulation_api.did_last_motion_timeout():
            return None
        actual_position = np.array(articulation_api.get_end_effector_position())
        actual_orientation = articulation_api._get_end_effector_world_orientation()
        pos_diff = float(np.linalg.norm(actual_position - wrist))
        print(
            f"    [aim diag] computed_wrist={wrist.tolist()} actual_position={actual_position.tolist()} "
            f"pos_diff={pos_diff:.5f}"
        )
        print(
            f"    [aim diag] computed_orientation={orientation.tolist()} actual_orientation={actual_orientation.tolist()}"
        )
        if pos_diff > 0.05:
            # 雖然 is_motion_complete()=True、也沒被判定逾時，但實際位置離
            # 目標還有 5cm 以上——這代表卡在中繼 waypoint 就被誤判完成（跟
            # 「1.4 公尺」那次同一類問題），不是真的到位。
            print(f"    [aim diag] REJECTED: pos_diff={pos_diff:.5f} > 0.05，判定為未真正收斂")
            return None
        return wrist, orientation, tilt_rad

    def _run_strike(wrist, orientation, tilt_rad, shot_angle_deg, backswing_distance, verbose=True):
        direction_unit = cue_pose_calculator.compute_tilted_direction(shot_angle_deg, tilt_rad)
        waypoints = swing_trajectory_calculator.compute_swing_waypoints(
            contact_position=wrist.tolist(), contact_orientation=orientation.tolist(),
            direction_unit=direction_unit.tolist(), cue_ball_speed=_CUE_BALL_SPEED,
            backswing_distance=backswing_distance,
        )
        print(f"    [strike diag] waypoint[0].position={waypoints[0].position} orientation={waypoints[0].orientation}")
        print(f"    [strike diag] waypoint[1].position={waypoints[1].position} linear_velocity={waypoints[1].linear_velocity}")
        articulation_api.move_through_poses(waypoints)

        max_base_yaw = max_shoulder_pitch = max_palm_yaw = -999.0
        settled_step = None
        for step in range(_STRIKE_MAX_STEPS):
            simulation_app.update()
            joints = np.asarray(articulation_api._articulation.get_dof_positions())[0]
            max_base_yaw = max(max_base_yaw, abs(float(joints[0])))
            max_shoulder_pitch = max(max_shoulder_pitch, float(joints[1]))
            max_palm_yaw = max(max_palm_yaw, abs(float(joints[6])))
            if step % 300 == 0 or verbose:
                print(f"      [strike step] step={step} waypoint_index={articulation_api._waypoint_index} joints={np.round(joints,4).tolist()}")
            if articulation_api.is_motion_complete():
                settled_step = step
                print(f"      [strike step] BREAK at step={step} waypoint_index={articulation_api._waypoint_index}")
                break
        else:
            print(f"      [strike step] EXHAUSTED _STRIKE_MAX_STEPS={_STRIKE_MAX_STEPS} without converging or timing out")
            return {"status": "INCONCLUSIVE", "settled_step": None, "waypoint_index": articulation_api._waypoint_index,
                    "max_base_yaw": max_base_yaw, "max_shoulder_pitch": max_shoulder_pitch, "max_palm_yaw": max_palm_yaw}
        timed_out = articulation_api.did_last_motion_timeout()
        final_position = np.array(articulation_api.get_end_effector_position())
        final_target = waypoints[-1].position
        pos_diff = float(np.linalg.norm(final_position - np.array(final_target)))
        print(f"      [strike diag] final_position={final_position.tolist()} final_target={final_target} pos_diff={pos_diff:.5f}")
        if timed_out or pos_diff > 0.05:
            status = "TIMEOUT" if timed_out else "REJECTED(pos_diff>0.05)"
        else:
            status = "OK"
        return {
            "status": status,
            "settled_step": settled_step,
            "waypoint_index": articulation_api._waypoint_index,
            "max_base_yaw": max_base_yaw,
            "max_shoulder_pitch": max_shoulder_pitch,
            "max_palm_yaw": max_palm_yaw,
        }

    # 13 個 roll 值（含之前測過的 0°/45°）全部真正收斂失敗，證實問題不是
    # roll 選擇——換一個安全起點候選（不再侷限 CANONICAL_REST_JOINTS，那組
    # 是為 flat 案例的桿身指向精度校正過的，跟「給差動 IK 一個好起點」是
    #兩個不同的目標，見 docs/issue-180-reachability-analysis.md 第十四節）。
    # 目標：roll=0 時卡住的是 wrist_pitch 撞下限(-1.5707)、palm_yaw 撞上限
    # (3.0)——起點往反方向偏移，理論上能換到更多可用空間。
    cue_ball, shot_angle_deg = _CASES[0]
    # 系統化網格：固定 shoulder_pitch/elbow_pitch=baseline、roll=0，掃
    # wrist_pitch × palm_yaw 找真正可行的安全起點（縮短 max_steps 加快搜尋，
    # 2500 步已經足夠讓真正的逾時發生，見前幾輪實測）。
    # 前一輪固定 shoulder_pitch=1.9/elbow_pitch=1.8 掃 wrist_pitch×palm_yaw
    # 9 組全部失敗——這輪把 shoulder_pitch/elbow_pitch 也一起納入（4 個關節
    # 同時調整），這才是真正涵蓋全部相關關節的組合，不是只調其中一半。
    _CANDIDATES_4D = [
        (1.9, 1.8, -0.5585, 1.5010),   # baseline
        (1.1, 2.7, 0.0, 0.0),          # sp 最低+ep 最高+wrist 置中
        (1.1, 2.7, 0.5585, -1.5010),   # sp 最低+ep 最高+wrist 反向
        (1.5, 2.4, 0.0, 0.0),          # 中間值組合
    ]
    found = None
    for shoulder_pitch, elbow_pitch, wrist_pitch, palm_yaw in _CANDIDATES_4D:
        for _dummy in [0]:  # 保留巢狀結構方便之後擴充
            safe_joints_6dof = (shoulder_pitch, 0.0, elbow_pitch, 0.0, wrist_pitch, palm_yaw)
            label = f"sp={shoulder_pitch:.2f},ep={elbow_pitch:.2f},wp={wrist_pitch:.3f},py={palm_yaw:.3f}"
            print(f"=== {label} ===")
            result = _run_aim_custom_start(
                cue_ball, shot_angle_deg, safe_joints_6dof, rotate_steps=24, roll_rad_override=0.0, max_steps=2500
            )
            if result is None:
                print(f"  [GRID RESULT] {label}  AIM_FAILED")
                continue
            print(f"  [GRID RESULT] {label}  AIM_CONVERGED（真正收斂，非假陽性）")
            found = (label, safe_joints_6dof, result)
            break
        if found:
            break

    if found is None:
        print("=== 4 組全關節組合全部失敗 ===")
    else:
        label, safe_joints_6dof, result = found
        wrist, orientation, tilt_rad = result
        for backswing_distance in _BACKSWING_CANDIDATES_M:
            print(f"  --- backswing_distance={backswing_distance} ---")
            strike_result = _run_strike(wrist, orientation, tilt_rad, shot_angle_deg, backswing_distance, verbose=False)
            print(
                f"  [STRIKE RESULT] {label} backswing={backswing_distance} "
                f"status={strike_result['status']} waypoint_index={strike_result['waypoint_index']}"
            )
            if strike_result["status"] == "OK":
                break


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
