"""
scripts/verify_swing_trajectory.py — Issue #181 揮桿軌跡生成的端到端驗證。

比照 docs/WAM_IK_implementation_and_verification.md 第 3 節的驗收方法：找出
桿尖跟母球距離最小的 timestep 當接觸時刻，用 Jacobian×關節角速度＋ω×r（桿尖
相對握把的槓桿臂）算桿尖實際速度，比對目標值（位置≤球半徑/10、方向 3-5°、
速度 ±10-15%）。

這支腳本直接呼叫 core/services/cue_pose_calculator.py、
core/services/swing_trajectory_calculator.py 與
extension/isaac_sim_impl_6_0/articulation_api_impl.py 的 move_to_joint_
position()/move_through_poses()，不經過完整的 DemoTableOrchestrator（那需要
組裝 TableBallSet/ScriptController 等一整套依賴，跟這裡想驗證的「軌跡規劃
本身對不對」是兩件事），跟 scripts/scan_elevated_bridge_approach.py 同一種
研究腳本風格。

用法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/verify_swing_trajectory.py
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")

for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import math
import itertools

# 稀疏測試網格：邊界+中點，避免全交叉組合爆炸（單次 headless 執行需控制在
# 10 分鐘上限內，比照 scan_elevated_bridge_approach.py 的 ROLL_CANDIDATES_DEG
# 限制在 3 個候選值同一個考量）。
_CUE_BALL_X_CANDIDATES = None  # 執行時從 action_bounds 動態算出，見 _build_test_cases()
_SHOT_ANGLE_CANDIDATES_DEG = (-30.0, -15.0, 0.0, 15.0, 30.0)
_POSITION_OFFSET_CANDIDATES = ([0.0, 0.0], [0.4, 0.0], [-0.4, 0.0])

# 驗收容許值，見 docs/WAM_IK_implementation_and_verification.md 第 3.3 節。
_BALL_RADIUS = 0.028575
_POSITION_TOLERANCE_M = _BALL_RADIUS / 10.0
_DIRECTION_TOLERANCE_DEG = 5.0
_SPEED_TOLERANCE_RATIO = 0.15

_AIM_MAX_STEPS = 1200
_STRIKE_MAX_STEPS = 1200


def _build_test_cases():
    from core.models.action_bounds import (
        CUE_BALL_PLACEMENT_X, CUE_BALL_PLACEMENT_Y, CUE_BALL_SPEED,
    )

    x_candidates = (CUE_BALL_PLACEMENT_X[0], sum(CUE_BALL_PLACEMENT_X) / 2, CUE_BALL_PLACEMENT_X[1])
    y_candidates = (CUE_BALL_PLACEMENT_Y[0], sum(CUE_BALL_PLACEMENT_Y) / 2, CUE_BALL_PLACEMENT_Y[1])
    speed_candidates = (CUE_BALL_SPEED[0], sum(CUE_BALL_SPEED) / 2, CUE_BALL_SPEED[1])

    cases = []
    # 稀疏交叉：位置用完整 3x3 網格，角度/速度/偏移各自只搭配「中心位置」，
    # 避免全交叉組合數量爆炸。
    center_x, center_y = x_candidates[1], y_candidates[1]
    for x, y in itertools.product(x_candidates, y_candidates):
        cases.append({"cue_ball": (x, y), "shot_angle_deg": 0.0, "cue_ball_speed": speed_candidates[1], "position_offset": [0.0, 0.0]})
    for shot_angle_deg in _SHOT_ANGLE_CANDIDATES_DEG:
        cases.append({"cue_ball": (center_x, center_y), "shot_angle_deg": shot_angle_deg, "cue_ball_speed": speed_candidates[1], "position_offset": [0.0, 0.0]})
    for cue_ball_speed in speed_candidates:
        cases.append({"cue_ball": (center_x, center_y), "shot_angle_deg": 0.0, "cue_ball_speed": cue_ball_speed, "position_offset": [0.0, 0.0]})
    for position_offset in _POSITION_OFFSET_CANDIDATES:
        cases.append({"cue_ball": (center_x, center_y), "shot_angle_deg": 0.0, "cue_ball_speed": speed_candidates[1], "position_offset": position_offset})
    return cases


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import Usd, UsdPhysics, Sdf

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
        CANONICAL_FLAT_ORIENTATION, CANONICAL_REST_JOINTS, compute_base_pose, compute_canonical_wrist_position,
    )
    from core.services import cue_pose_calculator, swing_trajectory_calculator

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()
    physics_api = PhysicsAPIImpl()

    table_base_path = "/World/VerifySwingTable"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))
    table_z = 0.0

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

    cue_stick_prim_path = robot_manager.get_cue_stick_prim_path()
    physics_api.enable_contact_reporting(cue_stick_prim_path)
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    for prim in Usd.PrimRange(robot_prim):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            physics_api.enable_contact_reporting(prim.GetPath().pathString)
    contacts: list[ContactEvent] = []
    physics_api.subscribe_contact_events(lambda e: contacts.append(e))

    def _is_self_path(p: str) -> bool:
        return p.startswith(robot_prim_path) or p.startswith(cue_stick_prim_path)

    def _shot_direction_deg_to_forward(shot_angle_deg):
        theta = math.radians(shot_angle_deg)
        return np.array([-math.sin(theta), math.cos(theta), 0.0])

    def _run_aim(cue_ball, shot_angle_deg, position_offset):
        base_position, base_yaw_rad = compute_base_pose(cue_ball[0], cue_ball[1], shot_angle_deg, table_z)
        robot.reposition(base_position)
        for _ in range(30):
            simulation_app.update()

        wrist, orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
            cue_ball, shot_angle_deg, table_z, _BALL_RADIUS, position_offset
        )
        if tilt_rad is None:
            return {"status": "GEOMETRICALLY_INFEASIBLE"}

        contacts.clear()
        if tilt_rad <= 1e-6:
            from core.services.base_placement_calculator import required_grip_position
            grip_position = required_grip_position(cue_ball[0], cue_ball[1], shot_angle_deg)
            joint_targets = [base_yaw_rad, *CANONICAL_REST_JOINTS]
            articulation_api.move_to_joint_position(
                joint_targets, [grip_position[0], grip_position[1], table_z + _BALL_RADIUS]
            )
        else:
            safe_joint_targets = [0.0, *CANONICAL_REST_JOINTS]
            safe_target_position = list(compute_canonical_wrist_position(base_position, 0.0))
            # 跟 table_orchestrator._execute_aim() 用同一支查表函式選 roll，
            # 保證這支驗證腳本忠實代表正式流程的行為。
            roll_rad = cue_pose_calculator.lookup_roll_rad(cue_ball)
            bridge_waypoints = cue_pose_calculator.compute_elevated_bridge_waypoints(
                safe_target_position, list(CANONICAL_FLAT_ORIENTATION),
                cue_ball, shot_angle_deg, table_z, _BALL_RADIUS,
                position_offset=position_offset, roll_rad=roll_rad,
            )
            if bridge_waypoints is None:
                return {"status": "GEOMETRICALLY_INFEASIBLE"}
            articulation_api.move_through_poses(
                bridge_waypoints, preceding_joint_targets=(safe_joint_targets, safe_target_position)
            )

        settled_step = None
        for step in range(_AIM_MAX_STEPS):
            simulation_app.update()
            if articulation_api.is_motion_complete():
                settled_step = step
                break

        timed_out = articulation_api.did_last_motion_timeout()
        blocking_partners = {p for c in contacts for p in (c.collider_path_a, c.collider_path_b) if not _is_self_path(p) and "Surface" not in p}
        status = "AIM_TIMEOUT" if timed_out else ("COLLISION" if blocking_partners else "OK")
        return {
            "status": status, "settled_step": settled_step, "tilt_rad": tilt_rad,
            "partners": sorted(blocking_partners), "wrist": wrist, "orientation": orientation,
        }

    def _run_strike(cue_ball, shot_angle_deg, cue_ball_speed, position_offset):
        wrist, orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
            cue_ball, shot_angle_deg, table_z, _BALL_RADIUS, position_offset
        )
        if tilt_rad is None:
            return {"status": "GEOMETRICALLY_INFEASIBLE"}
        if tilt_rad > 1e-6:
            # 跟 table_orchestrator._execute_strike() 一樣：flat 案例維持
            # roll_rad=0，高架橋案例查表重算一次拿正確的 wrist/orientation。
            roll_rad = cue_pose_calculator.lookup_roll_rad(cue_ball)
            wrist, orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
                cue_ball, shot_angle_deg, table_z, _BALL_RADIUS, position_offset, roll_rad=roll_rad
            )

        direction_unit = cue_pose_calculator.compute_tilted_direction(shot_angle_deg, tilt_rad)
        waypoints = swing_trajectory_calculator.compute_swing_waypoints(
            contact_position=wrist.tolist(), contact_orientation=orientation.tolist(),
            direction_unit=direction_unit.tolist(), cue_ball_speed=cue_ball_speed,
        )

        contacts.clear()
        articulation_api.move_through_poses(waypoints)

        # 桿尖相對握把（=end-effector）的偏移向量：direction_unit *
        # CUE_STICK_GRIP_TO_TIP。之前這裡直接把 get_end_effector_position()
        # （腕部位置）當桿尖位置跟母球比距離，漏了這個 1.35m 的偏移量，
        # 算出的 position_error 整個對不上（誤差量級剛好落在 1.35m 附近）；
        # contact_index 也因此選到「腕部離球最近」而不是「桿尖離球最近」的
        # 錯誤時間點，連帶讓 actual_speed 量到不相干時刻的速度。
        from core.services.base_placement_calculator import CUE_STICK_GRIP_TO_TIP as _CUE_STICK_GRIP_TO_TIP
        tip_offset = direction_unit * _CUE_STICK_GRIP_TO_TIP
        tip_positions = []
        for step in range(_STRIKE_MAX_STEPS):
            simulation_app.update()
            tip_positions.append(np.array(articulation_api.get_end_effector_position()) + tip_offset)
            if step % 100 == 0:
                current_joints = np.asarray(articulation_api._articulation.get_dof_positions())[0]
                target_pos = articulation_api._target_position
                target_orient = articulation_api._target_orientation
                actual_pos = np.array(articulation_api.get_end_effector_position())
                actual_orient = articulation_api._get_end_effector_world_orientation()
                print(
                    f"    [strike diag] step={step} waypoint_index={articulation_api._waypoint_index} "
                    f"is_complete={articulation_api.is_motion_complete()} "
                    f"pos_err={float(np.linalg.norm(actual_pos - target_pos)):.5f} "
                    f"target_pos={target_pos.tolist()} actual_pos={actual_pos.tolist()} "
                    f"target_orient={target_orient.tolist()} actual_orient={actual_orient.tolist()} "
                    f"joints={np.round(current_joints, 4).tolist()}"
                )
            if articulation_api.is_motion_complete():
                break
        print(f"    [strike diag] FINAL step={step} waypoint_index={articulation_api._waypoint_index} timed_out={articulation_api.did_last_motion_timeout()}")

        ball_center = np.array([cue_ball[0], cue_ball[1], table_z + _BALL_RADIUS])
        distances = [np.linalg.norm(p - ball_center) for p in tip_positions]
        contact_index = int(np.argmin(distances))

        jacobians = np.asarray(articulation_api._articulation.get_jacobian_matrices().numpy())[0]
        J = jacobians[articulation_api._jac_link_index]
        dof_velocities_raw = articulation_api._articulation.get_dof_velocities()
        if hasattr(dof_velocities_raw, "numpy"):
            dof_velocities_raw = dof_velocities_raw.numpy()
        joint_velocities = np.asarray(dof_velocities_raw)[0]
        end_effector_twist = J @ joint_velocities
        end_effector_linear = end_effector_twist[:3]
        end_effector_angular = end_effector_twist[3:]

        # 桿尖相對握把的偏移向量：direction_unit * CUE_STICK_GRIP_TO_TIP。
        from core.services.base_placement_calculator import CUE_STICK_GRIP_TO_TIP
        r = direction_unit * CUE_STICK_GRIP_TO_TIP
        actual_tip_velocity = end_effector_linear + np.cross(end_effector_angular, r)

        required_tip_speed = swing_trajectory_calculator.compute_required_tip_speed(cue_ball_speed)
        target_velocity = required_tip_speed * direction_unit

        position_error = float(distances[contact_index])
        actual_speed = float(np.linalg.norm(actual_tip_velocity))
        speed_error_ratio = abs(actual_speed - required_tip_speed) / required_tip_speed if required_tip_speed > 0 else float("nan")
        cos_angle = float(np.dot(actual_tip_velocity, target_velocity) / (actual_speed * required_tip_speed + 1e-8))
        direction_error_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))

        blocking_partners = {p for c in contacts for p in (c.collider_path_a, c.collider_path_b) if not _is_self_path(p) and "Surface" not in p}
        passed = (
            position_error <= _POSITION_TOLERANCE_M
            and direction_error_deg <= _DIRECTION_TOLERANCE_DEG
            and speed_error_ratio <= _SPEED_TOLERANCE_RATIO
            and not blocking_partners
        )
        return {
            "status": "OK" if passed else "FAIL",
            "position_error_mm": position_error * 1000.0,
            "direction_error_deg": direction_error_deg,
            "speed_error_ratio": speed_error_ratio,
            "actual_speed": actual_speed,
            "required_tip_speed": required_tip_speed,
            "partners": sorted(blocking_partners),
        }

    test_cases = _build_test_cases()
    results = []
    for case in test_cases:
        cue_ball = case["cue_ball"]
        shot_angle_deg = case["shot_angle_deg"]
        cue_ball_speed = case["cue_ball_speed"]
        position_offset = case["position_offset"]

        aim_result = _run_aim(cue_ball, shot_angle_deg, position_offset)
        print(f"[AIM] cue_ball={cue_ball} shot_angle={shot_angle_deg} offset={position_offset} => {aim_result['status']}")

        if aim_result["status"] not in ("OK",):
            results.append({"case": case, "aim": aim_result, "strike": None})
            continue

        strike_result = _run_strike(cue_ball, shot_angle_deg, cue_ball_speed, position_offset)
        print(
            f"[STRIKE] cue_ball={cue_ball} shot_angle={shot_angle_deg} speed={cue_ball_speed:.3f} "
            f"offset={position_offset} => {strike_result['status']} {strike_result}"
        )
        results.append({"case": case, "aim": aim_result, "strike": strike_result})

    n_total = len(results)
    n_aim_ok = sum(1 for r in results if r["aim"]["status"] == "OK")
    n_strike_ok = sum(1 for r in results if r["strike"] is not None and r["strike"]["status"] == "OK")
    n_aim_timeout = sum(1 for r in results if r["aim"]["status"] == "AIM_TIMEOUT")
    print(
        f"\n=== FINAL SUMMARY: total={n_total} aim_ok={n_aim_ok} strike_ok={n_strike_ok} "
        f"aim_timeout={n_aim_timeout} ==="
    )

    physics_api.unsubscribe_contact_events()


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
