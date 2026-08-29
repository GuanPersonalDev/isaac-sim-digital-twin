"""
scripts/diagnose_move_swing.py — 驗證新實作的 `ArticulationAPIImpl.
move_swing()`（揮桿專用速度最優控制，見 docs/issue-180-reachability-
analysis.md 第十六節）：對最難的 Kitchen 案例（y=-0.9382125，24 個 roll
候選裡沒有任何一個在「姿態完全鎖死」下能達到所需速度）用真實母球物理
速度（不透過任何軟體完成判定）驗收，取代舊的 compute_swing_waypoints()
+ move_through_poses() 兩段式呼叫。

用法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/diagnose_move_swing.py
"""

import logging
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
_SWING_MAX_STEPS = 400
_BACKSWING_DISTANCE_M = 0.15
_ORIENTATION_GAIN = float(os.environ.get("SWING_ORIENTATION_GAIN", "1.0"))
_MAX_ANGULAR_SPEED = float(os.environ.get("SWING_MAX_ANGULAR_SPEED", "1.0"))


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf
    from isaacsim.core.experimental.prims import RigidPrim

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("isaac_sim_impl_6_0.articulation_api_impl").setLevel(logging.DEBUG)

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

    table_base_path = "/World/DiagnoseMoveSwingTable"
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

    # 直接查詢執行時的碰撞相關 USD attribute，確認球桿的 Cylinder／母球
    # 有沒有真的開啟碰撞、有沒有被意外過濾。
    cue_stick_cylinder_path = robot_manager.get_cue_stick_prim_path() + "/Cylinder"
    cylinder_prim = stage.GetPrimAtPath(cue_stick_cylinder_path)
    ball_prim_for_check = stage.GetPrimAtPath(ball_prim_path + "/Ball")
    for label, prim in [("CueStick/Cylinder", cylinder_prim), ("Ball", ball_prim_for_check)]:
        if not prim.IsValid():
            print(f"[diag] {label} prim 不存在：{prim.GetPath()}")
            continue
        collision_enabled_attr = prim.GetAttribute("physics:collisionEnabled")
        has_collision_api = prim.HasAPI(UsdPhysics.CollisionAPI)
        filtered_pairs_rel = prim.GetRelationship("physics:filteredPairs")
        filtered_targets = filtered_pairs_rel.GetTargets() if filtered_pairs_rel else []
        print(
            f"[diag] {label}: path={prim.GetPath()} has_CollisionAPI={has_collision_api} "
            f"collisionEnabled={collision_enabled_attr.Get() if collision_enabled_attr and collision_enabled_attr.IsValid() else 'N/A(預設True)'} "
            f"filteredPairs={filtered_targets}"
        )

    # ⚠️ 實驗性檢查：CCD（Continuous Collision Detection）在整個專案裡
    # 從未被設定過（預設關閉）。球桿 Cylinder 半徑只有 0.01m（很細）、
    # 揮桿速度快，標準離散碰撞偵測（只在時間步起訖點取樣）對這種細長、
    # 高速的物體容易「穿透而過」——幾何上算出來有重疊（廣相位偵測到，
    # ContactEvent 因此觸發），但窄相位求解器可能因為沒有真正的滲透深度
    # 而算出零衝量。用環境變數 SWING_ENABLE_CCD=1 測試開啟 CCD 會不會
    # 修好這個問題，要在 timeline.play() 之前設定。
    if os.environ.get("SWING_ENABLE_CCD"):
        cue_stick_rigid_prim_for_ccd = stage.GetPrimAtPath(robot_manager.get_cue_stick_prim_path())
        ball_rigid_prim_for_ccd = stage.GetPrimAtPath(ball_prim_path)
        for label, prim in [("CueStick", cue_stick_rigid_prim_for_ccd), ("Ball", ball_rigid_prim_for_ccd)]:
            ccd_attr = prim.GetAttribute("physxRigidBody:enableCCD")
            if not ccd_attr or not ccd_attr.IsValid():
                print(f"[diag] {label} 沒有 physxRigidBody:enableCCD attribute，跳過")
                continue
            print(f"[diag] {label} CCD BEFORE={ccd_attr.Get()}")
            ccd_attr.Set(True)
            print(f"[diag] {label} CCD AFTER={ccd_attr.Get()}")

    contacts: list[tuple[int, ContactEvent]] = []
    _step_counter = {"value": -1}  # AIM 開始前是 -1，之後由主迴圈更新
    physics_api.enable_contact_reporting(ball_prim_path)
    physics_api.enable_contact_reporting(robot_manager.get_cue_stick_prim_path())
    physics_api.subscribe_contact_events(lambda e: contacts.append((_step_counter["value"], e)))

    # ⚠️ assets/barrett_wam/wam7/payloads/Physics/physics.usda 把
    # solverPositionIterationCount／solverVelocityIterationCount 都設成
    # 255（PhysX 上限），是之前修正 move_to_joint_position() 大幅度
    # joint-space 移動殘留誤差用的（見 scripts/probe_solver_iterations.py），
    # 但也正好對應到整個 session 反覆出現的「more than 4 velocity
    # iterations being added to a TGS scene」警告——懷疑跟這次的零衝量
    # 問題有關（TGS 求解器正常建議低 velocity iteration，Isaac Sim 預設
    # 是 1）。這裡在 timeline.play() 之前只覆寫 velocity 迭代次數（保留
    # position=255 不動，避免重新踩到原本的 joint-space 收斂問題），
    # 用環境變數 SWING_VELOCITY_ITERATIONS 方便快速測試不同值。必須在
    # timeline.play()／Articulation tensor view 建立之前設定，PhysX 只在
    # cook articulation 當下讀取一次，不是每個 tick 動態生效。
    velocity_iterations_override = os.environ.get("SWING_VELOCITY_ITERATIONS")
    if velocity_iterations_override is not None:
        from pxr import Usd
        robot_prim = stage.GetPrimAtPath(robot_prim_path)
        articulation_root_prim = None
        for prim in Usd.PrimRange(robot_prim):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_root_prim = prim
                break
        if articulation_root_prim is None:
            print("[diag] WARNING: 找不到 ArticulationRootAPI prim，無法覆寫 solver iteration")
        else:
            vel_attr = articulation_root_prim.GetAttribute("physxArticulation:solverVelocityIterationCount")
            pos_attr = articulation_root_prim.GetAttribute("physxArticulation:solverPositionIterationCount")
            print(f"[diag] solver iteration BEFORE: position={pos_attr.Get()} velocity={vel_attr.Get()}")
            vel_attr.Set(int(velocity_iterations_override))
            print(f"[diag] solver iteration AFTER: position={pos_attr.Get()} velocity={vel_attr.Get()}")

    # ⚠️ 實驗性檢查：每個關節的 PD 驅動 stiffness=1745.3292（位置增益）
    # 非常高（見 assets/barrett_wam/wam7/payloads/Physics/physics.usda），
    # 懷疑即使在 velocity 控制模式下，若 targetPosition 沒有隨當下實際
    # 位置持續更新，殘留的位置誤差乘上這麼高的 stiffness 會產生巨大修正力，
    # 讓手臂在碰撞瞬間表現得像無限剛性的牆，吸收掉碰撞衝量而不傳給球。
    # 用環境變數 SWING_JOINT_STIFFNESS 測試調低/歸零這個值會不會修好
    # 零衝量問題，同樣要在 timeline.play() 之前設定。
    joint_stiffness_override = os.environ.get("SWING_JOINT_STIFFNESS")
    if joint_stiffness_override is not None:
        from pxr import Usd
        physics_root = stage.GetPrimAtPath(robot_prim_path)
        joint_count = 0
        for prim in Usd.PrimRange(physics_root):
            stiffness_attr = prim.GetAttribute("drive:angular:physics:stiffness")
            if stiffness_attr and stiffness_attr.IsValid():
                stiffness_attr.Set(float(joint_stiffness_override))
                joint_count += 1
        print(f"[diag] 已把 {joint_count} 個關節的 drive stiffness 改成 {joint_stiffness_override}")

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
    print(f"[diag] roll_deg={math.degrees(roll_rad):.1f}")
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
        _step_counter["value"] = f"aim:{step}"
        simulation_app.update()
        if articulation_api.is_motion_complete():
            break
    print(f"[diag] AIM done at step={step}  timed_out={articulation_api.did_last_motion_timeout()}")

    direction_unit = np.array(cue_pose_calculator.compute_tilted_direction(_SHOT_ANGLE_DEG, tilt_rad))
    required_tip_speed = swing_trajectory_calculator.compute_required_tip_speed(_CUE_BALL_SPEED)
    follow_through_distance = swing_trajectory_calculator.compute_follow_through_distance(required_tip_speed)
    contact_position = np.array(wrist)
    backswing_position = swing_trajectory_calculator.compute_backswing_position(
        contact_position, direction_unit, _BACKSWING_DISTANCE_M
    )
    follow_through_position = contact_position + follow_through_distance * direction_unit

    print(f"[diag] required_tip_speed={required_tip_speed:.4f}  orientation_gain={_ORIENTATION_GAIN}  max_angular_speed={_MAX_ANGULAR_SPEED}")
    print(f"[diag] backswing={backswing_position.tolist()}  swing_end={follow_through_position.tolist()}")

    articulation_api.move_swing(
        backswing_position.tolist(), orientation.tolist(), follow_through_position.tolist(),
        orientation_gain=_ORIENTATION_GAIN, max_angular_speed=_MAX_ANGULAR_SPEED,
    )

    from core.services.base_placement_calculator import CUE_STICK_GRIP_TO_TIP
    # 揮桿允許姿態漂移，不能用固定的 nominal direction_unit 當桿尖偏移量
    # （1.35m 槓桿臂，角度誤差會被放大成很大的位置誤差）——跟
    # ArticulationAPIImpl._rotate_vector_by_quat() 同一套做法，用「當下
    # 實際姿態」把桿身局部 +Y 軸（direction_unit 定義時的參考軸，見
    # cue_pose_calculator.compute_tilted_wrist_pose() 的 _shortest_arc_quat
    # 呼叫）轉到世界座標，才是桿尖真正的即時方向。
    local_y_axis = np.array([0.0, 1.0, 0.0])
    ball_center = np.array([_CUE_BALL[0], _CUE_BALL[1], _TABLE_Z + _BALL_RADIUS])
    min_tip_to_ball = float("inf")
    min_tip_to_ball_step = -1
    max_ball_speed = 0.0
    max_ball_speed_step = -1
    max_orient_err_deg = 0.0
    for step in range(_SWING_MAX_STEPS):
        _step_counter["value"] = f"swing:{step}(is_swing_motion={articulation_api._is_swing_motion})"
        simulation_app.update()
        linear_vel, _angular_vel = ball_rigid_prim.get_velocities()
        ball_speed = float(np.linalg.norm(np.asarray(linear_vel)[0]))
        if ball_speed > max_ball_speed:
            max_ball_speed = ball_speed
            max_ball_speed_step = step
        current_orientation_for_tip = articulation_api._get_end_effector_world_orientation()
        current_tip_direction = articulation_api._rotate_vector_by_quat(current_orientation_for_tip, local_y_axis)
        tip_position = np.array(articulation_api.get_end_effector_position()) + current_tip_direction * CUE_STICK_GRIP_TO_TIP
        tip_to_ball = float(np.linalg.norm(tip_position - ball_center))
        if tip_to_ball < min_tip_to_ball:
            min_tip_to_ball = tip_to_ball
            min_tip_to_ball_step = step
        current_orientation = articulation_api._get_end_effector_world_orientation()
        q_error = articulation_api._quat_error(current_orientation, np.asarray(orientation))
        orient_err_deg = math.degrees(2.0 * np.linalg.norm(q_error[1:]))
        max_orient_err_deg = max(max_orient_err_deg, orient_err_deg)
        joint_velocities = np.asarray(articulation_api._articulation.get_dof_velocities())[0]
        if step % 2 == 0 or (35 <= step <= 65):
            cue_stick_pose = articulation_api._get_cue_stick_world_pose()
            cue_stick_pos = cue_stick_pose[0] if cue_stick_pose else None
            cue_stick_orient = cue_stick_pose[1] if cue_stick_pose else None
            wrist_orient = current_orientation.tolist()
            # 桿尖用「球桿自己回報的姿態」重算一次，跟用「腕部姿態」算出來的
            # 版本對照——如果兩者算出來的桿尖位置有落差，代表 FixedJoint
            # 兩端的姿態沒有像位置一樣完全同步，是真正碰不到球的原因。
            if cue_stick_orient is not None:
                cue_stick_tip_dir = articulation_api._rotate_vector_by_quat(np.array(cue_stick_orient), local_y_axis)
                cue_stick_tip_pos = np.array(cue_stick_pos) + cue_stick_tip_dir * CUE_STICK_GRIP_TO_TIP
                tip_to_ball_via_cuestick = float(np.linalg.norm(cue_stick_tip_pos - ball_center))
            else:
                tip_to_ball_via_cuestick = None
            print(
                f"[diag] step={step} ball_speed={ball_speed:.4f} tip_to_ball={tip_to_ball:.4f} "
                f"tip_to_ball_via_cuestick={tip_to_ball_via_cuestick} orient_err_deg={orient_err_deg:.2f} "
                f"is_complete={articulation_api.is_motion_complete()} is_swing_motion={articulation_api._is_swing_motion} "
                f"joint_vel_norm={float(np.linalg.norm(joint_velocities)):.3f}\n"
                f"    wrist_pos={np.round(np.array(articulation_api.get_end_effector_position()),4).tolist()} wrist_orient={np.round(wrist_orient,4).tolist()}\n"
                f"    cue_stick_pos={np.round(cue_stick_pos,4).tolist() if cue_stick_pos else None} cue_stick_orient={np.round(cue_stick_orient,4).tolist() if cue_stick_orient is not None else None}"
            )
        if articulation_api.is_motion_complete():
            print(f"[diag] swing complete at step={step}")
            break
    else:
        print(f"[diag] swing EXHAUSTED {_SWING_MAX_STEPS} steps without completing")

    # 收尾：多跑幾步讓碰撞完全結算。
    for i in range(60):
        _step_counter["value"] = f"settle:{i}"
        simulation_app.update()
        linear_vel, _angular_vel = ball_rigid_prim.get_velocities()
        ball_speed = float(np.linalg.norm(np.asarray(linear_vel)[0]))
        if ball_speed > max_ball_speed:
            max_ball_speed = ball_speed
            max_ball_speed_step = -2

    print(f"[diag] contact_events_count={len(contacts)}")
    for i, (phase, c) in enumerate(contacts):
        print(f"[diag] contact[{i}] phase={phase}: {c.collider_path_a} <-> {c.collider_path_b}  impulse={c.impulse}")

    print(f"[diag] max_ball_speed={max_ball_speed:.4f} m/s at step={max_ball_speed_step}  required_tip_speed={required_tip_speed:.4f}")
    print(f"[diag] max_orient_err_deg={max_orient_err_deg:.2f}")
    print(f"[diag] min_tip_to_ball={min_tip_to_ball:.4f} at step={min_tip_to_ball_step}  ball_radius={_BALL_RADIUS}")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
