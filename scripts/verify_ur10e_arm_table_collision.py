"""
scripts/verify_ur10e_arm_table_collision.py - 診斷 joint-space 收尾階段
wrist_1_joint 卡在離目標 0.033rad 不動的問題（見 test_ur10e_table_flat.py
的除錯記錄）：懷疑是手臂／退到後擺位置的球桿實際頂到球檯或自身其他連桿，
但先前只對 CueStick／Ball_0 訂閱 contact reporting，沒有涵蓋手臂本身的
連桿，看不到證據。

這支腳本對 UR10e 手臂底下每一個帶 RigidBodyAPI 的連桿（base_link～
wrist_3_link）都呼叫 enable_contact_reporting()，並且在 RESET／AIM 各階段
之間切換一個「目前階段」標記，讓每筆 contact event 都能回報是哪個階段
發生的——藉此確認 wrist_1 卡住的當下，手臂到底跟什麼東西產生了非零衝量
的接觸。

⚠️ 另一個要交叉檢查的假設：production 路徑（ArticulationAPIImpl／
Ur10eRmpflowController／test_ur10e_table_flat.py）從頭到尾都沒有呼叫過
add_ground_plane()/add_obstacle()（只有 verify_ur10e_home_pose.py 這支
一次性驗證腳本呼叫過，見程式碼搜尋結果）——也就是說「RMPflow 已經知道
球檯在哪裡，理論上不會撞到」這個前提本身就不成立，RMPflow 對球檯完全沒
有障礙物知識。這支腳本刻意不註冊障礙物，如實反映目前 production 路徑的
真實狀態，用來確認：(a) 是否真的撞到球檯（支持「該補上障礙物註冊」）、
還是 (b) 撞到手臂自身其他連桿（self-collision，障礙物註冊也救不了，
因為 add_obstacle() 只能讓 RMPflow 避開外部物體，不會改變它自己的
逆向運動學解）。

另外，_step_joint_space_finish()（目前卡住的那一段）是直接下 PD
position-mode 關節指令，完全繞過 _rmp_flow.compute_joint_targets()——
即使有註冊障礙物，這一段的避障邏輯也不會生效，這點也在腳本開頭印出來
提醒。

跑法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/verify_ur10e_arm_table_collision.py

可選：DEBUG_UR10E_FINISH_IK=1 一併打開，能同時看到 joint_space_finish
逐 tick 的關節誤差趨勢跟這裡的碰撞事件時間點對照。
"""

import os
import sys
import traceback

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_CUE_BALL = (0.0, 0.5)
_SHOT_ANGLE_DEG = 0.0
_PHYSICS_DT = 1.0 / 60.0
_MAX_STEPS_PER_ACTION = 4000


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf, Usd

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl
    from isaac_sim_impl_6_0.physics_api_impl import PhysicsAPIImpl
    from isaac_sim_impl_6_0.articulation_api_impl import ArticulationAPIImpl

    from core.models.action import Action
    from core.models.billiard_table import BilliardTable
    from core.models.table_robot_manager import TableRobotManager
    from core.models.ur10e_robot import UR10eRobot
    from core.services import cue_pose_calculator
    from core.services.ur10e_swing_strategy import Ur10eSwingStrategy
    from isaacsim.core.experimental.prims import RigidPrim

    print(
        "[diag] 提醒：production 路徑目前完全沒有呼叫 add_ground_plane()/"
        "add_obstacle()（只有 verify_ur10e_home_pose.py 這支一次性腳本呼叫過），"
        "RMPflow 對球檯沒有任何障礙物知識；_step_joint_space_finish() 更是"
        "直接下 PD position-mode 關節指令，完全繞過 RMPflow 的避障計算。"
        "這支腳本刻意不註冊障礙物，如實反映 production 路徑的真實狀態。"
    )
    sys.stdout.flush()

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()
    physics_api = PhysicsAPIImpl()

    table_base_path = "/World/TestUr10eArmTableCollision"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))
    table_ball_set = table.get_table_ball_set()
    table_z = table_ball_set.get_table_z()
    ball_radius = table_ball_set.DEFAULT_BALL_RADIUS

    wrist_position, wrist_orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
        _CUE_BALL, _SHOT_ANGLE_DEG, table_z, ball_radius,
    )
    if tilt_rad is None:
        raise RuntimeError(f"cue_ball={_CUE_BALL} 幾何無解")

    print("[diag] 建立 ArticulationAPIImpl ...")
    sys.stdout.flush()
    robot_prim_path = UR10eRobot.get_prim_path(table_base_path)
    end_effector_prim_path = UR10eRobot.get_end_effector_prim_path(table_base_path)
    articulation_api = ArticulationAPIImpl(robot_prim_path, end_effector_prim_path)

    print("[diag] 建立 TableRobotManager ...")
    sys.stdout.flush()
    robot_manager = TableRobotManager(
        table.get_table_center(), table_base_path, stage_api, articulation_api, UR10eRobot,
    )
    robot_arm = robot_manager.get_robot()
    cue_stick_prim_path = robot_manager.get_cue_stick_prim_path()
    ball_prim_path = table_ball_set.get_ball_prim_paths()[0]

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    table_ball_set.place_ball(0, _CUE_BALL[0], _CUE_BALL[1])
    for _ in range(5):
        simulation_app.update()

    print("[diag] articulation_api.initialize() ...")
    sys.stdout.flush()
    articulation_api.initialize()
    for _ in range(5):
        simulation_app.update()

    # api-lookup agent 查證結果（讀原始碼確認 update_default_gains 機制沒問題
    # 後）最可疑的剩餘假設：wrist_1_joint 的 PhysX drive type 是不是
    # "none"（USD 沒有 authored PhysicsDriveAPI）——這種情況下 stiffness/
    # damping/position target 全部寫進 PhysX 不會讀的 buffer，唯一真正
    # 產生力矩的只剩重力補償力，完全符合「幾乎不動、只有微幅漂移」的症狀。
    # 直接查全部 7 個 DOF 的 drive type 跟初始 max_effort，一次排除或坐實
    # 這個假設。
    _dof_names_for_diag = list(articulation_api._articulation.dof_names)
    _drive_types = articulation_api._articulation.get_dof_drive_types()
    _max_efforts_initial = articulation_api._articulation.get_dof_max_efforts()
    _max_efforts_initial = np.asarray(
        _max_efforts_initial.numpy() if hasattr(_max_efforts_initial, "numpy") else _max_efforts_initial,
        dtype=float,
    )
    print("[diag] 初始 drive_type / max_effort（逐 DOF）：")
    for i, name in enumerate(_dof_names_for_diag):
        print(f"[diag]   {name}: drive_type={_drive_types[0][i]}  max_effort={_max_efforts_initial[0][i]}")
    sys.stdout.flush()

    # 全連桿 contact reporting：把手臂底下每一個帶 RigidBodyAPI 的 prim
    # （UR10e 官方 USD 的 base_link/shoulder_link/upper_arm_link/
    # forearm_link/wrist_1_link/wrist_2_link/wrist_3_link）都啟用，而不是
    # 只有 CueStick/Ball_0——PhysX 的 contact report 只需要碰撞雙方其中一方
    # 有 ContactReportAPI 就會回報，不需要對球檯本身（沒有 RigidBodyAPI 的
    # 靜態 collider）也啟用，也不應該——physics_api_impl.enable_contact_
    # reporting() 對沒有 RigidBodyAPI 的 prim 會套用 PhysxTriggerAPI（球袋
    # 感測器慣例），套到球檯桌面/庫邊會讓它變成不會真的碰撞回應的 trigger，
    # 是危險的誤用。
    robot_root_prim = stage.GetPrimAtPath(robot_prim_path)
    link_prim_paths = []
    joint_limit_info = []
    for prim in Usd.PrimRange(robot_root_prim):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            link_prim_paths.append(str(prim.GetPath()))
        if prim.IsA(UsdPhysics.RevoluteJoint):
            revolute = UsdPhysics.RevoluteJoint(prim)
            lower = revolute.GetLowerLimitAttr().Get()
            upper = revolute.GetUpperLimitAttr().Get()
            joint_limit_info.append((str(prim.GetPath()), lower, upper))
    print(f"[diag] 手臂底下找到 {len(link_prim_paths)} 個 RigidBody 連桿，全部開啟 contact reporting：")
    for p in link_prim_paths:
        print(f"[diag]   {p}")
        physics_api.enable_contact_reporting(p)
    sys.stdout.flush()

    print(f"[diag] 手臂底下找到 {len(joint_limit_info)} 個 RevoluteJoint，關節極限（度）：")
    for path, lower, upper in joint_limit_info:
        print(f"[diag]   {path}  lower={lower}  upper={upper}")
    sys.stdout.flush()

    contacts = []
    phase = {"name": "SETUP"}

    def _on_contact(event):
        contacts.append((phase["name"], event))

    physics_api.enable_contact_reporting(cue_stick_prim_path)
    physics_api.enable_contact_reporting(ball_prim_path)
    physics_api.subscribe_contact_events(_on_contact)

    strategy = Ur10eSwingStrategy(robot_arm, articulation_api)
    action = Action(
        cue_ball_speed=1.995,
        shot_angle=_SHOT_ANGLE_DEG,
        position_offset=[0.0, 0.0],
        cue_ball_placement=list(_CUE_BALL),
        should_execute_action=True,
    )

    initial_base_offset = TableRobotManager._ROBOT_OFFSET_FROM_TABLE_CENTER
    initial_base_position = [
        table.get_table_center()[0] + initial_base_offset[0],
        table.get_table_center()[1] + initial_base_offset[1],
        table.get_table_center()[2] + initial_base_offset[2],
    ]
    articulation_api.set_robot_base_pose(initial_base_position, [1.0, 0.0, 0.0, 0.0])

    def _run_until_complete(label: str) -> int:
        phase["name"] = label
        step = 0
        while not articulation_api.is_motion_complete() and step < _MAX_STEPS_PER_ACTION:
            simulation_app.update()
            step += 1
        print(f"[diag] {label} 完成，steps={step} did_last_motion_timeout={articulation_api.did_last_motion_timeout()}")
        return step

    print("[diag] 呼叫 articulation_api.move_to_home()（RESET）...")
    articulation_api.move_to_home()
    _run_until_complete("RESET(HOME)")

    # joint_space_finish 逐 tick 追蹤：目前的 contact report 只回報
    # CONTACT_FOUND（剛接觸的瞬間，見 physics_api_impl.py），如果桿尖跟
    # 母球整段時間持續重疊（沒有真正分開過），後續每個 tick 的「持續接觸」
    # 事件會被這個 filter 全部吃掉，只留下第一筆——可能誤導成「只撞了一下」
    # 但實際上是全程卡住。改成直接逐 tick 量測母球速度／桿尖-母球距離，
    # 不依賴 contact report 的事件計數，交叉驗證是不是這個情況。
    ball_rigid_prim = RigidPrim(paths=ball_prim_path)
    cue_stick_rigid_prim = RigidPrim(paths=cue_stick_prim_path)

    def _compute_bbox_tip_local_offset(prim_path: str) -> np.ndarray:
        from pxr import UsdGeom
        prim = stage.GetPrimAtPath(prim_path)
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        )
        local_range = bbox_cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        min_pt = np.array(local_range.GetMin())
        max_pt = np.array(local_range.GetMax())
        if np.any(min_pt > max_pt):
            return np.zeros(3)
        axis_index = int(np.argmax(max_pt - min_pt))
        tip_local = np.zeros(3)
        tip_local[axis_index] = (
            max_pt[axis_index] if abs(max_pt[axis_index]) > abs(min_pt[axis_index]) else min_pt[axis_index]
        )
        return tip_local

    def _rotate_vector_by_quat(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
        w = quat_wxyz[0]
        q_xyz = quat_wxyz[1:]
        t = 2.0 * np.cross(q_xyz, vec)
        return vec + w * t + np.cross(q_xyz, t)

    cue_tip_local_offset = _compute_bbox_tip_local_offset(cue_stick_prim_path)

    def _tip_ball_distance_and_ball_velocity():
        cue_position, cue_orientation = cue_stick_rigid_prim.get_world_poses()
        cue_position = np.asarray(cue_position[0], dtype=float)
        cue_orientation = np.asarray(cue_orientation[0], dtype=float)
        tip_world = cue_position + _rotate_vector_by_quat(cue_orientation, cue_tip_local_offset)
        ball_position, _ = ball_rigid_prim.get_world_poses()
        ball_position = np.asarray(ball_position[0], dtype=float)
        distance = float(np.linalg.norm(tip_world - ball_position))
        ball_velocity, _ = ball_rigid_prim.get_velocities()
        ball_speed = float(np.linalg.norm(np.asarray(ball_velocity[0], dtype=float)))
        return distance, ball_speed

    rmp_ctrl = articulation_api._ur10e_rmpflow_controller

    print("[diag] 呼叫 Ur10eSwingStrategy.execute_aim() ...")
    strategy.execute_aim(action, tuple(_CUE_BALL), table_z, ball_radius)
    phase["name"] = "AIM"
    step = 0
    joint_finish_tick = 0
    while not articulation_api.is_motion_complete() and step < _MAX_STEPS_PER_ACTION:
        simulation_app.update()
        step += 1
        if rmp_ctrl._joint_finish_active:
            joint_finish_tick += 1
            distance, ball_speed = _tip_ball_distance_and_ball_velocity()
            if joint_finish_tick <= 5 or joint_finish_tick % 20 == 0:
                print(
                    f"[diag] joint_finish tick={joint_finish_tick} "
                    f"桿尖-母球距離={distance:.5f}m（球半徑={ball_radius:.5f}m） "
                    f"母球速度={ball_speed:.5f}m/s"
                )
    print(f"[diag] AIM 完成，steps={step} did_last_motion_timeout={articulation_api.did_last_motion_timeout()}")

    current_full_positions = np.asarray(articulation_api._articulation.get_dof_positions())[0]
    current_active_positions = current_full_positions[rmp_ctrl._active_dof_indices]
    if rmp_ctrl._joint_finish_target is not None:
        joint_gap = current_active_positions - rmp_ctrl._joint_finish_target
        print(f"[diag] joint_space_finish 目標={rmp_ctrl._joint_finish_target.tolist()}")
        print(f"[diag] joint_space_finish 實際={current_active_positions.tolist()}")
        print(f"[diag] joint_space_finish 誤差={joint_gap.tolist()} max_abs={np.max(np.abs(joint_gap)):.6f} rad")
        print("[diag] 逐關節比對目標值 vs 該關節的 RevoluteJoint 極限（檢查是不是撞到關節極限被 PhysX clamp 住）：")
        joint_limit_by_name = {path.rsplit("/", 1)[-1]: (lower, upper) for path, lower, upper in joint_limit_info}
        for name, cur, tgt in zip(rmp_ctrl._active_joint_names, current_active_positions, rmp_ctrl._joint_finish_target):
            limit = joint_limit_by_name.get(name)
            limit_str = f"lower={np.radians(limit[0]):.5f}rad upper={np.radians(limit[1]):.5f}rad" if limit else "（找不到極限資料）"
            print(f"[diag]   {name}: 目前={cur:.5f}rad 目標={tgt:.5f}rad {limit_str}")

    print(f"[diag] 全部 contact events 數量={len(contacts)}")
    nonzero_contacts = [(p, e) for p, e in contacts if e.impulse > 0.0]
    print(f"[diag] impulse>0 的 contact events 數量={len(nonzero_contacts)}")
    for phase_name, e in nonzero_contacts:
        print(
            f"[diag]   phase={phase_name}  a={e.actor_path_a}  b={e.actor_path_b}  "
            f"collider_a={e.collider_path_a}  collider_b={e.collider_path_b}  impulse={e.impulse}"
        )

    if not nonzero_contacts:
        print(
            "[diag] 沒有偵測到任何非零衝量的碰撞事件——wrist_1 卡住不是"
            "碰撞造成的，需要往其他方向排查（例如關節限制 clamp、"
            "articulation drive 內部限制）。"
        )


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    except Exception:
        print("[diag] _run() 拋出例外：")
        traceback.print_exc()
        sys.stdout.flush()
        raise
    finally:
        simulation_app.close()
