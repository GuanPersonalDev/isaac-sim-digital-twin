"""
scripts/probe_first_case_residual_error.py — 查「序列裡第一個案例殘留誤差」
的根因：(-0.25, -0.1) 這個 flat（tilt=0）案例，在完全沒有任何差動 IK 呼叫
墊底、initialize() 後的第一個動作就是它時，穩定在 ~23.2mm 誤差，跟同一組
CANONICAL_REST_JOINTS、只是 base_yaw 不同的其他案例（0.5mm、14.2mm）不一致。

假設清單：
  H1 快照過期／假陽性收斂：第一次 is_motion_complete() 讀到還沒更新的
     tensor 快照，提早誤判為已到位，實際上手臂還在路上。
  H2 真的卡在穩態但位置錯：PhysX position drive 真的把手臂帶到某個穩定但
     偏離目標的關節配置（例如關節限位卡住、或跟 H1 不同，是 target 真的沒
     設對）。
  H3 跟「有沒有東西墊過場」有關：不是「順序第一個」本身的問題，而是「這次
     joint-space 位移量特別大」（從初始展開姿態跳到 CANONICAL_REST_JOINTS）
     所需時間比其他案例長，1000 步不夠。

做法：對 (-0.25, -0.1) 這個案例，initialize() 之後立刻執行，但這次逐步印出
每隔 20 步的 position_error、目前關節角度，一路印到 1000 步或收斂為止，
藉此判斷是哪種情況；然後緊接著對「同一個目標」再下一次一模一樣的
move_to_joint_position() 呼叫，看第二次是否瞬間收斂（如果是，代表跟
「這是不是第一次呼叫」本身有關，而不是目標配置或關節限位的問題）。

用法：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/probe_first_case_residual_error.py
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
_CUE_BALL = (-0.25, -0.1)


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
    from core.services.base_placement_calculator import CANONICAL_REST_JOINTS, compute_base_pose

    from scripts.scan_elevated_bridge_approach import compute_tilted_wrist_pose

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()

    table_base_path = "/World/FirstCaseProbeTable"
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

    physics_api = PhysicsAPIImpl()
    cue_stick_prim_path = robot_manager.get_cue_stick_prim_path()
    physics_api.enable_contact_reporting(cue_stick_prim_path)
    from pxr import Usd, UsdPhysics as _UsdPhysics
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    robot_link_paths = []
    for prim in Usd.PrimRange(robot_prim):
        if prim.HasAPI(_UsdPhysics.RigidBodyAPI):
            robot_link_paths.append(prim.GetPath().pathString)
            physics_api.enable_contact_reporting(prim.GetPath().pathString)
    print(f"enabled contact reporting on {len(robot_link_paths)} robot links")
    contacts: list[ContactEvent] = []
    physics_api.subscribe_contact_events(lambda e: contacts.append(e))

    print(f"dof_names={articulation_api._articulation.dof_names}")
    print(f"joint positions right after initialize(): "
          f"{np.asarray(articulation_api._articulation.get_dof_positions())[0].tolist()}")
    max_efforts = np.asarray(articulation_api._articulation.get_dof_max_efforts())
    print(f"dof max_efforts (URDF <limit effort=...>)={max_efforts.tolist()}")

    angle_deg = _shot_angle_deg(_CUE_BALL, _TARGET_BALL)
    base_position, base_yaw_rad = compute_base_pose(_CUE_BALL[0], _CUE_BALL[1], angle_deg, table_z=table_z)
    print(f"cue_ball={_CUE_BALL}  angle_deg={angle_deg:.2f}  base_position={base_position}  base_yaw_rad={base_yaw_rad:.4f}")

    robot.reposition(base_position)
    for _ in range(30):
        simulation_app.update()

    wrist0, orientation0, tilt0, crossing0 = compute_tilted_wrist_pose(
        _CUE_BALL, angle_deg, table_z, ball_radius, roll_rad=0.0
    )
    print(f"tilt0={tilt0}  wrist0={wrist0.tolist()}")

    joint_targets = [base_yaw_rad, *CANONICAL_REST_JOINTS]
    print(f"joint_targets={joint_targets}")

    def _run_to_target(label, target_wrist, max_steps=1000, log_every=20):
        articulation_api.move_to_joint_position(joint_targets, target_wrist.tolist())
        settled_step = None
        for step in range(max_steps):
            simulation_app.update()
            current = np.array(articulation_api.get_end_effector_position())
            err = float(np.linalg.norm(current - target_wrist))
            complete = articulation_api.is_motion_complete()
            if step % log_every == 0 or complete:
                actual_joints = np.asarray(articulation_api._articulation.get_dof_positions())[0]
                print(f"  [{label}] step={step:4d} err={err:.5f} m complete={complete} "
                      f"joints={np.round(actual_joints, 4).tolist()}")
            if complete and settled_step is None:
                settled_step = step
                break
        final = np.array(articulation_api.get_end_effector_position())
        final_err = float(np.linalg.norm(final - target_wrist))
        print(f"  [{label}] FINAL settled_step={settled_step} final_err={final_err:.5f} m")
        return settled_step, final_err

    print("\n=== 第一次呼叫（initialize() 後的第一個動作，無任何墊場）===")
    _run_to_target("first_call", wrist0)

    # 額外多跑 500 步，確認"settled"之後的殘留誤差是不是真的穩定值（H2），
    # 還是其實還在緩慢收斂只是還沒到 tolerance 之內（H3）。
    print("\n=== 多跑 500 步觀察是否仍在緩慢收斂 ===")
    for step in range(500):
        simulation_app.update()
        if step % 50 == 0:
            current = np.array(articulation_api.get_end_effector_position())
            err = float(np.linalg.norm(current - wrist0))
            print(f"  [extra] step={step:4d} err={err:.5f} m")

    print("\n=== 第二次呼叫（同一個目標，立刻再下一次 move_to_joint_position）===")
    _run_to_target("second_call", wrist0)

    print("\n=== 第三次呼叫（同一個目標，再下一次）===")
    contacts.clear()
    _run_to_target("third_call", wrist0)
    for _ in range(30):
        simulation_app.update()
    partners = sorted({c.collider_path_a for c in contacts} | {c.collider_path_b for c in contacts})
    print(f"卡住狀態下累積的碰撞事件數={len(contacts)}  partners={partners}")
    for c in contacts[-20:]:
        print(f"  contact: a={c.collider_path_a} b={c.collider_path_b}")

    # 驗證假設：wrist_yaw（index 4）URDF effort=10 N·m 的力矩上限，是否就是
    # case A 卡在 +0.096 rad 回不去 0 的原因——把上限大幅拉高後再下同一個
    # 目標一次，如果能瞬間收斂到 <5mm，就證實是力矩飽和，不是別的原因
    # （例如關節限位、自我碰撞）。
    print("\n=== 驗證：拉高 wrist_yaw (dof index 4) 的 max_effort 後重試 ===")
    articulation_api._articulation.set_dof_max_efforts([1000.0], dof_indices=[4])
    readback_max_efforts = np.asarray(articulation_api._articulation.get_dof_max_efforts())
    print(f"after set_dof_max_efforts: {readback_max_efforts.tolist()}")
    _run_to_target("after_raise_max_effort", wrist0, max_steps=300, log_every=10)

    # max_effort 不是瓶頸——改驗證 PD 彈簧本身的 stiffness 夠不夠。目前這組
    # 值是「position 模式第一次呼叫時」快取下來的 USD authored 值，如果太
    # 低，spring force = stiffness * error 在很小的 error 就跟重力力矩打平，
    # 產生穩定但偏離目標的平衡點——跟 max_effort 完全是兩回事。
    print("\n=== 驗證：讀出目前 stiffness/damping，並拉高 wrist_yaw 的 stiffness 後重試 ===")
    stiffnesses, dampings = articulation_api._articulation.get_dof_gains()
    print(f"current stiffnesses={np.asarray(stiffnesses).tolist()}")
    print(f"current dampings={np.asarray(dampings).tolist()}")
    articulation_api._articulation.set_dof_gains(
        stiffnesses=[float(np.asarray(stiffnesses)[0][4]) * 50.0], dof_indices=[4], update_default_gains=False
    )
    new_stiffnesses, _ = articulation_api._articulation.get_dof_gains()
    print(f"after boosting stiffness x50: {np.asarray(new_stiffnesses).tolist()}")
    _run_to_target("after_raise_stiffness", wrist0, max_steps=300, log_every=10)

    # stiffness x50 完全無感——代表這個關節根本不是被彈簧公式卡在那個值，
    # 懷疑是 PhysX solver iteration count 不夠，7 自由度鏈 + 球桿 FixedJoint
    # 耦合系統在預設迭代次數下數值上收斂到一個穩定但錯誤的解、且對 gain
    # 不敏感（典型的「solver 迭代太少」假影特徵）。USD 裡沒有明確 author
    # solverPositionIterationCount，代表用的是 PhysX/Isaac Sim 預設值
    # （通常偏低，例如 4）。直接拉高後重試同一個目標。
    print("\n=== 驗證：拉高 solver iteration counts 後重試 ===")
    position_iters, velocity_iters = articulation_api._articulation.get_solver_iteration_counts()
    print(f"current solver_iteration_counts: position={np.asarray(position_iters).tolist()} "
          f"velocity={np.asarray(velocity_iters).tolist()}")
    articulation_api._articulation.set_solver_iteration_counts(position_iteration_count=255, velocity_iteration_count=255)
    new_position_iters, new_velocity_iters = articulation_api._articulation.get_solver_iteration_counts()
    print(f"after boosting: position={np.asarray(new_position_iters).tolist()} "
          f"velocity={np.asarray(new_velocity_iters).tolist()}")
    _run_to_target("after_raise_solver_iters", wrist0, max_steps=300, log_every=10)

    # 對照組：(0.0, 0.4) 之前回報收斂到 0.5mm，同一個 session、同一份快取
    # 的 default gains，只是 base_yaw 不同、CANONICAL_REST_JOINTS 完全相同。
    # 如果 joint4 的重力垂陷量是跟 base_yaw 無關的固定物理量，這裡應該量到
    # 同樣的 ~0.096 rad 偏移；如果量到的偏移明顯不同，代表垂陷量本身跟
    # base_yaw / 起始關節配置有關，不是單純的「固定重力力矩」。
    print("\n\n########## 對照案例 (0.0, 0.4) ##########")
    cue_ball_b = (0.0, 0.4)
    angle_deg_b = _shot_angle_deg(cue_ball_b, _TARGET_BALL)
    base_position_b, base_yaw_rad_b = compute_base_pose(cue_ball_b[0], cue_ball_b[1], angle_deg_b, table_z=table_z)
    print(f"cue_ball={cue_ball_b}  angle_deg={angle_deg_b:.2f}  base_position={base_position_b}  base_yaw_rad={base_yaw_rad_b:.4f}")

    robot.reposition(base_position_b)
    for _ in range(30):
        simulation_app.update()

    wrist0_b, orientation0_b, tilt0_b, crossing0_b = compute_tilted_wrist_pose(
        cue_ball_b, angle_deg_b, table_z, ball_radius, roll_rad=0.0
    )
    print(f"tilt0_b={tilt0_b}  wrist0_b={wrist0_b.tolist()}")
    joint_targets_b = [base_yaw_rad_b, *CANONICAL_REST_JOINTS]
    articulation_api.move_to_joint_position(joint_targets_b, wrist0_b.tolist())
    # 不要在第一次 is_motion_complete()==True 就停止觀察——case A 已經證明
    # 「暫時滑過 5mm 容許帶」跟「真的到達穩態」是兩回事，這裡故意跑滿 1000
    # 步，即使中途曾經 complete=True 也繼續看 joint4 會不會之後又飄走。
    for step in range(1000):
        simulation_app.update()
        if step % 20 == 0 or step == 999:
            current = np.array(articulation_api.get_end_effector_position())
            err = float(np.linalg.norm(current - wrist0_b))
            actual_joints = np.asarray(articulation_api._articulation.get_dof_positions())[0]
            print(f"  [case_b] step={step:4d} err={err:.5f} m complete={articulation_api.is_motion_complete()} "
                  f"joints={np.round(actual_joints, 4).tolist()}")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
