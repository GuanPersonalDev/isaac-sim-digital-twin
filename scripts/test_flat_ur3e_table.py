"""
scripts/test_flat_ur3e_table.py — 在真正有球檯（不是空場景）的情況下，直接
呼叫**正式生產程式碼**（`ur3e_placement_calculator.py` 的
`compute_flat_base_position_and_joint_targets()`／
`compute_bridge_base_position_and_joint_targets()` 等）驗證 UR3e 的姿態/基座
計算，在真實 GUI 場景（`billiard_digital_twin.py` 把 `_ROBOT_ARM_CLASS` 換成
`UR3eRobot` 之後）觀察到手臂「蜷縮陷入球檯邊緣、motion timeout+ERROR」之後，
用來直接定位問題出在姿態計算本身還是別的地方。

## 2026-09-02 重要修正

原本假設截圖的母球座標 (-0.036, -0.752) 是 flat 案例，第一次執行才發現
`compute_tilted_wrist_pose()` 對這個座標算出 `tilt_rad=0.1134`（6.50°）——
**其實是高架橋案例**，不是 flat（母球離庫邊夠近，需要抬高球桿閃避）。用這個
真實座標重算 `compute_bridge_base_position_and_joint_targets()` 得到
`base_position≈(1.4925, -2.3659, -0.6265)`，跟截圖 Property 面板實測的
`Translate=(1.49331, -2.36642, -0.61828)` 幾乎完全吻合——證明正式程式碼的
姿態計算本身**如預期運作**，套用的正是已經驗證過 96.1% 達成率（tilt=5.34°）
的那組姿態（nearest-neighbor 查表到 y=-0.635），只是這次目標 tilt（6.50°）
跟查到的姿態原本驗證的 tilt（5.34°）有約 1.16° 落差。這支腳本因此改成
**依 `tilt_rad` 動態分流**（flat／bridge 都支援，不是只測 flat），直接重現
GUI 看到的這個真實座標，而不是假設它是 flat。

## 跟 test_elevated_bridge_ur3e_table.py 的差異

高架橋案例當時是「先做姿態搜尋、再驗證」；這次姿態搜尋（Stage 1/2）已經做過
一輪並存進 `_FLAT_PLACEMENT`／`_PLACEMENT_LOOKUP_GRID`（見
`ur3e_placement_calculator.py` 常數說明），這支腳本**直接重用正式程式碼算出
的常數**（不重新跑搜尋網格），只做「把這組現有結果放進真實球檯場景驗證」
這一步。

`cue_ball` 用截圖裡實際看到的座標 (-0.036, -0.752)（Break shot demo 的母球
位置），不是憑空挑一個座標。

跑法（需要 Isaac Sim headless）：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" -u scripts/test_flat_ur3e_table.py

⚠️ 2026-09-02：發現這支腳本（跟 test_elevated_bridge_ur3e_table.py 同樣的
既有已知問題，見該檔案 `except BaseException` 註解）在丟出例外之後，
`simulation_app.close()` 有機率卡住不返回（Kit/PhysX 原生層級的既有問題，
不是這支腳本新引入的），必須用 `-u`（unbuffered stdout）讓輸出即時落地，
且執行時要準備好在看到最後一行想要的 log 之後手動判斷是否需要強制關閉
process，不能只等 process 自然結束。
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_UR3E_PATH = "Isaac/Robots/UniversalRobots/ur3e/ur3e.usd"
_TABLE_Z = 0.0
_BALL_RADIUS = 0.028575
_CUE_BALL = (-0.036, -0.752)
"""截圖 Billiard Debug 面板實測值（Break shot demo 開局母球位置），不是任意
挑選——用同一個座標才能重現 GUI 看到的行為。"""
_SHOT_ANGLE_DEG = 0.0
_CUE_BALL_SPEED = 1.995
_SWEEP_DEG = 30.0
_SETTLE_STEPS = 30
_EXTRA_STEPS_AFTER_T = 30


def _rotate_vector_by_quat(quat_wxyz, vec):
    import numpy as np
    w = quat_wxyz[0]
    q_xyz = quat_wxyz[1:]
    t = 2.0 * np.cross(q_xyz, vec)
    return vec + w * t + np.cross(q_xyz, t)


def _apply_velocity_with_gravity_compensation(articulation, qdot):
    import numpy as np
    articulation.set_dof_velocity_targets(qdot[None, :])
    gravity_compensation_forces = articulation.get_dof_gravity_compensation_forces()
    articulation.set_dof_efforts(gravity_compensation_forces)


def _solve_quintic_coeffs(q0, q1, v1, T):
    import numpy as np
    A = np.array([
        [T ** 3, T ** 4, T ** 5],
        [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
        [6 * T, 12 * T ** 2, 20 * T ** 3],
    ])
    b = np.array([q1 - q0, v1, 0.0])
    c3, c4, c5 = np.linalg.solve(A, b)
    return c3, c4, c5


def _quintic_velocity(c3, c4, c5, t):
    return 3 * c3 * t ** 2 + 4 * c4 * t ** 3 + 5 * c5 * t ** 4


def _peak_abs_velocity(c3, c4, c5, T, samples=200):
    import numpy as np
    ts = np.linspace(0.0, T, samples)
    vs = [abs(_quintic_velocity(c3, c4, c5, t)) for t in ts]
    return max(vs)


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf, Usd

    from isaacsim.core.experimental.prims import Articulation, RigidPrim

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.material_api_impl import MaterialAPIImpl
    from isaac_sim_impl_6_0.rigid_body_api_impl import RigidBodyAPIImpl
    from isaac_sim_impl_6_0.physics_api_impl import PhysicsAPIImpl

    from core.models.billiard_table import BilliardTable
    from core.models.contact_event import ContactEvent
    from core.services.base_placement_calculator import CUE_STICK_GRIP_TO_TIP
    from core.services.asset_utility import CUE_STICK_PATH
    from core.services import cue_pose_calculator, ur3e_placement_calculator

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()
    physics_api = PhysicsAPIImpl()

    # ---- 直接呼叫正式程式碼算目標腕部位置＋基座/關節目標（跟
    # table_orchestrator._execute_aim_ur3e()/_execute_strike_ur3e() 用同一組
    # 函式，不重新實作）----
    wrist, _orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
        _CUE_BALL, _SHOT_ANGLE_DEG, _TABLE_Z, _BALL_RADIUS,
    )
    if tilt_rad is None:
        raise RuntimeError(f"cue_ball={_CUE_BALL} 幾何無解（tilt_rad=None）")
    print(f"[flat] cue_ball={_CUE_BALL}  tilt_rad={tilt_rad:.6f} ({np.degrees(tilt_rad):.2f}°)  crossing={crossing}")
    print(f"[flat] wrist_position={list(wrist)}")

    # ⚠️ 2026-09-02：這裡跟 table_orchestrator._execute_aim_ur3e()/
    # _execute_strike_ur3e() 用完全一樣的 tilt_rad<=1e-6 分流判斷，不再假設
    # 一定是 flat——第一次執行才發現截圖那個母球座標其實是 tilt=6.50° 的
    # 高架橋案例，寫死跳過 bridge 分支會直接誤判成「無解」。
    target_direction = cue_pose_calculator.compute_tilted_direction(_SHOT_ANGLE_DEG, tilt_rad)
    if tilt_rad <= 1e-6:
        base_position, joint_targets = ur3e_placement_calculator.compute_flat_base_position_and_joint_targets(
            tuple(wrist), _SHOT_ANGLE_DEG
        )
        target_elbow_velocity = ur3e_placement_calculator.compute_flat_target_elbow_velocity(_CUE_BALL_SPEED)
    else:
        base_position, joint_targets = ur3e_placement_calculator.compute_bridge_base_position_and_joint_targets(
            tuple(wrist), tuple(target_direction), _CUE_BALL[1]
        )
        target_elbow_velocity = ur3e_placement_calculator.compute_bridge_target_elbow_velocity(
            _CUE_BALL_SPEED, _CUE_BALL[1]
        )
    print(f"[flat] target_direction={target_direction.tolist()}")
    elbow_dof_index = ur3e_placement_calculator.UR3E_ELBOW_DOF_INDEX
    print(f"[flat] base_position={base_position}")
    print(f"[flat] joint_targets(pan,shoulder_lift,elbow,wrist1,wrist2,wrist3)={joint_targets}")
    print(f"[flat] target_elbow_velocity={target_elbow_velocity:.4f} rad/s")

    if base_position[2] < -1.0 or base_position[2] > 0.5:
        raise RuntimeError(
            f"base_position Z={base_position[2]:.4f} 超出合理範圍（預期接近 WAM7 參考值 -0.6 附近），"
            "很可能會讓手臂陷進地板或懸空過高，中止執行"
        )

    # ---- 建 UR3e，直接瞬移到算好的 base_position/joint_targets（不做搜尋，
    # 只驗證這組現有常數）----
    robot_base_path = "/World/FlatUR3eTableTest"
    robot_prim_path = robot_base_path + "/Robot"
    stage_api.create_reference_prim(robot_prim_path, _UR3E_PATH)
    stage_api.set_prim_translate(robot_prim_path, *base_position)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    articulation = Articulation(paths=robot_prim_path)
    for _ in range(5):
        simulation_app.update()

    dof_max_velocities = np.asarray(articulation.get_dof_max_velocities())
    if hasattr(dof_max_velocities, "numpy"):
        dof_max_velocities = dof_max_velocities.numpy()
    dof_max_velocities = np.asarray(dof_max_velocities, dtype=float).reshape(-1)
    num_joints = dof_max_velocities.size
    elbow_limit = dof_max_velocities[elbow_dof_index]

    link_names = None
    for attr in ("link_names", "body_names"):
        if hasattr(articulation, attr):
            link_names = list(getattr(articulation, attr))
            break
    end_effector_link_name = "wrist_3_link"
    end_effector_rigid_prim = RigidPrim(paths=f"{robot_prim_path}/{end_effector_link_name}")

    joints_contact = np.asarray(joint_targets, dtype=float)

    # ---- 平移/擺姿態後驗證：桿尖世界座標應該等於 target_wrist_position ----
    articulation.set_dof_positions(joints_contact[None, :])
    articulation.set_dof_velocities(np.zeros((1, num_joints)))
    for _ in range(5):
        simulation_app.update()
    verify_tip_pos, verify_tip_orient = end_effector_rigid_prim.get_world_poses()
    verify_tip_orient = np.asarray(verify_tip_orient[0])
    verify_direction_guess = _rotate_vector_by_quat(verify_tip_orient, np.array([0.0, 0.0, 1.0]))
    verify_direction_guess = verify_direction_guess / np.linalg.norm(verify_direction_guess)
    verify_tip_offset = CUE_STICK_GRIP_TO_TIP * verify_direction_guess
    verify_world_tip = np.asarray(verify_tip_pos[0]) + verify_tip_offset
    target_wrist_position = np.asarray(wrist, dtype=float)
    position_error = float(np.linalg.norm(verify_world_tip - target_wrist_position))
    print(f"[flat] 實測桿尖位置={verify_world_tip.tolist()}  跟目標誤差={position_error:.5f} m（應該接近 0）")
    print(f"[flat] 實測 wrist_3_link 世界位置={np.asarray(verify_tip_pos[0]).tolist()}")

    # ---- 建球檯（真正的碰撞幾何）----
    table_base_path = "/World/FlatUR3eTableTestTable"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))
    ball_positions = {i: (5.0 + i * 0.2, 5.0) for i in range(10)}
    ball_positions[0] = _CUE_BALL
    table.get_table_ball_set().build(ball_positions)
    ball_prim_path = table.get_table_ball_set().get_ball_prim_paths()[0]
    ball_rigid_prim = RigidPrim(paths=ball_prim_path)

    # ---- 掛球桿 ----
    end_effector_prim_path = f"{robot_prim_path}/{end_effector_link_name}"
    cue_stick_prim_path = robot_base_path + "/CueStick"
    stage_api.create_reference_prim(cue_stick_prim_path, CUE_STICK_PATH)
    stage_api.align_prim_to_target(cue_stick_prim_path, end_effector_prim_path)
    stage_api.filter_collision_pair(cue_stick_prim_path, end_effector_prim_path)
    joint_path = cue_stick_prim_path + "/FixedJointToRobot"
    stage_api.create_fixed_joint(joint_path, cue_stick_prim_path, end_effector_prim_path)
    for _ in range(5):
        simulation_app.update()

    contacts: list[tuple[int, ContactEvent]] = []
    _step_counter = {"value": -1}
    physics_api.enable_contact_reporting(ball_prim_path)
    physics_api.enable_contact_reporting(cue_stick_prim_path)
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_prim_path)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
            physics_api.enable_contact_reporting(str(prim.GetPath()))
    physics_api.subscribe_contact_events(lambda e: contacts.append((_step_counter["value"], e)))

    # ---- Backswing Pose：只有肘關節角度往回轉 sweep_rad ----
    sweep_rad = np.radians(_SWEEP_DEG)
    joints_backswing = joints_contact.copy()
    joints_backswing[elbow_dof_index] -= sweep_rad
    print(f"[flat] sweep_rad={sweep_rad:.4f} ({_SWEEP_DEG}°)  joints_backswing={joints_backswing.tolist()}")

    q0 = float(joints_backswing[elbow_dof_index])
    q1 = float(joints_contact[elbow_dof_index])
    v1 = float(target_elbow_velocity)
    T = max(abs(q1 - q0) / max(v1, 1e-6), 0.05)
    peak_velocity = v1
    for _attempt in range(50):
        c3, c4, c5 = _solve_quintic_coeffs(q0, q1, v1, T)
        peak_velocity = _peak_abs_velocity(c3, c4, c5, T)
        if peak_velocity <= elbow_limit + 1e-9:
            break
        T *= (peak_velocity / elbow_limit) * 1.05
    print(f"[flat] quintic T={T:.4f}s  peak_elbow_velocity={peak_velocity:.4f} rad/s  elbow_limit={elbow_limit:.4f}")

    # ---- 瞬移到 Backswing Pose，穩定幾步（帶重力補償）----
    articulation.set_dof_positions(joints_backswing[None, :])
    articulation.set_dof_velocities(np.zeros((1, num_joints)))
    articulation.switch_dof_control_mode("velocity")
    _step_counter["value"] = 0
    for step in range(_SETTLE_STEPS):
        _step_counter["value"] = step
        _apply_velocity_with_gravity_compensation(articulation, np.zeros(num_joints))
        simulation_app.update()

    live_joints_before = np.asarray(articulation.get_dof_positions())[0]
    print(f"[flat] 瞬移＋穩定後實際關節角={live_joints_before.tolist()}")
    ball_vel_before, _ = ball_rigid_prim.get_velocities()
    print(f"[flat] settle 後母球速度={np.asarray(ball_vel_before[0]).tolist()}（應接近 0）")

    # ---- velocity-mode 逐 tick 餵 q̇(t)：只有肘關節非零 ----
    physics_dt = 1.0 / 60.0
    required_tip_direction = target_direction
    from core.services import swing_trajectory_calculator
    required_tip_speed = swing_trajectory_calculator.compute_required_tip_speed(_CUE_BALL_SPEED)

    num_steps = int(T / physics_dt) + _EXTRA_STEPS_AFTER_T
    history = []
    for step in range(num_steps):
        _step_counter["value"] = _SETTLE_STEPS + step
        t = min(step * physics_dt, T)
        qdot_ref = np.zeros(num_joints)
        qdot_ref[elbow_dof_index] = _quintic_velocity(c3, c4, c5, t)
        _apply_velocity_with_gravity_compensation(articulation, qdot_ref)
        simulation_app.update()

        live_joints_now = np.asarray(articulation.get_dof_positions())[0]
        wrist_linear, wrist_angular = end_effector_rigid_prim.get_velocities()
        wrist_linear = np.asarray(wrist_linear[0])
        wrist_angular = np.asarray(wrist_angular[0])
        _, current_orientation = end_effector_rigid_prim.get_world_poses()
        current_orientation = np.asarray(current_orientation[0])
        current_tip_direction = _rotate_vector_by_quat(current_orientation, np.array([0.0, 0.0, 1.0]))
        current_tip_direction = current_tip_direction / np.linalg.norm(current_tip_direction)
        tip_offset_now = CUE_STICK_GRIP_TO_TIP * current_tip_direction
        tip_velocity = wrist_linear + np.cross(wrist_angular, tip_offset_now)

        elbow_angle_error = float(live_joints_now[elbow_dof_index] - joints_contact[elbow_dof_index])
        history.append((abs(elbow_angle_error), elbow_angle_error, tip_velocity.copy()))

        if step % 10 == 0 or step >= num_steps - 5:
            print(f"[flat] step={step} t={t:.3f}  qdot_ref_elbow={qdot_ref[elbow_dof_index]:.4f}  elbow_angle_error={elbow_angle_error:.4f}")

    history.sort(key=lambda h: h[0])
    _, elbow_angle_error_at_contact, tip_velocity = history[0]
    tip_speed_along_direction = float(np.dot(tip_velocity, required_tip_direction))
    tip_speed_total = float(np.linalg.norm(tip_velocity))

    print("")
    print(f"[flat] 接觸瞬間肘關節角度誤差={elbow_angle_error_at_contact:.5f} rad")
    print(f"[flat] 接觸瞬間桿尖速度向量={tip_velocity.tolist()}")
    print(f"[flat] 沿方向分量={tip_speed_along_direction:.4f} m/s  總速度={tip_speed_total:.4f} m/s")
    print(f"[flat] required_tip_speed={required_tip_speed:.4f} m/s  達成率(沿方向)={100 * tip_speed_along_direction / required_tip_speed:.1f}%")

    ball_vel_after, _ = ball_rigid_prim.get_velocities()
    print(f"[flat] 揮桿結束後母球速度={np.asarray(ball_vel_after[0]).tolist()}")

    ball_contacts = [(s, e) for s, e in contacts if ball_prim_path in (e.actor_path_a, e.actor_path_b)]
    other_contacts = [(s, e) for s, e in contacts if ball_prim_path not in (e.actor_path_a, e.actor_path_b)]
    print("")
    print(f"[flat] 母球碰撞事件數={len(ball_contacts)}")
    for step_idx, e in ball_contacts:
        phase = "settle" if step_idx < _SETTLE_STEPS else "swing"
        print(f"    [{phase}] step={step_idx}  {e.actor_path_a} <-> {e.actor_path_b}  impulse={e.impulse:.4f}")
    print(f"[flat] 其他碰撞事件數（庫邊/桌面/手臂自撞）={len(other_contacts)}")
    for step_idx, e in other_contacts:
        phase = "settle" if step_idx < _SETTLE_STEPS else "swing"
        print(f"    [{phase}] step={step_idx}  {e.actor_path_a} <-> {e.actor_path_b}  impulse={e.impulse:.4f}")

    if other_contacts:
        print("[flat] [WARN] 揮桿過程中偵測到跟球以外的東西碰撞——需要檢查是不是撞到球檯/地板")
    else:
        print("[flat] [OK] 全程沒有偵測到跟球以外的東西碰撞")


if __name__ == "__main__":
    import traceback

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
