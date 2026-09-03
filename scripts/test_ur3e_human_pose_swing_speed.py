"""
scripts/test_ur3e_human_pose_swing_speed.py — 用
scripts/design_human_like_ur3e_pose.py Phase B 選出的最佳候選姿態
（純肘關節轉動、base/肩關節精確靜止），在空場景（只有手臂+球桿，沒有球檯、
沒有球）跑真正的 quintic 軌跡執行，量測桿尖實際到達的線速度是否達到
required_tip_speed。這是 Phase A/B「確定需求→設計姿態」之後的驗證步驟：
姿態算出來可行，不代表真的執行過去、追蹤誤差不會啃掉多少速度。

流程：
  1. 用 design_human_like_ur3e_pose.py 選出的最佳候選姿態當 Contact Pose
     （joints_contact，硬寫死在 _JOINTS_CONTACT，避免每次重跑一次網格搜尋）。
  2. 在 Contact Pose 現場量 Jacobian 的肘關節那一欄，算出這個姿態實際需要
     的 ω_elbow（不是抄 Phase B 印出來的數字，避免兩支腳本之間的姿態/版本
     兜不起來）。
  3. Backswing Pose：只把肘關節角度往回轉 _SWEEP_DEG（其餘 5 個關節角度
     跟 Contact Pose 完全相同），對應「純肘關節揮桿」的定義。
  4. 只對肘關節解一段 joint-space quintic（q0=backswing 肘角，q1=contact
     肘角，q̇(0)=0，q̇(T)=ω_elbow_needed，q̈ 兩端都是 0），其餘 5 個關節
     全程角速度指令=0（對應「base/肩關節精確靜止」的設計）。
  5. 瞬移到 Backswing Pose→velocity-mode 逐 tick 餵 q̇(t)→量測桿尖實際
     線速度（含 CUE_STICK_GRIP_TO_TIP 桿尖偏移的剛體速度合成），跟
     required_tip_speed 比較。

跑法（需要 Isaac Sim headless）：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/test_ur3e_human_pose_swing_speed.py
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
for _p in (_EXT_DIR, _PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_UR3E_PATH = "Isaac/Robots/UniversalRobots/ur3e/ur3e.usd"
_CUE_BALL_SPEED = float(os.environ.get("ISO_CUE_BALL_SPEED", "1.995"))
_SWEEP_DEG = float(os.environ.get("ISO_SWEEP_DEG", "30.0"))
_ELBOW_DOF_INDEX = 2
_SETTLE_STEPS = 30
_EXTRA_STEPS_AFTER_T = 30

# design_human_like_ur3e_pose.py 這次跑出的排行 #1（見對話紀錄）：
# joints=[shoulder_pan, shoulder_lift, elbow, wrist1, wrist2, wrist3]
_JOINTS_CONTACT = [0.0, -1.7, -0.9, -1.6, -1.5708, 0.0]


def _skew_matrix(v):
    import numpy as np
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def _rotate_vector_by_quat(quat_wxyz, vec):
    import numpy as np
    w = quat_wxyz[0]
    q_xyz = quat_wxyz[1:]
    t = 2.0 * np.cross(q_xyz, vec)
    return vec + w * t + np.cross(q_xyz, t)


def _apply_velocity_with_gravity_compensation(articulation, qdot):
    """跟 `extension/isaac_sim_impl_6_0/articulation_api_impl.py` 新增的
    `_apply_velocity_targets_with_gravity_compensation()` 同一套做法（那邊
    有完整背景說明：velocity 控制模式的 damping 不保證能抗重力，需要額外
    疊加重力補償力矩前饋）。這支腳本沒有透過 `ArticulationAPIImpl` 走，
    直接操作 `isaacsim.core.experimental.prims.Articulation`，所以在這裡
    複刻同一個技巧，不是重新設計一套：`set_dof_velocity_targets()` 下達
    速度目標，`get_dof_gravity_compensation_forces()` 讀出目前姿態需要的
    重力補償力矩，`set_dof_efforts()` 疊加上去（額外的 actuation force，
    跟 velocity drive 的 PD 力矩相加，不是取代）。`set_dof_efforts()` 不是
    常駐設定，必須每個 physics tick 重新呼叫——這個函式因此也要在每個
    tick 被呼叫，不能只呼叫一次。"""
    import numpy as np
    articulation.set_dof_velocity_targets(qdot[None, :])
    gravity_compensation_forces = articulation.get_dof_gravity_compensation_forces()
    articulation.set_dof_efforts(gravity_compensation_forces)


def _solve_quintic_coeffs(q0, q1, v1, T):
    """單一關節 joint-space quintic：q(0)=q0,q̇(0)=0,q̈(0)=0,q(T)=q1,
    q̇(T)=v1,q̈(T)=0。回傳 (c3,c4,c5)（c0=q0,c1=0,c2=0 已知）。

    ⚠️ v(T)=v1 這個邊界條件不會隨 T 縮放——如果 v1 本身就超過關節限制，
    加大 T 救不了（scripts/test_isolated_swing_speed.py 曾經因為沒有先檢查
    這點，time-scaling 迴圈發散到 T≈1e17s）。這裡呼叫端在進迴圈前就已經
    用 Phase A/B 驗證過 v1 在限制內，這個函式本身不重複防呆。"""
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
    from pxr import UsdPhysics, Sdf

    from isaacsim.core.experimental.prims import Articulation, RigidPrim

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from core.services.base_placement_calculator import CUE_STICK_GRIP_TO_TIP
    from core.services import swing_trajectory_calculator

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    prim_path = "/World/UR3eHumanPoseSwingTest"
    stage_api.create_reference_prim(prim_path, _UR3E_PATH)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    articulation = Articulation(paths=prim_path)
    for _ in range(5):
        simulation_app.update()

    # ⚠️ 2026-09-01 改版：第一版直接關掉這個 articulation 的重力
    # （`set_link_enabled_gravities([False])`）來繞開「UR3e 官方 USD 預設
    # velocity 模式阻尼不足以抗重力，settle/swing 過程中整支手臂自由落體
    # 式漂移」的問題（`switch_dof_control_mode("velocity")` 只會把
    # stiffness 歸零，damping 沿用 USD 內建值，不會自動幫忙抗重力）。這樣
    # 驗證的是「重力不存在時姿態/速度控制對不對」，跟正式場景（球檯有
    # 重力）不是同一回事，說服力不夠。改成真正解掉這個問題：重力保持開著
    # （不呼叫 set_link_enabled_gravities），改用重力補償力矩前饋——這是
    # `extension/isaac_sim_impl_6_0/articulation_api_impl.py` 新增的
    # `_apply_velocity_targets_with_gravity_compensation()` 同一套做法
    # （該檔案裡有完整背景說明），這支腳本沒有透過 ArticulationAPIImpl
    # 走（直接操作 `isaacsim.core.experimental.prims.Articulation`），所以
    # 這裡手動複刻同一個技巧：每個 physics tick 除了下達 `set_dof_velocity_
    # targets()`，額外呼叫 `get_dof_gravity_compensation_forces()` 讀出
    # 「維持目前姿態靜止所需要的重力補償力矩」，用 `set_dof_efforts()`
    # 疊加上去（`set_dof_efforts()` 是額外的 actuation force，跟 velocity
    # drive 本身的 PD 力矩在 PhysX 內部相加，不需要切到 "effort" 控制模式）。
    # 見下面 `_apply_velocity_with_gravity_compensation()` 這個小函式。

    dof_max_velocities = np.asarray(articulation.get_dof_max_velocities())
    if hasattr(dof_max_velocities, "numpy"):
        dof_max_velocities = dof_max_velocities.numpy()
    dof_max_velocities = np.asarray(dof_max_velocities, dtype=float).reshape(-1)
    num_joints = dof_max_velocities.size
    print(f"[human-pose] num_joints={num_joints}  dof_max_velocities(rad/s)={dof_max_velocities.tolist()}")

    link_names = None
    for attr in ("link_names", "body_names"):
        if hasattr(articulation, attr):
            link_names = list(getattr(articulation, attr))
            break
    end_effector_link_name = "wrist_3_link"

    jac_probe = np.asarray(articulation.get_jacobian_matrices().numpy())[0]
    idx = link_names.index(end_effector_link_name)
    if jac_probe.shape[0] == len(link_names) - 1:
        jac_link_index = idx - 1
    elif jac_probe.shape[0] == len(link_names):
        jac_link_index = idx
    else:
        raise RuntimeError(f"Jacobian link 數 {jac_probe.shape[0]} 與 link 名稱數 {len(link_names)} 對不上")

    end_effector_rigid_prim = RigidPrim(paths=f"{prim_path}/{end_effector_link_name}")

    def _settle(steps=3):
        for _ in range(steps):
            simulation_app.update()

    joints_contact = np.array(_JOINTS_CONTACT[:num_joints])

    # ---- 在 Contact Pose 現場量 Jacobian，算這個姿態真正需要的 ω_elbow ----
    articulation.set_dof_positions(joints_contact[None, :])
    articulation.set_dof_velocities(np.zeros((1, num_joints)))
    _settle()

    jac_all = np.asarray(articulation.get_jacobian_matrices().numpy())[0]
    J = jac_all[jac_link_index]

    tip_pos, tip_orient = end_effector_rigid_prim.get_world_poses()
    tip_orient = np.asarray(tip_orient[0])
    cue_local_axis = np.array([0.0, 0.0, 1.0])
    tip_direction_guess = _rotate_vector_by_quat(tip_orient, cue_local_axis)
    tip_direction_guess = tip_direction_guess / np.linalg.norm(tip_direction_guess)
    tip_offset = CUE_STICK_GRIP_TO_TIP * tip_direction_guess

    Jv = J[:3, :]
    Jang = J[3:, :]
    Jv_tip = Jv - _skew_matrix(tip_offset) @ Jang

    elbow_col_v = Jv_tip[:, _ELBOW_DOF_INDEX]
    speed_per_unit_omega = float(np.linalg.norm(elbow_col_v))
    direction = elbow_col_v / speed_per_unit_omega

    required_tip_speed = swing_trajectory_calculator.compute_required_tip_speed(_CUE_BALL_SPEED)
    omega_elbow_needed = required_tip_speed / speed_per_unit_omega
    elbow_limit = dof_max_velocities[_ELBOW_DOF_INDEX]
    print(f"[human-pose] joints_contact={joints_contact.tolist()}")
    print(f"[human-pose] required_tip_speed={required_tip_speed:.4f} m/s  speed_per_unit_omega={speed_per_unit_omega:.4f}  direction={direction.tolist()}")
    print(f"[human-pose] omega_elbow_needed={omega_elbow_needed:.4f} rad/s  elbow_limit={elbow_limit:.4f} rad/s")
    if omega_elbow_needed > elbow_limit:
        print("[human-pose] [WARN] omega_elbow_needed 超過肘關節限制，這組姿態在這個 cue_ball_speed 下不可行，中止")
        return

    # ---- Backswing Pose：只有肘關節角度往回轉 sweep_rad ----
    sweep_rad = np.radians(_SWEEP_DEG)
    joints_backswing = joints_contact.copy()
    joints_backswing[_ELBOW_DOF_INDEX] -= sweep_rad
    print(f"[human-pose] sweep_rad={sweep_rad:.4f} ({_SWEEP_DEG}°)  joints_backswing={joints_backswing.tolist()}")

    # ---- 只對肘關節解 quintic，其餘關節全程 q̇=0 ----
    q0 = float(joints_backswing[_ELBOW_DOF_INDEX])
    q1 = float(joints_contact[_ELBOW_DOF_INDEX])
    v1 = float(omega_elbow_needed)
    T = max(abs(q1 - q0) / max(v1, 1e-6), 0.05)
    for _attempt in range(50):
        c3, c4, c5 = _solve_quintic_coeffs(q0, q1, v1, T)
        peak_velocity = _peak_abs_velocity(c3, c4, c5, T)
        if peak_velocity <= elbow_limit + 1e-9:
            break
        T *= (peak_velocity / elbow_limit) * 1.05
    else:
        print("[human-pose] [WARN] time-scaling 50 次仍未收斂，直接用目前的 T")
    print(f"[human-pose] quintic T={T:.4f}s  peak_elbow_velocity={peak_velocity:.4f} rad/s")

    # ---- 瞬移到 Backswing Pose，穩定幾步 ----
    articulation.set_dof_positions(joints_backswing[None, :])
    articulation.set_dof_velocities(np.zeros((1, num_joints)))
    articulation.switch_dof_control_mode("velocity")
    # switch_dof_control_mode() 只改 stiffness/damping，不會動到既有的
    # target，這裡明確把速度目標歸零，避免殘留 target 干擾 settle 階段。
    # settle 階段（速度目標=0，靠重力補償撐住姿態）正是第一版重力漂移最
    # 明顯的地方（其他關節在這個階段就先漂移 0.4+ rad），所以這裡也要用
    # 帶重力補償的版本，不能只呼叫 set_dof_velocity_targets()。
    for _ in range(_SETTLE_STEPS):
        _apply_velocity_with_gravity_compensation(articulation, np.zeros(num_joints))
        simulation_app.update()

    live_joints_before = np.asarray(articulation.get_dof_positions())[0]
    print(f"[human-pose] 瞬移＋穩定後實際關節角={live_joints_before.tolist()}")

    # ---- velocity-mode 逐 tick 餵 q̇(t)：只有肘關節非零 ----
    #
    # ⚠️ 量測時機修正：t≥T 之後 quintic 的邊界條件讓 qdot_ref 停在 v1（非
    # 零——接觸瞬間本來就該還在動），如果像舊版一樣繼續餵 v1 一段固定的
    # _EXTRA_STEPS_AFTER_T（30 tick≈0.5s），肘關節會多轉 30×dt×v1≈0.5rad
    # ——遠遠衝過 joints_contact 目標角，量到的「桿尖速度方向」會是衝過頭
    # 之後的方向，不是真正接觸瞬間的方向（實測：沿方向分量 88.5% vs 總
    # 速度 99.3%，兩者對不上正是這個衝過頭造成的假象）。改成每個 tick 都
    # 記錄桿尖速度＋肘關節角度誤差，事後從整條紀錄裡挑「肘關節角度最接近
    # joints_contact」那一個 tick 當作真正的接觸瞬間量測點，不管迴圈實際
    # 跑了幾步。
    physics_dt = 1.0 / 60.0
    num_steps = int(T / physics_dt) + _EXTRA_STEPS_AFTER_T
    max_qdot_error = 0.0
    max_other_joint_drift = 0.0
    history = []
    for step in range(num_steps):
        t = min(step * physics_dt, T)
        qdot_ref = np.zeros(num_joints)
        qdot_ref[_ELBOW_DOF_INDEX] = _quintic_velocity(c3, c4, c5, t)
        _apply_velocity_with_gravity_compensation(articulation, qdot_ref)
        simulation_app.update()

        live_qdot = np.asarray(articulation.get_dof_velocities())[0]
        qdot_error = float(np.max(np.abs(live_qdot - qdot_ref)))
        max_qdot_error = max(max_qdot_error, qdot_error)

        live_joints_now = np.asarray(articulation.get_dof_positions())[0]
        other_drift = float(np.max(np.abs(
            np.delete(live_joints_now, _ELBOW_DOF_INDEX) - np.delete(joints_backswing, _ELBOW_DOF_INDEX)
        )))
        max_other_joint_drift = max(max_other_joint_drift, other_drift)

        wrist_linear, wrist_angular = end_effector_rigid_prim.get_velocities()
        wrist_linear = np.asarray(wrist_linear[0])
        wrist_angular = np.asarray(wrist_angular[0])
        _, current_orientation = end_effector_rigid_prim.get_world_poses()
        current_orientation = np.asarray(current_orientation[0])
        current_tip_direction = _rotate_vector_by_quat(current_orientation, cue_local_axis)
        current_tip_direction = current_tip_direction / np.linalg.norm(current_tip_direction)
        tip_offset_now = CUE_STICK_GRIP_TO_TIP * current_tip_direction
        tip_velocity = wrist_linear + np.cross(wrist_angular, tip_offset_now)

        elbow_angle_error = float(live_joints_now[_ELBOW_DOF_INDEX] - joints_contact[_ELBOW_DOF_INDEX])
        history.append((abs(elbow_angle_error), elbow_angle_error, tip_velocity.copy(), live_joints_now.copy()))

        if step % 10 == 0 or step >= num_steps - 5:
            print(f"[human-pose] step={step} t={t:.3f}  qdot_ref_elbow={qdot_ref[_ELBOW_DOF_INDEX]:.4f}  qdot_error={qdot_error:.4f}  other_joint_drift={other_drift:.5f}  elbow_angle_error={elbow_angle_error:.4f}")

    print(f"[human-pose] 全程最大關節角速度追蹤誤差={max_qdot_error:.4f} rad/s")
    print(f"[human-pose] 全程 base/肩/wrist 關節最大漂移={max_other_joint_drift:.5f} rad（應該接近 0，代表真的只有肘關節在動）")

    # ---- 從紀錄裡挑肘關節角度最接近 joints_contact 的那一個 tick 當「接觸瞬間」 ----
    history.sort(key=lambda h: h[0])
    _, elbow_angle_error_at_contact, tip_velocity, joints_at_contact = history[0]

    tip_speed_along_direction = float(np.dot(tip_velocity, direction))
    tip_speed_total = float(np.linalg.norm(tip_velocity))

    print(f"[human-pose] 接觸瞬間肘關節角度誤差={elbow_angle_error_at_contact:.5f} rad（挑全程最接近 joints_contact 的 tick）")
    print(f"[human-pose] 接觸瞬間桿尖速度向量={tip_velocity.tolist()}")
    print(f"[human-pose] 沿原始 direction 分量={tip_speed_along_direction:.4f} m/s  總速度={tip_speed_total:.4f} m/s")
    print(f"[human-pose] required_tip_speed={required_tip_speed:.4f} m/s  達成率(沿方向)={100 * tip_speed_along_direction / required_tip_speed:.1f}%  達成率(總速度)={100 * tip_speed_total / required_tip_speed:.1f}%")

    print(f"[human-pose] 接觸瞬間關節角={joints_at_contact.tolist()}  目標 joints_contact={joints_contact.tolist()}")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
