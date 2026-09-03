"""
scripts/test_isolated_swing_speed.py — 空場景（只有手臂+球桿，沒有球檯、沒有
球）驗證新揮桿速度控制方法論：Backswing/Contact Pose（IK，continuation 保持
同一關節分支）→ Jacobian 偽逆解 Contact Pose 所需關節角速度（檢查馬達最大
角速度限制）→ joint-space quintic polynomial 軌跡（兩端點給滿 position+
velocity+acceleration 邊界條件）→ velocity-mode 逐 tick 執行，量測桿尖實際
到達的速度是否達到 required_tip_speed。

背景：docs/issue-180-reachability-analysis.md 第十六節記錄的 move_swing()
（後擺/隨揮同姿態平移 waypoint + 逐 tick LP 最大化速度）實測只達目標球速
55%；事後用純 IK 可達性反推「能不能加長後擺距離」，兩次都被真實 Isaac Sim
推翻（一次是基座偏移讓差動 IK 不收斂，一次是單純加長後擺距離也讓桿身撞上
球檯庫邊）——純位置可達性分析回答不了「這個姿態能不能真的達到目標速度」
這個問題。這支腳本改用使用者提供文件裡的標準機器人學做法（Jacobian 偽逆
＋quintic 軌跡規劃），先在跟球檯/高架橋幾何完全脫鉤的乾淨場景驗證方法論
本身，不是先跳進耦合了球檯碰撞的完整場景除錯。

範圍界定（這次刻意不做的部分，見 plan）：
- 不建球檯、不建球、不處理高架橋 tilt/roll——純平面情境，direction 直接用
  CANONICAL_REST_JOINTS 姿態自身的桿尖指向，不透過
  cue_pose_calculator.compute_tilted_wrist_pose()。
- 不重新推導 m_eff（桿頭有效質量）——沿用既有
  swing_trajectory_calculator.compute_required_tip_speed()。
- 不做 manipulability 橢球式基座最佳化——基座位置固定用
  TableRobotManager 的既有公式。

跑法（需要 Isaac Sim headless）：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/test_isolated_swing_speed.py
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_EXT_DIR = os.path.join(_PROJECT_ROOT, "extension")
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")

for _p in (_EXT_DIR, _PROJECT_ROOT, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_CUE_BALL_SPEED = float(os.environ.get("ISO_CUE_BALL_SPEED", "1.995"))
_BACKSWING_DISTANCE_M = float(os.environ.get("ISO_BACKSWING_DISTANCE_M", "0.15"))
_DLS_LAMBDA = 0.05
_SETTLE_STEPS = 30
_EXTRA_STEPS_AFTER_T = 30


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


def _solve_quintic_coeffs(q0, q1, v1, T):
    """單一關節的 joint-space quintic：邊界條件 q(0)=q0,q̇(0)=0,q̈(0)=0,
    q(T)=q1,q̇(T)=v1,q̈(T)=0。回傳 (c3,c4,c5)（c0=q0,c1=0,c2=0 已知）。"""
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

    import wam7_kinematics as wk
    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from isaac_sim_impl_6_0.articulation_api_impl import ArticulationAPIImpl
    from core.models.barrett_wam_robot import BarrettWamRobot
    from core.models.table_robot_manager import TableRobotManager
    from core.services.base_placement_calculator import (
        CANONICAL_REST_JOINTS, CUE_STICK_GRIP_TO_TIP,
    )
    from core.services import swing_trajectory_calculator

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()

    base_path = "/World/IsolatedSwingSpeedTest"
    robot_prim_path = BarrettWamRobot.get_prim_path(base_path)
    end_effector_prim_path = BarrettWamRobot.get_end_effector_prim_path(base_path)
    articulation_api = ArticulationAPIImpl(robot_prim_path, end_effector_prim_path)
    robot_manager = TableRobotManager(
        (0.0, 0.0, 0.0), base_path, stage_api, articulation_api, BarrettWamRobot
    )
    robot = robot_manager.get_robot()

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()
    articulation_api.initialize()
    for _ in range(5):
        simulation_app.update()

    # ⚠️ base_position 沿用 TableRobotManager._ROBOT_OFFSET_FROM_TABLE_CENTER
    # 的既有公式（table_center=(0,0,0) 時機器人在 (1.5,0,0)），這次不做基座
    # 最佳化，固定用這個值即可——見 plan 的範圍界定。
    base_position = tuple(TableRobotManager._ROBOT_OFFSET_FROM_TABLE_CENTER)
    print(f"[iso] base_position={base_position}")

    # 用 CANONICAL_REST_JOINTS 本身的 FK 當「自然」contact pose——不透過
    # cue_pose_calculator，跟球檯/母球完全脫鉤，且保證這個目標本身一定
    # IK 可達（種子就是它自己），把方法論驗證跟幾何可達性驗證完全分開。
    canonical_joint_angles = [0.0, *CANONICAL_REST_JOINTS]
    natural_wrist, natural_orientation = wk.forward_kinematics(canonical_joint_angles, base_position)
    direction = _rotate_vector_by_quat(natural_orientation, np.array([0.0, 1.0, 0.0]))
    direction = direction / np.linalg.norm(direction)
    print(f"[iso] natural_wrist={natural_wrist.tolist()}  orientation={natural_orientation.tolist()}  direction={direction.tolist()}")

    # ---- 1. Contact Pose IK（種子＝CANONICAL_REST_JOINTS，理論上該目標
    # 本身就是這組關節角的 FK 結果，IK 應該幾乎不用迭代就收斂）----
    q_contact, contact_converged, contact_pos_err, contact_orient_err = wk.solve_ik(
        natural_wrist, natural_orientation, canonical_joint_angles, base_position=base_position
    )
    print(f"[iso] Contact Pose IK: converged={contact_converged} pos_err={contact_pos_err:.6f} orient_err={contact_orient_err:.6f}")
    print(f"[iso] q_contact={q_contact.tolist()}")

    # ---- 2. required_tip_speed（沿用既有動量公式，不重推 m_eff）----
    required_tip_speed = swing_trajectory_calculator.compute_required_tip_speed(_CUE_BALL_SPEED)
    print(f"[iso] required_tip_speed={required_tip_speed:.4f}  (cue_ball_speed={_CUE_BALL_SPEED})")

    # ---- 3. Jacobian 偽逆解 Contact Pose 所需關節角速度 ----
    J = wk._numerical_jacobian(q_contact, base_position)
    Jv = J[:3, :]
    Jang = J[3:, :]
    tip_offset = CUE_STICK_GRIP_TO_TIP * direction
    Jv_tip = Jv - _skew_matrix(tip_offset) @ Jang
    J_tip_full = np.vstack([Jv_tip, Jang])  # 6x7：前3列桿尖線速度，後3列角速度

    target_twist = np.concatenate([required_tip_speed * direction, np.zeros(3)])  # 姿態鎖死：角速度=0
    JJt = J_tip_full @ J_tip_full.T + (_DLS_LAMBDA ** 2) * np.eye(6)
    qdot_contact = J_tip_full.T @ np.linalg.solve(JJt, target_twist)
    print(f"[iso] qdot_contact={np.round(qdot_contact, 4).tolist()}")

    # ---- 4. Backswing Pose IK（種子＝q_contact，continuation 保持同一
    # 關節分支，近似 SEW 一致性）----
    backswing_position = natural_wrist - _BACKSWING_DISTANCE_M * direction
    q_backswing, backswing_converged, backswing_pos_err, backswing_orient_err = wk.solve_ik(
        backswing_position, natural_orientation, q_contact, base_position=base_position
    )
    print(f"[iso] Backswing Pose IK: converged={backswing_converged} pos_err={backswing_pos_err:.6f} orient_err={backswing_orient_err:.6f}")
    print(f"[iso] q_backswing={q_backswing.tolist()}")
    print(f"[iso] q_contact - q_backswing (關節位移) = {(q_contact - q_backswing).tolist()}")

    # ---- 5. 馬達最大角速度限制檢查（真實 articulation 讀值，不是猜測）----
    dof_max_velocities = np.asarray(articulation_api._dof_limits, dtype=float)
    print(f"[iso] dof_max_velocities={dof_max_velocities.tolist()}")
    exceed = np.abs(qdot_contact) > dof_max_velocities
    if exceed.any():
        print(f"[iso] ⚠️ qdot_contact 有關節超過馬達最大角速度：{exceed.tolist()}，超出量={np.round(np.abs(qdot_contact) - dof_max_velocities, 4).tolist()}")
    else:
        print("[iso] qdot_contact 全部關節都在馬達最大角速度限制內")

    # ---- 6. Quintic polynomial：time-scaling 找最小可行 T ----
    T = max(_BACKSWING_DISTANCE_M / max(required_tip_speed, 1e-6), 0.05)
    for _attempt in range(50):
        coeffs = [
            _solve_quintic_coeffs(q_backswing[i], q_contact[i], qdot_contact[i], T)
            for i in range(wk.NUM_JOINTS)
        ]
        peak_velocities = np.array([
            _peak_abs_velocity(c3, c4, c5, T) for c3, c4, c5 in coeffs
        ])
        if np.all(peak_velocities <= dof_max_velocities + 1e-9):
            break
        ratio = float(np.max(peak_velocities / np.maximum(dof_max_velocities, 1e-9)))
        T *= ratio * 1.05
    else:
        print("[iso] ⚠️ time-scaling 50 次仍未收斂，直接用目前的 T")
    print(f"[iso] quintic T={T:.4f}s  peak_velocities={np.round(peak_velocities, 4).tolist()}")

    # ---- 7. 瞬移到 q_backswing，穩定幾步 ----
    _BACKSWING_JOINTS = np.array([q_backswing])
    articulation_api._articulation.set_dof_positions(_BACKSWING_JOINTS)
    articulation_api._articulation.set_dof_velocities(np.zeros((1, wk.NUM_JOINTS)))
    articulation_api._articulation.switch_dof_control_mode("velocity")
    for _ in range(_SETTLE_STEPS):
        simulation_app.update()

    live_joints_before = np.asarray(articulation_api._articulation.get_dof_positions())[0]
    print(f"[iso] 瞬移＋穩定後實際關節角={live_joints_before.tolist()}")

    # ---- 8. velocity-mode 逐 tick 餵 q̇(t) ----
    physics_dt = 1.0 / 60.0
    num_steps = int(T / physics_dt) + _EXTRA_STEPS_AFTER_T
    max_qdot_error = 0.0
    for step in range(num_steps):
        t = min(step * physics_dt, T)
        qdot_ref = np.array([_quintic_velocity(c3, c4, c5, t) for c3, c4, c5 in coeffs])
        articulation_api._articulation.set_dof_velocity_targets(qdot_ref[None, :])
        simulation_app.update()

        live_qdot = np.asarray(articulation_api._articulation.get_dof_velocities())[0]
        qdot_error = float(np.max(np.abs(live_qdot - qdot_ref)))
        max_qdot_error = max(max_qdot_error, qdot_error)

        if step % 10 == 0 or step >= num_steps - 5:
            tip_position = np.array(articulation_api.get_end_effector_position())
            print(
                f"[iso] step={step} t={t:.3f}  qdot_ref_norm={np.linalg.norm(qdot_ref):.4f}  "
                f"qdot_error={qdot_error:.4f}  tip_position={np.round(tip_position, 4).tolist()}"
            )

    print(f"[iso] 全程最大關節角速度追蹤誤差={max_qdot_error:.4f} rad/s")

    # ---- 9. 量測最終桿尖實際速度 ----
    end_effector_rigid_prim = articulation_api._end_effector_rigid_prim
    wrist_linear, wrist_angular = end_effector_rigid_prim.get_velocities()
    wrist_linear = np.asarray(wrist_linear[0])
    wrist_angular = np.asarray(wrist_angular[0])
    current_orientation = articulation_api._get_end_effector_world_orientation()
    current_tip_direction = _rotate_vector_by_quat(current_orientation, np.array([0.0, 1.0, 0.0]))
    tip_offset_now = CUE_STICK_GRIP_TO_TIP * current_tip_direction
    tip_velocity = wrist_linear + np.cross(wrist_angular, tip_offset_now)
    tip_speed_along_direction = float(np.dot(tip_velocity, direction))
    tip_speed_total = float(np.linalg.norm(tip_velocity))

    print(f"[iso] 最終桿尖速度向量={tip_velocity.tolist()}")
    print(f"[iso] 沿 direction 分量={tip_speed_along_direction:.4f} m/s  總速度={tip_speed_total:.4f} m/s")
    print(f"[iso] required_tip_speed={required_tip_speed:.4f} m/s  達成率={100 * tip_speed_along_direction / required_tip_speed:.1f}%")

    # ---- 10. manipulability 記錄（這次不用它做決策，只記錄）----
    singular_values = np.linalg.svd(J_tip_full, compute_uv=False)
    manipulability = float(np.prod(singular_values))
    print(f"[iso] Contact Pose manipulability（singular values 乘積）={manipulability:.6f}  singular_values={singular_values.tolist()}")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    _headless = os.environ.get("ISO_HEADLESS", "1") != "0"
    simulation_app = SimulationApp({"headless": _headless})
    try:
        _run()
        if not _headless:
            print("[iso] 執行完畢，視窗保留中，關閉視窗以結束程式。")
            while simulation_app.is_running():
                simulation_app.update()
    finally:
        simulation_app.close()
