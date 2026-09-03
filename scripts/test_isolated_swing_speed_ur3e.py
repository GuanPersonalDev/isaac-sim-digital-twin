"""
scripts/test_isolated_swing_speed_ur3e.py — 用 UR3e 的真實運動學鏈（Isaac Sim
內建 USD，官方資產路徑 Isaac/Robots/UniversalRobots/ur3e/ur3e.usd，已用
scripts/test_load_ur3e.py 確認可載入）重跑
scripts/test_isolated_swing_speed.py 同一套「Jacobian 偽逆解 Contact Pose
所需關節角速度」分析，檢查跟 WAM7 比起來轉速餘裕夠不夠。

跟 WAM7 版本的差異：
- UR3e 是 6-DOF（沒有冗餘自由度），這裡直接用 Isaac Sim 內建 USD 的**真實
  Jacobian**（`Articulation.get_jacobian_matrices()`），不用另外手刻 DH
  參數/正向運動學模型（`wam7_kinematics.py` 那套）——省去可能的 DH 參數
  轉譯錯誤風險，也更貼近真實 articulation 的行為。
- 這次不接球桿（不建 FixedJoint），只用 CUE_STICK_GRIP_TO_TIP=1.35m 這個
  既有常數，沿著挑定的揮桿方向做等效桿尖偏移修正——先回答「轉速餘裕夠不夠」
  這個核心問題，球桿怎麼掛在 UR3e 上是另一個題目。
- UR3e 沒有這個專案既有的 CANONICAL_REST_JOINTS 慣例，這裡改成：先讀 USD
  載入後的預設關節角，量一次 Jacobian 條件數，如果太接近奇異（常見於 UR
  機種的全 0 姿態），才手動調整到一個明顯非奇異的姿態。

跑法（需要 Isaac Sim headless，且需要能連到 Nucleus/CDN 載入 UR3e USD）：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/test_isolated_swing_speed_ur3e.py
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
_DLS_LAMBDA = 0.05


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


def _run() -> None:
    import numpy as np
    import omni.usd
    import omni.timeline
    from pxr import UsdPhysics, Sdf, Usd

    from isaacsim.core.experimental.prims import Articulation, RigidPrim

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from core.services.base_placement_calculator import CUE_STICK_GRIP_TO_TIP
    from core.services import swing_trajectory_calculator

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    prim_path = "/World/UR3eSwingSpeedTest"
    stage_api.create_reference_prim(prim_path, _UR3E_PATH)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(5):
        simulation_app.update()

    articulation = Articulation(paths=prim_path)
    for _ in range(5):
        simulation_app.update()

    dof_max_velocities = np.asarray(articulation.get_dof_max_velocities())
    if hasattr(dof_max_velocities, "numpy"):
        dof_max_velocities = dof_max_velocities.numpy()
    dof_max_velocities = np.asarray(dof_max_velocities, dtype=float).reshape(-1)
    num_joints = dof_max_velocities.size
    print(f"[ur3e] num_joints={num_joints}")
    print(f"[ur3e] dof_max_velocities(rad/s)={dof_max_velocities.tolist()}")
    print(f"[ur3e] dof_max_velocities(deg/s)={np.degrees(dof_max_velocities).tolist()}")

    default_joints = np.asarray(articulation.get_dof_positions())
    if hasattr(default_joints, "numpy"):
        default_joints = default_joints.numpy()
    default_joints = default_joints.reshape(-1)
    print(f"[ur3e] 預設關節角(rad)={default_joints.tolist()}")

    # 找 end-effector link：UR3e 最後一個連桿是 wrist_3_link（見
    # test_load_ur3e.py 掃出的子層清單）。用 link_names/body_names 找索引，
    # 比照 ArticulationAPIImpl._resolve_end_effector_jacobian_index() 的做法。
    end_effector_link_name = "wrist_3_link"
    link_names = None
    for attr in ("link_names", "body_names"):
        if hasattr(articulation, attr):
            link_names = list(getattr(articulation, attr))
            break
    if link_names is None:
        raise RuntimeError("Articulation 沒有 link_names/body_names 屬性，無法定位 end-effector")
    link_index = link_names.index(end_effector_link_name)

    # ⚠️ fixed-base articulation 的 Jacobian 不含 base link（跟
    # ArticulationAPIImpl._resolve_end_effector_jacobian_index() 同一個
    # 既有邏輯，第一版沒套用這個修正直接用 link_index 當 jac_link_index，
    # 導致索引對不上、程式無聲崩潰）。
    jac_probe = np.asarray(articulation.get_jacobian_matrices().numpy())[0]
    if jac_probe.shape[0] == len(link_names) - 1:
        jac_link_index = link_index - 1
    elif jac_probe.shape[0] == len(link_names):
        jac_link_index = link_index
    else:
        raise RuntimeError(
            f"Jacobian link 數 {jac_probe.shape[0]} 與 link 名稱數 {len(link_names)} 對不上"
        )
    print(f"[ur3e] link_names={link_names}  jacobian_rows={jac_probe.shape[0]}  link_index={link_index}  jac_link_index={jac_link_index}")

    end_effector_prim_path = f"{prim_path}/{end_effector_link_name}"
    end_effector_rigid_prim = RigidPrim(paths=end_effector_prim_path)

    def _current_orientation():
        _pos, orient = end_effector_rigid_prim.get_world_poses()
        return np.asarray(orient[0])

    def _jacobian_condition_and_manipulability(joints):
        articulation.set_dof_positions(joints[None, :])
        articulation.set_dof_velocities(np.zeros((1, num_joints)))
        for _ in range(3):
            simulation_app.update()
        jac_all = np.asarray(articulation.get_jacobian_matrices().numpy())[0]
        J = jac_all[jac_link_index]
        singular_values = np.linalg.svd(J, compute_uv=False)
        return J, singular_values

    J_default, sv_default = _jacobian_condition_and_manipulability(default_joints)
    print(f"[ur3e] 預設姿態 Jacobian singular values={sv_default.tolist()}  manipulability={float(np.prod(sv_default)):.6f}")

    # UR 家族全 0 姿態（手臂完全伸直朝上/前）常常接近奇異（elbow 伸直、
    # wrist 軸線重合），若最小奇異值太小就換一個常見的「彎肘」ready pose。
    if sv_default.min() < 0.05:
        print("[ur3e] 預設姿態疑似接近奇異（最小奇異值過小），改用常見的彎肘 ready pose 重試")
        ready_pose = np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])[:num_joints]
        joints = ready_pose
        J, sv = _jacobian_condition_and_manipulability(joints)
        print(f"[ur3e] ready pose Jacobian singular values={sv.tolist()}  manipulability={float(np.prod(sv)):.6f}")
    else:
        joints = default_joints
        J = J_default
        sv = sv_default

    contact_orientation = _current_orientation()
    direction = _rotate_vector_by_quat(contact_orientation, np.array([0.0, 0.0, 1.0]))
    direction = direction / np.linalg.norm(direction)
    print(f"[ur3e] contact_orientation={contact_orientation.tolist()}  揮桿方向(取末端 local +Z 世界向量)={direction.tolist()}")

    required_tip_speed = swing_trajectory_calculator.compute_required_tip_speed(_CUE_BALL_SPEED)
    print(f"[ur3e] required_tip_speed={required_tip_speed:.4f}  (cue_ball_speed={_CUE_BALL_SPEED})")

    Jv = J[:3, :]
    Jang = J[3:, :]
    tip_offset = CUE_STICK_GRIP_TO_TIP * direction
    Jv_tip = Jv - _skew_matrix(tip_offset) @ Jang
    J_tip_full = np.vstack([Jv_tip, Jang])

    target_twist = np.concatenate([required_tip_speed * direction, np.zeros(3)])
    JJt = J_tip_full @ J_tip_full.T + (_DLS_LAMBDA ** 2) * np.eye(6)
    qdot_contact = J_tip_full.T @ np.linalg.solve(JJt, target_twist)
    print(f"[ur3e] qdot_contact(rad/s)={np.round(qdot_contact, 4).tolist()}")
    print(f"[ur3e] qdot_contact(deg/s)={np.round(np.degrees(qdot_contact), 2).tolist()}")

    exceed = np.abs(qdot_contact) > dof_max_velocities
    if exceed.any():
        print(f"[ur3e] ⚠️ qdot_contact 有關節超過馬達最大角速度：{exceed.tolist()}")
        print(f"[ur3e]    超出量(rad/s)={np.round(np.abs(qdot_contact) - dof_max_velocities, 4).tolist()}")
    else:
        margin = dof_max_velocities - np.abs(qdot_contact)
        print(f"[ur3e] ✅ qdot_contact 全部關節都在馬達最大角速度限制內，餘裕(rad/s)={np.round(margin, 4).tolist()}")

    print(f"[ur3e] Jacobian singular values={sv.tolist()}  manipulability={float(np.prod(sv)):.6f}")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
