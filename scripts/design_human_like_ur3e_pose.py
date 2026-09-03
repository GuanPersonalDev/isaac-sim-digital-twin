"""
scripts/design_human_like_ur3e_pose.py — Phase A（確定 Strike 需要的姿態/轉速
需求）＋ Phase B（在滿足需求的前提下，搜尋符合人體擊球特徵的姿態），對象是
UR3e。見 plan：C:\\Users\\Kuan\\.claude\\plans\\ancient-skipping-wand.md。

使用者確認的人體化標準（AskUserQuestion，三項都要）：
  1. 揮桿主要由「肘關節」驅動，base/肩關節盡量維持靜止。
  2. 握把端（腕部）軌跡接近直線，不會左右/上下飄移。
  3. 手臂關節構型落在自然人體夠域內（不是接近伸直/接近極限彎曲）。

⚠️ Phase B 的做法跟原計畫「枚舉離散 IK 分支」有一處刻意的調整：這支腳本是
**空場景**（沒有球檯/母球），沒有一個外部給定、非改不可的 Cartesian contact
pose 目標——目標姿態本身就是這次要「設計」的東西。與其對一個任意選定的
目標點做反向 IK 再枚舉分支，這裡直接對（shoulder_lift, elbow, wrist1）做
正向网格搜尋（shoulder_pan/wrist2/wrist3 固定），每個候選姿態的 FK 結果
就是候選 Contact Pose——更直接、也更容易跟「肘關節主導」這個目標對齊
（見下方 direction_from_elbow 的做法）。這跟枚舉 IK 分支的精神一致（多個
候選、可行性+評分排序、人工核對挑一個），只是產生候選的機制不同，這裡
明文記錄，不是漏掉了計畫要求。

⚠️ 2026-09-01 二次修正：Phase B 原本用加權 DLS 偽逆＋角速度鎖死為 0 求解
關節轉速，實測對 UR3e（6-DOF 非冗餘、姿態鎖定=6 條等式對 6 個關節）完全
無效——J 是方陣滿秩時 qdot=J⁻¹·twist 是唯一解，任何加權矩陣都救不了，
最佳候選 shoulder_lift 仍要動到限制的 69%。改成直接讓其他關節角速度精確
為 0，只用肘關節那一欄 Jacobian（Jv_tip[:,elbow_idx]）算「純肘關節轉動」
能給桿尖多少速度、需要多少 ω_elbow——不用解線性系統，且 base/肩關節速度
精確為 0，不是「盡量小」。代價是放棄「角速度鎖死為 0」這個姿態鎖定假設
（純肘關節轉動天生會讓桿身跟著轉，這本來就是真人揮桿手肘擺動的物理，
不是要修正的誤差）。

跑法（需要 Isaac Sim headless）：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/design_human_like_ur3e_pose.py
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
_BACKSWING_SWEEP_DEG = float(os.environ.get("ISO_BACKSWING_SWEEP_DEG", "30.0"))

# Phase B 網格搜尋範圍（rad）。shoulder_pan 固定 0（isolated 場景沒有球檯
# 方位可以對齊，這個自由度留給正式整合時再決定）；wrist2/wrist3 固定，
# 因為它們主要負責姿態鎖定的微調，不是揮桿主軸。
_SHOULDER_PAN_FIXED = 0.0
_SHOULDER_LIFT_CANDIDATES = [-1.9, -1.7, -1.5, -1.3, -1.1]
_ELBOW_CANDIDATES = [-2.4, -2.1, -1.8, -1.5, -1.2, -0.9]
_WRIST1_CANDIDATES = [-1.6, -1.3, -1.0]
_WRIST2_FIXED = -1.5708
_WRIST3_FIXED = 0.0


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
    from pxr import UsdPhysics, Sdf

    from isaacsim.core.experimental.prims import Articulation, RigidPrim

    from isaac_sim_impl_6_0.stage_api_impl import StageAPIImpl
    from core.services.base_placement_calculator import CUE_STICK_GRIP_TO_TIP
    from core.services import swing_trajectory_calculator

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    prim_path = "/World/DesignHumanLikeUR3e"
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
    print(f"[design] num_joints={num_joints}  dof_max_velocities(rad/s)={dof_max_velocities.tolist()}")

    lower_limits, upper_limits = articulation.get_dof_limits()
    lower_limits = np.asarray(lower_limits.numpy() if hasattr(lower_limits, "numpy") else lower_limits, dtype=float).reshape(-1)
    upper_limits = np.asarray(upper_limits.numpy() if hasattr(upper_limits, "numpy") else upper_limits, dtype=float).reshape(-1)
    joint_mid = (lower_limits + upper_limits) / 2.0
    joint_half_range = (upper_limits - lower_limits) / 2.0
    print(f"[design] joint lower(rad)={lower_limits.tolist()}")
    print(f"[design] joint upper(rad)={upper_limits.tolist()}")

    dof_names = None
    for attr in ("dof_names", "joint_names"):
        if hasattr(articulation, attr):
            dof_names = list(getattr(articulation, attr))
            break
    print(f"[design] dof_names={dof_names}")

    link_names = None
    for attr in ("link_names", "body_names"):
        if hasattr(articulation, attr):
            link_names = list(getattr(articulation, attr))
            break
    if link_names is None:
        raise RuntimeError("Articulation 沒有 link_names/body_names 屬性，無法定位 link")

    def _jac_index(link_name, jac_rows):
        idx = link_names.index(link_name)
        if jac_rows == len(link_names) - 1:
            return idx - 1
        if jac_rows == len(link_names):
            return idx
        raise RuntimeError(f"Jacobian link 數 {jac_rows} 與 link 名稱數 {len(link_names)} 對不上")

    end_effector_link_name = "wrist_3_link"
    elbow_link_name = "forearm_link"

    jac_probe = np.asarray(articulation.get_jacobian_matrices().numpy())[0]
    jac_link_index = _jac_index(end_effector_link_name, jac_probe.shape[0])
    print(f"[design] link_names={link_names}  jacobian_rows={jac_probe.shape[0]}  jac_link_index(end_effector)={jac_link_index}")

    # ⚠️ dof_names 對應「哪個關節是 elbow」——若讀不到 dof_names，退回
    # UR 家族標準關節順序假設 [shoulder_pan, shoulder_lift, elbow, wrist1,
    # wrist2, wrist3]，第 3 個（index 2）是 elbow。
    if dof_names is not None:
        elbow_dof_index = next((i for i, n in enumerate(dof_names) if "elbow" in n.lower()), 2)
    else:
        elbow_dof_index = 2
    print(f"[design] elbow_dof_index={elbow_dof_index}")

    end_effector_rigid_prim = RigidPrim(paths=f"{prim_path}/{end_effector_link_name}")
    elbow_rigid_prim = RigidPrim(paths=f"{prim_path}/{elbow_link_name}")
    wrist1_rigid_prim = RigidPrim(paths=f"{prim_path}/wrist_1_link")

    def _settle():
        for _ in range(3):
            simulation_app.update()

    def _set_pose(joints):
        articulation.set_dof_positions(joints[None, :])
        articulation.set_dof_velocities(np.zeros((1, num_joints)))
        _settle()

    # ---- Phase A 步驟 2：真實量出 forearm 長度，秒級的「單純肘關節擺錘」
    # 合理性估算，不用等 Phase B 網格搜尋 ----
    _set_pose(np.array([0.0, -1.57, 1.57, -1.57, -1.57, 0.0])[:num_joints])
    elbow_pos, _ = elbow_rigid_prim.get_world_poses()
    wrist1_pos, _ = wrist1_rigid_prim.get_world_poses()
    forearm_length = float(np.linalg.norm(np.asarray(wrist1_pos[0]) - np.asarray(elbow_pos[0])))
    l_effective = forearm_length + CUE_STICK_GRIP_TO_TIP
    required_tip_speed = swing_trajectory_calculator.compute_required_tip_speed(_CUE_BALL_SPEED)
    omega_elbow_needed = required_tip_speed / l_effective
    elbow_limit = dof_max_velocities[elbow_dof_index]
    print("")
    print("=== Phase A：需求確定 ===")
    print(f"[A] forearm_length(elbow→wrist1 實測)={forearm_length:.4f} m")
    print(f"[A] L_effective(forearm+cue)={l_effective:.4f} m")
    print(f"[A] required_tip_speed={required_tip_speed:.4f} m/s  (cue_ball_speed={_CUE_BALL_SPEED})")
    print(f"[A] 單純肘關節擺錘估算 ω_elbow_needed={omega_elbow_needed:.4f} rad/s  vs elbow 限制={elbow_limit:.4f} rad/s")
    if omega_elbow_needed <= elbow_limit:
        margin_pct = 100.0 * (elbow_limit - omega_elbow_needed) / elbow_limit
        print(f"[A] [OK] 合理性檢查通過，理論餘裕 {margin_pct:.1f}%——只要姿態設計讓桿尖速度方向對齊肘關節轉動方向，肘關節轉速應該夠用")
    else:
        print(f"[A] [WARN] 合理性檢查未通過——即使理想對齊，肘關節本身轉速都不夠，需要先跟使用者討論方向本身是否可行，不建議繼續 Phase B")
        return

    backswing_sweep_rad = np.radians(_BACKSWING_SWEEP_DEG)
    print(f"[A] Backswing 回擺角度取 {_BACKSWING_SWEEP_DEG}°={backswing_sweep_rad:.4f} rad（待 Phase B 選定姿態後換算實際 backswing_distance）")

    # ---- Phase B：正向網格搜尋 + 純肘關節轉動（zero-order-hold，其餘關節
    # 完全鎖死）+ 評分 ----
    #
    # ⚠️ 2026-09-01 改版：原本用加權 DLS 偽逆＋「角速度鎖死為 0」的姿態鎖定
    # 目標，實測發現對 UR3e（6-DOF、非冗餘，姿態鎖定=6 條等式跟 6 個關節
    # 一一對應）完全無效——J 是方陣、滿秩時 qdot=J⁻¹·twist 是唯一解，跟
    # 任何加權矩陣 W 無關（只有存在 null space／冗餘自由度時，加權才能
    # 引導解往「哪些關節多動、哪些少動」偏移）。實測結果證實了這點：
    # shoulder_lift 仍然要動到 2.16 rad/s（接近它 3.14 rad/s 限制的 69%），
    # 完全不是「base/肩部盡量靜止」。
    #
    # 真正符合「揮桿主要由肘關節驅動，base/肩關節盡量靜止」這個人體化標準
    # 的做法，不是求解一個「盡量少用其他關節」的近似解——是直接把其他關節
    # 的角速度設為精確的 0，只讓肘關節單獨轉動（qdot = [0,0,ω_elbow,0,0,0]）。
    # 這同時也是為什麼要放棄「姿態鎖定」（角速度=0）這個目標的原因：純肘
    # 關節轉動天生會讓桿身跟著肘關節轉動軸一起轉（跟真人手肘揮桿的物理
    # 完全一致——手肘轉、前臂跟著轉，姿態本來就會跟著改變，不是異常），
    # 硬要求角速度=0 才是逼其他關節（尤其 shoulder）介入的根本原因。姿態
    # 只需要在「接觸瞬間」正確，不需要揮桿全程鎖死不變。
    #
    # 做法：對每個候選姿態，只看 Jacobian 的「肘關節那一欄」
    # （Jv_tip[:,elbow_idx]／Jang[:,elbow_idx]）——這欄本身就是「若只有肘
    # 關節以單位角速度轉動，桿尖會往哪個方向、多快移動」，不需要解任何
    # 線性系統。桿尖速度大小 = ω_elbow × |Jv_tip[:,elbow_idx]|，所以
    # ω_elbow_needed = required_tip_speed / |Jv_tip[:,elbow_idx]|——跟這個
    # 值比較 elbow 的馬達限制即可判斷可行性。方向本身沒有「對不對齊」的
    # 問題（這就是肘關節能提供的方向，直接拿來當這個候選姿態的揮桿方向）。
    print("")
    print("=== Phase B：姿態搜尋（正向網格 + 純肘關節轉動 + 人體化評分）===")

    candidates = []
    for shoulder_lift in _SHOULDER_LIFT_CANDIDATES:
        for elbow in _ELBOW_CANDIDATES:
            for wrist1 in _WRIST1_CANDIDATES:
                joints = np.array([_SHOULDER_PAN_FIXED, shoulder_lift, elbow, wrist1, _WRIST2_FIXED, _WRIST3_FIXED])[:num_joints]
                if np.any(joints < lower_limits) or np.any(joints > upper_limits):
                    continue

                _set_pose(joints)
                jac_all = np.asarray(articulation.get_jacobian_matrices().numpy())[0]
                J = jac_all[jac_link_index]
                singular_values = np.linalg.svd(J, compute_uv=False)
                if singular_values.min() < 1e-4:
                    continue  # 奇異姿態，跳過

                tip_pos, tip_orient = end_effector_rigid_prim.get_world_poses()
                tip_orient = np.asarray(tip_orient[0])
                # ⚠️ 沿用既有腳本的既有假設：桿身沿末端 local +Z 世界向量，
                # 未經真實掛桿驗證，見 test_isolated_swing_speed_ur3e.py 註解。
                cue_local_axis = np.array([0.0, 0.0, 1.0])
                tip_direction_guess = _rotate_vector_by_quat(tip_orient, cue_local_axis)
                tip_direction_guess = tip_direction_guess / np.linalg.norm(tip_direction_guess)
                tip_offset = CUE_STICK_GRIP_TO_TIP * tip_direction_guess

                Jv = J[:3, :]
                Jang = J[3:, :]
                Jv_tip = Jv - _skew_matrix(tip_offset) @ Jang

                elbow_col_v = Jv_tip[:, elbow_dof_index]
                elbow_col_w = Jang[:, elbow_dof_index]
                speed_per_unit_omega = float(np.linalg.norm(elbow_col_v))
                if speed_per_unit_omega < 1e-6:
                    continue  # 桿尖剛好落在肘關節轉動軸上，這個姿態無法靠肘關節推動桿尖

                direction_from_elbow = elbow_col_v / speed_per_unit_omega
                omega_elbow_needed_here = required_tip_speed / speed_per_unit_omega
                feasible = bool(omega_elbow_needed_here <= elbow_limit + 1e-9)

                qdot = np.zeros(num_joints)
                qdot[elbow_dof_index] = omega_elbow_needed_here
                tip_angular_speed = float(np.linalg.norm(elbow_col_w)) * omega_elbow_needed_here

                range_deviation = float(np.max(np.abs((joints - joint_mid) / joint_half_range)))
                elbow_world_pos, _ = elbow_rigid_prim.get_world_poses()
                lever_arm = float(np.linalg.norm(np.asarray(tip_pos[0]) + tip_offset - np.asarray(elbow_world_pos[0])))
                elbow_margin_ratio = float(omega_elbow_needed_here / elbow_limit)

                candidates.append({
                    "joints": joints.copy(),
                    "qdot": qdot.copy(),
                    "feasible": feasible,
                    "elbow_margin_ratio": elbow_margin_ratio,
                    "range_deviation": range_deviation,
                    "lever_arm": lever_arm,
                    "tip_pos": np.asarray(tip_pos[0]) + tip_offset,
                    "direction": direction_from_elbow,
                    "tip_angular_speed": tip_angular_speed,
                    "manipulability": float(np.prod(singular_values)),
                })

    print(f"[B] 網格總候選數={len(_SHOULDER_LIFT_CANDIDATES) * len(_ELBOW_CANDIDATES) * len(_WRIST1_CANDIDATES)}  有效候選(非奇異且在關節限位內)={len(candidates)}")

    feasible_candidates = [c for c in candidates if c["feasible"]]
    print(f"[B] 轉速可行候選數={len(feasible_candidates)} / {len(candidates)}  （這些候選 base/肩關節速度全部精確等於 0，不是近似值）")

    if not feasible_candidates:
        print("[B] [WARN] 沒有任何候選同時滿足肘關節轉速在限制內——如實回報，不硬選踩線的候選")
        candidates.sort(key=lambda c: c["elbow_margin_ratio"])
        print("[B] 肘關節轉速餘裕最好的前 5 名（皆不可行，僅供參考）：")
        for c in candidates[:5]:
            print(f"     joints={np.round(c['joints'],3).tolist()}  elbow_margin_ratio={c['elbow_margin_ratio']:.2f}x")
        return

    # 評分：肘關節轉速餘裕（越大代表越輕鬆，優先）＋關節夠域自然度＋桿尖
    # 角速度（越小代表握把軌跡越接近直線，對應人體化標準 2）。三項各自
    # 正規化到可行候選裡的最大值後加總，越小越好。
    max_elbow_margin = max(c["elbow_margin_ratio"] for c in feasible_candidates) or 1.0
    max_range_dev = max(c["range_deviation"] for c in feasible_candidates) or 1.0
    max_tip_angular = max(c["tip_angular_speed"] for c in feasible_candidates) or 1.0
    for c in feasible_candidates:
        c["score"] = (
            (c["elbow_margin_ratio"] / max_elbow_margin)
            + (c["range_deviation"] / max_range_dev)
            + (c["tip_angular_speed"] / max_tip_angular)
        )

    feasible_candidates.sort(key=lambda c: c["score"])

    print("")
    print("[B] 可行候選排行（前 5 名，分數越低越好）：")
    for rank, c in enumerate(feasible_candidates[:5], start=1):
        print(f"  #{rank} score={c['score']:.4f}")
        print(f"      joints(rad)={np.round(c['joints'], 4).tolist()}")
        print(f"      qdot(rad/s)={np.round(c['qdot'], 4).tolist()}  (base/肩關節精確為 0)")
        print(f"      肘關節轉速餘裕比例(需求/限制)={c['elbow_margin_ratio']:.4f}  關節夠域偏移(0=中點,1=貼限位)={c['range_deviation']:.4f}")
        print(f"      lever_arm(elbow→桿尖)={c['lever_arm']:.4f} m  桿尖角速度={c['tip_angular_speed']:.4f} rad/s  manipulability={c['manipulability']:.6f}")
        print(f"      direction_from_elbow={np.round(c['direction'], 4).tolist()}")

    best = feasible_candidates[0]
    backswing_distance = backswing_sweep_rad * best["lever_arm"]
    print("")
    print("=== 結論 ===")
    print(f"[結論] 最佳候選 joints={np.round(best['joints'], 4).tolist()}")
    print(f"[結論] qdot_contact={np.round(best['qdot'], 4).tolist()}  (dof_max_velocities={dof_max_velocities.tolist()})")
    print("[結論] base+肩關節速度精確為 0——完全符合『上手臂不動、靠手肘擺動』，不是近似值")
    print(f"[結論] 用這個姿態的實測 lever_arm={best['lever_arm']:.4f} m 換算 backswing_distance({_BACKSWING_SWEEP_DEG}°回擺)={backswing_distance:.4f} m")
    print(f"[結論] 桿尖角速度={best['tip_angular_speed']:.4f} rad/s（純肘關節轉動天生會讓桿身跟著轉，這裡只記錄大小，不是零；握把軌跡在契合期間會有一定弧度，不是精確直線）")
    print("[結論] Contact Pose 位置（含桿尖偏移）=", np.round(best["tip_pos"], 4).tolist())
    print("[結論] 這是人工核對用的候選，不是自動拍板——請跟這份輸出比對，確認是否要採用這組姿態。")


if __name__ == "__main__":
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    finally:
        simulation_app.close()
