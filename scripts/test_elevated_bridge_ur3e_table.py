"""
scripts/test_elevated_bridge_ur3e_table.py — 把高架橋（elevated bridge）姿態
需求補進「純肘關節轉動」計算邏輯，並在真正有球檯（不是空場景）的情況下，
用 UR3e 驗證高架橋擊球能不能達到目標桿尖速度、揮桿過程會不會撞到庫邊。

## 跟 scripts/design_human_like_ur3e_pose.py／test_ur3e_human_pose_swing_speed.py
## 的差異（這次新增的部分）

之前兩支腳本都是「空場景、自我參照方向」：Contact Pose 的揮桿方向就是
「肘關節那一欄 Jacobian 自己指向哪裡」，沒有一個外部給定、非對齊不可的
目標方向。高架橋案例不一樣——`cue_pose_calculator.compute_tilted_wrist_pose()`
／`compute_tilted_direction()` 已經把「因為庫邊高度擋住，桿身需要抬高
tilt_rad 角度」這個幾何需求算好了，是一個**外部給定的目標方向**，姿態
搜尋要對齊這個方向，不能再自我參照。

做法：Phase B 網格多加一個 `shoulder_pan` 維度（之前固定 0，這次真的當
自由變數掃——UR3e 的 `shoulder_pan_joint` 是繞基座 Z 軸轉的第一個關節，
角色跟 WAM7 的 `wam_base_yaw_joint` 一樣，負責把整條手臂繞基座旋轉到正確
的水平朝向；純繞 Z 軸旋轉不會改變 direction 的 Z 分量，只會轉 XY 分量，
所以 shoulder_lift/elbow/wrist1 這三個決定「傾斜幅度」的維度沿用之前搜尋
過、已知能給出各種 Z 分量的範圍，shoulder_pan 只負責把 XY 方向轉到目標
角度）。對每個候選姿態算出 `elbow_col_v`（肘關節那一欄 Jacobian 的線速度
部分）跟目標方向 `target_direction` 的**投影**（不是自我參照的方向），
`alignment = dot(elbow_col_v, target_direction) / |elbow_col_v|` 越接近 1
代表這個候選姿態「純肘關節轉動」的方向跟高架橋需要的方向越吻合。

選到最佳候選後，這組姿態在**任意位置**都能提供同一個 alignment（旋轉/
平移基座不改變關節相對構型算出來的 Jacobian 方向），所以只要把整支手臂
的基座**平移**（不需要旋轉——這個專案的既有慣例是基座只平移、旋轉靠
`base_yaw`/`shoulder_pan` 這個關節本身處理，跟 `RobotArm.reposition()`
介面只有 position 沒有 orientation 是同一個設計）到「這個姿態的桿尖剛好
落在目標接觸點」的位置即可，不需要任何四元數/旋轉矩陣運算。

## 範圍界定（這次刻意不做的部分）

- 不驗證「從 home 姿態安全接近到後擺姿態」這一段（正式的
  `compute_elevated_bridge_waypoints()` B1/B2/C1/C2 爬升-平移-轉向-下降
  序列，那是另一個獨立的問題，且目前是 WAM7 專用寫法，要用在 UR3e 需要
  另外設計）。這支腳本沿用之前兩支腳本的做法：直接瞬移到後擺姿態，只驗證
  「後擺→揮桿」這一段揮桿動作本身的速度與碰撞。
- roll（球桿繞自身軸的旋轉）不強制對齊 `cue_pose_calculator._ROLL_LOOKUP_GRID`
  查表值——那個查表值是專門為了閃避 WAM7 手臂本體形狀而調的，物理上球桿
  擊球本身不需要特定的 roll（圓柱桿頭繞自身軸旋轉對稱），對 UR3e（外形
  跟 WAM7 完全不同）沿用同一個角度沒有意義。這裡把 roll 當自由/浮現的
  結果（姿態搜尋自然決定），實際會不會撞庫邊靠真正的碰撞回報驗證，不是
  靠查表值保證。

跑法（需要 Isaac Sim headless）：
    ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=YES \\
    PYTHONIOENCODING=utf-8 PYTHONUTF8=1 \\
    "/c/Users/Kuan/isaac-project/venv/Scripts/python.exe" scripts/test_elevated_bridge_ur3e_table.py
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
_CUE_BALL = (0.0, -0.635)
"""高架橋案例代表點：見 cue_pose_calculator.py 的 9 點網格，這是唯一
tilt_rad 不是 None、且不屬於「任何 roll 都到不了目標球速」已知不可解那排
（y=-0.9382125）的點，tilt≈5.34°、最靠近球檯中線，最適合當第一個驗證
案例。"""
_SHOT_ANGLE_DEG = 0.0
_CUE_BALL_SPEED = 1.995
_SWEEP_DEG = 30.0
_ELBOW_DOF_INDEX = 2
_SETTLE_STEPS = 30
_EXTRA_STEPS_AFTER_T = 30

_DEBUG_NARROW_GRID = os.environ.get("BRIDGE_DEBUG_NARROW_GRID") == "1"
if _DEBUG_NARROW_GRID:
    # 除錯用：鎖定已知會通過 Stage 1/2 的那組候選，跳過昂貴的完整網格
    # （原本 1176 組要跑 ~90s），單獨快速重現後面 base 平移/建球檯/揮桿
    # 那一段的問題，不是正式行為的一部分。
    _SHOULDER_LIFT_CANDIDATES = [-0.4]
    _ELBOW_CANDIDATES = [2.8]
    _WRIST1_CANDIDATES = [-0.4]
    _WRIST2_CANDIDATES = [-0.5]
else:
    # 2026-09-01：原本 7×8×3×7=1176 組網格只找到 1 個候選（餘裕僅 6.4%，
    # 實測揮桿只達標 43.1%）。使用者選擇先提高解析度，看附近有沒有餘裕
    # 更大、網格間距沒踩到的姿態——elbow/wrist1/wrist2 加密，shoulder_lift
    # 維持原本 7 個值（總數 7×15×5×13=6825，控制在可接受的執行時間內）。
    _SHOULDER_LIFT_CANDIDATES = [-2.8, -2.4, -2.0, -1.6, -1.2, -0.8, -0.4]
    _ELBOW_CANDIDATES = [-2.8, -2.4, -2.0, -1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8]
    _WRIST1_CANDIDATES = [-1.6, -1.3, -1.0, -0.7, -0.4]
    _WRIST2_CANDIDATES = [-2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
"""2026-09-01 新增為搜尋維度：第一版固定 wrist2=-π/2，發現「Z 傾斜對齊」
跟「桿尖高度合理」這兩個篩選條件在這個固定值下完全湊不出交集（0 個候選
通過）——純肘關節轉動要給出很小的 Z 傾斜（這次目標只要 5.34°），力臂
（肘關節到桿尖的向量）需要接近垂直，但固定 wrist2 時「力臂接近垂直」
剛好也對應桿尖被推到很高（~1.85m，換算基座要陷進地板），兩個條件被
這個固定值绑死。wrist2 決定力臂在腕部之後怎麼折，加進來當自由變數才能
同時滿足兩個條件。"""
_WRIST3_FIXED = 0.0
_MIN_TIP_HEIGHT_M = 0.3
_MAX_TIP_HEIGHT_M = 0.85
"""桿尖離基座（Z=0 時）的合理高度範圍——第一版沒有這個限制，網格選中的
最佳候選（純看方向對齊）桿尖高度到 1.85m，換算基座位置要放到地板以下
1.7m，導致手臂整支陷進地板、物理直接爆掉（關節角瞬間飆到上萬 rad）。
WAM7 既有的 `_LOCAL_TIP_HEIGHT=0.787`（見 base_placement_calculator.py）
是這個專案已驗證可行的參考值，這裡抓一個涵蓋該值、但留了餘裕的範圍。"""
_MAX_Z_TILT_ERROR = 0.04
"""direction_from_elbow 的 Z 分量（=傾斜幅度，繞 shoulder_pan 轉動不會
改變這個分量，見下方 Stage 1/Stage 2 說明）跟 target_direction 的 Z 分量
容許誤差。2026-09-01：從 0.02 放寬到 0.04——0.02 時整個 1176 組網格只有
1 個候選通過，沒有空間讓「肘關節轉速餘裕」這個新加的排序條件發揮作用
（唯一候選餘裕只剩 6%，實測揮桿只達標 43%）。放寬容許誤差換取更多候選
可以比較。"""
_MIN_ALIGNMENT = 0.97  # Stage 2 方向對齊門檻：cos(alignment) 對應約 14° 誤差內


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
    """跟 extension/isaac_sim_impl_6_0/articulation_api_impl.py 的
    `_apply_velocity_targets_with_gravity_compensation()` 同一套做法，見該
    檔案的完整背景說明。"""
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
    from core.services import cue_pose_calculator, swing_trajectory_calculator

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath("/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))

    stage_api = StageAPIImpl()
    material_api = MaterialAPIImpl()
    rigid_body_api = RigidBodyAPIImpl()
    physics_api = PhysicsAPIImpl()

    # ---- 高架橋幾何需求（外部給定的目標方向/位置，見檔案開頭說明）----
    wrist, _orientation, tilt_rad, crossing = cue_pose_calculator.compute_tilted_wrist_pose(
        _CUE_BALL, _SHOT_ANGLE_DEG, _TABLE_Z, _BALL_RADIUS,
        position_offset=[0.0, 0.0], roll_rad=0.0,
    )
    if tilt_rad is None:
        raise RuntimeError(f"cue_ball={_CUE_BALL} 幾何無解（tilt_rad=None），換一個測試點")
    target_direction = cue_pose_calculator.compute_tilted_direction(_SHOT_ANGLE_DEG, tilt_rad)
    target_wrist_position = np.asarray(wrist, dtype=float)
    required_tip_speed = swing_trajectory_calculator.compute_required_tip_speed(_CUE_BALL_SPEED)
    print(f"[bridge] cue_ball={_CUE_BALL}  tilt_rad={tilt_rad:.4f} ({np.degrees(tilt_rad):.2f}°)  crossing={crossing}")
    print(f"[bridge] target_direction={target_direction.tolist()}")
    print(f"[bridge] target_wrist_position={target_wrist_position.tolist()}")
    print(f"[bridge] required_tip_speed={required_tip_speed:.4f} m/s  (cue_ball_speed={_CUE_BALL_SPEED})")

    # ---- 先建 UR3e（暫時位置，姿態搜尋用，還沒接球檯）----
    robot_base_path = "/World/BridgeUR3eTest"
    robot_prim_path = robot_base_path + "/Robot"
    stage_api.create_reference_prim(robot_prim_path, _UR3E_PATH)

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

    lower_limits, upper_limits = articulation.get_dof_limits()
    lower_limits = np.asarray(lower_limits.numpy() if hasattr(lower_limits, "numpy") else lower_limits, dtype=float).reshape(-1)
    upper_limits = np.asarray(upper_limits.numpy() if hasattr(upper_limits, "numpy") else upper_limits, dtype=float).reshape(-1)
    joint_mid = (lower_limits + upper_limits) / 2.0
    joint_half_range = (upper_limits - lower_limits) / 2.0

    dof_names = list(articulation.dof_names) if hasattr(articulation, "dof_names") else None
    elbow_dof_index = (
        next((i for i, n in enumerate(dof_names) if "elbow" in n.lower()), _ELBOW_DOF_INDEX)
        if dof_names is not None else _ELBOW_DOF_INDEX
    )

    link_names = None
    for attr in ("link_names", "body_names"):
        if hasattr(articulation, attr):
            link_names = list(getattr(articulation, attr))
            break
    end_effector_link_name = "wrist_3_link"
    idx = link_names.index(end_effector_link_name)
    jac_probe = np.asarray(articulation.get_jacobian_matrices().numpy())[0]
    if jac_probe.shape[0] == len(link_names) - 1:
        jac_link_index = idx - 1
    elif jac_probe.shape[0] == len(link_names):
        jac_link_index = idx
    else:
        raise RuntimeError(f"Jacobian link 數 {jac_probe.shape[0]} 與 link 名稱數 {len(link_names)} 對不上")

    end_effector_rigid_prim = RigidPrim(paths=f"{robot_prim_path}/{end_effector_link_name}")
    elbow_limit = dof_max_velocities[elbow_dof_index]
    print(f"[bridge] num_joints={num_joints}  elbow_dof_index={elbow_dof_index}  elbow_limit={elbow_limit:.4f} rad/s")

    def _set_pose(joints):
        articulation.set_dof_positions(joints[None, :])
        articulation.set_dof_velocities(np.zeros((1, num_joints)))
        for _ in range(3):
            simulation_app.update()

    # ---- Phase B Stage 1：pan=0 時掃 (shoulder_lift, elbow, wrist1, wrist2)，
    # 篩選「桿尖高度合理」+「Z 傾斜跟目標一致」的候選。繞 shoulder_pan
    # （Z 軸）旋轉不會改變任何向量的 Z 分量，也不會改變桿尖離基座的高度，
    # 所以這兩項可以在 pan=0 時就先篩，不用跟 pan 一起網格搜尋（第一版
    # 4 維網格會爆炸到數千組，且是這樣長出一個「桿尖高度 1.85m→基座陷到
    # 地板下 1.7m」的荒謬解，才改成這個兩階段做法）----
    print("")
    print("=== Phase B Stage 1：掃 shoulder_lift/elbow/wrist1/wrist2（pan=0），篩選高度合理+Z傾斜對齊的候選 ===")
    stage1_candidates = []
    total_grid = len(_SHOULDER_LIFT_CANDIDATES) * len(_ELBOW_CANDIDATES) * len(_WRIST1_CANDIDATES) * len(_WRIST2_CANDIDATES)
    for shoulder_lift in _SHOULDER_LIFT_CANDIDATES:
        for elbow in _ELBOW_CANDIDATES:
            for wrist1 in _WRIST1_CANDIDATES:
                for wrist2 in _WRIST2_CANDIDATES:
                    joints = np.array([0.0, shoulder_lift, elbow, wrist1, wrist2, _WRIST3_FIXED])[:num_joints]
                    if np.any(joints < lower_limits) or np.any(joints > upper_limits):
                        continue

                    _set_pose(joints)
                    jac_all = np.asarray(articulation.get_jacobian_matrices().numpy())[0]
                    J = jac_all[jac_link_index]
                    singular_values = np.linalg.svd(J, compute_uv=False)
                    if singular_values.min() < 1e-4:
                        continue

                    tip_pos, tip_orient = end_effector_rigid_prim.get_world_poses()
                    tip_orient = np.asarray(tip_orient[0])
                    cue_local_axis = np.array([0.0, 0.0, 1.0])
                    tip_direction_guess = _rotate_vector_by_quat(tip_orient, cue_local_axis)
                    tip_direction_guess = tip_direction_guess / np.linalg.norm(tip_direction_guess)
                    tip_offset = CUE_STICK_GRIP_TO_TIP * tip_direction_guess
                    local_tip_position = np.asarray(tip_pos[0]) + tip_offset

                    if not (_MIN_TIP_HEIGHT_M <= local_tip_position[2] <= _MAX_TIP_HEIGHT_M):
                        continue

                    # ⚠️ 中段連桿的地板淨空檢查試過但拿掉了：用跟桿尖同一個
                    # _MIN_TIP_HEIGHT_M 門檻套用在 forearm_link/wrist_1/2_link
                    # 上，結果連已經被真實物理驗證過安全（settle 階段只有
                    # impulse 0.27 的輕微觸碰）的既有候選都被這個門檻誤判
                    # 剔除——手臂中段連桿本來就常常比桿尖低（腕部鏈路把桿尖
                    # 往上抬），不能直接套桿尖的高度門檻。這裡沒有可靠的
                    # 解析代理能判斷「連桿多低算真的會撞地板」（不知道真實
                    # 地板的精確世界座標），改回只靠 Stage 2 選出最終候選後
                    # 的真實 settle 階段碰撞回報（見下方）當安全把關，不在
                    # Stage 1 網格階段做這個不可靠的預篩。

                    Jv = J[:3, :]
                    Jang = J[3:, :]
                    Jv_tip = Jv - _skew_matrix(tip_offset) @ Jang
                    elbow_col_v = Jv_tip[:, elbow_dof_index]
                    speed_per_unit_omega = float(np.linalg.norm(elbow_col_v))
                    if speed_per_unit_omega < 1e-6:
                        continue
                    direction_local = elbow_col_v / speed_per_unit_omega

                    z_tilt_error = abs(float(direction_local[2]) - float(target_direction[2]))
                    if z_tilt_error > _MAX_Z_TILT_ERROR:
                        continue

                    # ⚠️ 2026-09-01 新增：第一版排行只看 z_tilt_error/range_
                    # deviation，選到的候選（joints=[0,-0.4,2.8,-0.4,-0.5,0]）
                    # 肘關節轉速餘裕只剩 6%（omega_elbow_needed=2.94 vs 限制
                    # 3.14），實測真正執行 quintic 揮桿只達到目標速度的
                    # 43.1%——跟這次工作階段稍早（design_human_like_ur3e_
                    # pose.py）驗證過的教訓一樣：理論上「可行」（<=限制）
                    # 不等於「餘裕夠、實際追蹤得上」，餘裕太小時真實動態追蹤
                    # 誤差會把達成率吃掉一大截。這裡在 Stage 1 就先估
                    # omega_elbow_needed（用 pan=0 的 speed_per_unit_omega
                    # 估，跟 pan 轉正後的實際值只差在 alignment≈1 這個微小
                    # 因子，估計已經足夠準），優先選餘裕大的候選，不是隨便
                    # 選第一個「理論上可行」的。
                    omega_elbow_needed_est = required_tip_speed / speed_per_unit_omega
                    elbow_margin_ratio_est = omega_elbow_needed_est / elbow_limit

                    range_deviation = float(np.max(np.abs((joints - joint_mid) / joint_half_range)))

                    stage1_candidates.append({
                        "joints": joints.copy(),
                        "direction_local": direction_local,
                        "local_tip_position": local_tip_position,
                        "z_tilt_error": z_tilt_error,
                        "range_deviation": range_deviation,
                        "elbow_margin_ratio_est": elbow_margin_ratio_est,
                    })

    print(f"[bridge] Stage 1 網格總候選數={total_grid}  高度合理+Z傾斜對齊候選數={len(stage1_candidates)}")

    if not stage1_candidates:
        print(f"[bridge] [WARN] 沒有任何候選同時滿足桿尖高度合理({_MIN_TIP_HEIGHT_M}~{_MAX_TIP_HEIGHT_M}m)+Z傾斜對齊(誤差<{_MAX_Z_TILT_ERROR})——需要放寬範圍或換一組候選值")
        return

    max_range_dev_1 = max(c["range_deviation"] for c in stage1_candidates) or 1.0
    max_z_err = max(c["z_tilt_error"] for c in stage1_candidates) or 1.0
    max_margin_est = max(c["elbow_margin_ratio_est"] for c in stage1_candidates) or 1.0
    for c in stage1_candidates:
        c["stage1_score"] = (
            (c["z_tilt_error"] / max_z_err) * 5.0
            + (c["range_deviation"] / max_range_dev_1)
            + (c["elbow_margin_ratio_est"] / max_margin_est) * 5.0
        )
    stage1_candidates.sort(key=lambda c: c["stage1_score"])

    print("[bridge] Stage 1 候選排行（前 10 名）：")
    for rank, c in enumerate(stage1_candidates[:10], start=1):
        print(f"  #{rank} stage1_score={c['stage1_score']:.4f}  z_tilt_error={c['z_tilt_error']:.5f}  tip_height={c['local_tip_position'][2]:.4f}  elbow_margin_ratio_est={c['elbow_margin_ratio_est']:.4f}")
        print(f"      joints(pan=0)(rad)={np.round(c['joints'], 4).tolist()}")

    # ---- Stage 1.5（預設關閉）：對網格冠軍做局部連續精修（scipy
    # Nelder-Mead）。網格排行 #1 的餘裕只剩 6.4%（elbow_margin_ratio_est=
    # 0.9364），把 Z 傾斜容許誤差從 0.02 放寬到 0.04 也還是只有這一個候選
    # 通過篩選——4 維網格在這個離散解析度下就是只採樣到這一個點附近，不
    # 代表這附近的連續參數空間沒有餘裕更大的姿態，只是網格間距沒踩到。
    #
    # ⚠️ 2026-09-01 實測發現這個精修不安全，預設關閉：目標函式只檢查
    # 「桿尖高度」跟「Z傾斜誤差」，沒有檢查手臂其他連桿（wrist_2/wrist_3/
    # forearm 等）離地板的淨空——精修一次找到 elbow_margin_ratio_est
    # 從 0.94 降到 0.59（理論上餘裕變好很多）的解，但那組關節角讓
    # wrist_3_link/forearm_link 等直接撞進地板（settle 階段 impulse 62~217，
    # 手臂整個被撞歪，之後 elbow_angle_error 卡死不變、達成率算出離譜的
    # 376.6%——整段物理已經不合法，不是真的達標）。要安全啟用這個精修，
    # 目標函式需要額外加上「整支手臂每個連桿離地板淨空」的限制式，不是
    # 只看桿尖，這個延伸還沒做，先把這步驟關掉，用網格冠軍（已知安全，
    # 只是餘裕小、實測達成率 43%）當最終結果，如實回報限制。
    _ENABLE_STAGE_1_5_REFINEMENT = os.environ.get("BRIDGE_ENABLE_STAGE_1_5") == "1"
    print("")
    if not _ENABLE_STAGE_1_5_REFINEMENT:
        print("=== Phase B Stage 1.5：預設關閉（缺少手臂連桿地板淨空檢查，實測會精修出撞地板的解）===")
    else:
        print("=== Phase B Stage 1.5：對網格冠軍做局部連續精修（scipy Nelder-Mead）===")
        from scipy.optimize import minimize

        def _evaluate_bridge_pose(params):
            shoulder_lift, elbow, wrist1, wrist2 = params
            joints = np.clip(
                np.array([0.0, shoulder_lift, elbow, wrist1, wrist2, _WRIST3_FIXED])[:num_joints],
                lower_limits, upper_limits,
            )
            _set_pose(joints)
            jac_all = np.asarray(articulation.get_jacobian_matrices().numpy())[0]
            J = jac_all[jac_link_index]
            singular_values = np.linalg.svd(J, compute_uv=False)

            tip_pos, tip_orient = end_effector_rigid_prim.get_world_poses()
            tip_orient = np.asarray(tip_orient[0])
            cue_local_axis = np.array([0.0, 0.0, 1.0])
            tip_direction_guess = _rotate_vector_by_quat(tip_orient, cue_local_axis)
            tip_direction_guess = tip_direction_guess / np.linalg.norm(tip_direction_guess)
            tip_offset = CUE_STICK_GRIP_TO_TIP * tip_direction_guess
            local_tip_position = np.asarray(tip_pos[0]) + tip_offset

            Jv = J[:3, :]
            Jang = J[3:, :]
            Jv_tip = Jv - _skew_matrix(tip_offset) @ Jang
            elbow_col_v = Jv_tip[:, elbow_dof_index]
            speed_per_unit_omega = float(np.linalg.norm(elbow_col_v))
            if speed_per_unit_omega < 1e-6 or singular_values.min() < 1e-4:
                return None

            direction_local = elbow_col_v / speed_per_unit_omega
            z_tilt_error = abs(float(direction_local[2]) - float(target_direction[2]))
            height = float(local_tip_position[2])
            margin_ratio = (required_tip_speed / speed_per_unit_omega) / elbow_limit
            range_deviation = float(np.max(np.abs((joints - joint_mid) / joint_half_range)))
            return {
                "joints": joints,
                "direction_local": direction_local,
                "local_tip_position": local_tip_position,
                "z_tilt_error": z_tilt_error,
                "height": height,
                "elbow_margin_ratio_est": margin_ratio,
                "range_deviation": range_deviation,
            }

        def _penalized_objective(params):
            result = _evaluate_bridge_pose(params)
            if result is None:
                return 1e6
            penalty = 0.0
            if result["z_tilt_error"] > _MAX_Z_TILT_ERROR:
                penalty += 50.0 * (result["z_tilt_error"] - _MAX_Z_TILT_ERROR)
            if result["height"] < _MIN_TIP_HEIGHT_M:
                penalty += 50.0 * (_MIN_TIP_HEIGHT_M - result["height"])
            if result["height"] > _MAX_TIP_HEIGHT_M:
                penalty += 50.0 * (result["height"] - _MAX_TIP_HEIGHT_M)
            return result["elbow_margin_ratio_est"] + penalty

        x0 = stage1_candidates[0]["joints"][1:5]
        opt_result = minimize(
            _penalized_objective, x0, method="Nelder-Mead",
            options={"maxiter": 150, "xatol": 1e-3, "fatol": 1e-4},
        )
        refined = _evaluate_bridge_pose(opt_result.x)
        original_margin = stage1_candidates[0]["elbow_margin_ratio_est"]
        if (
            refined is not None
            and refined["z_tilt_error"] <= _MAX_Z_TILT_ERROR
            and _MIN_TIP_HEIGHT_M <= refined["height"] <= _MAX_TIP_HEIGHT_M
            and refined["elbow_margin_ratio_est"] < original_margin
        ):
            print(f"[bridge] 精修成功：elbow_margin_ratio_est {original_margin:.4f} -> {refined['elbow_margin_ratio_est']:.4f}")
            print(f"[bridge]   joints(pan=0)={np.round(refined['joints'], 4).tolist()}  tip_height={refined['height']:.4f}  z_tilt_error={refined['z_tilt_error']:.5f}")
            print("[bridge] [WARN] 這個精修結果沒有檢查手臂其他連桿的地板淨空，務必在後續的真實 settle 階段確認沒有大衝量碰撞再採信")
            refined["stage1_score"] = -1.0  # 精修結果優先，強制排第一
            stage1_candidates.insert(0, refined)
        else:
            reason = "無解" if refined is None else f"z_tilt_error={refined['z_tilt_error']:.5f} height={refined['height']:.4f} margin={refined['elbow_margin_ratio_est']:.4f}"
            print(f"[bridge] 精修沒有改善（{reason}），沿用網格冠軍")

    # ---- Stage 2：對 Stage 1 前幾名，解析算出對齊 target_direction 的 XY
    # 需要的 shoulder_pan——繞 Z 軸旋轉只改變 XY 方向角，不改變 Z 分量／
    # XY 幅值，不需要再網格搜尋，直接用 atan2 差角求解。----
    print("")
    print("=== Phase B Stage 2：解析解 shoulder_pan（不需要網格搜尋）===")
    best = None
    for c in stage1_candidates[:10]:
        direction_local = c["direction_local"]
        angle_current_xy = float(np.arctan2(direction_local[1], direction_local[0]))
        angle_target_xy = float(np.arctan2(target_direction[1], target_direction[0]))
        required_pan = angle_target_xy - angle_current_xy
        required_pan = (required_pan + np.pi) % (2 * np.pi) - np.pi  # wrap 到 [-pi, pi]

        joints = c["joints"].copy()
        joints[0] = required_pan
        _set_pose(joints)

        jac_all = np.asarray(articulation.get_jacobian_matrices().numpy())[0]
        J = jac_all[jac_link_index]
        tip_pos, tip_orient = end_effector_rigid_prim.get_world_poses()
        tip_orient = np.asarray(tip_orient[0])
        cue_local_axis = np.array([0.0, 0.0, 1.0])
        tip_direction_guess = _rotate_vector_by_quat(tip_orient, cue_local_axis)
        tip_direction_guess = tip_direction_guess / np.linalg.norm(tip_direction_guess)
        tip_offset = CUE_STICK_GRIP_TO_TIP * tip_direction_guess
        local_tip_position = np.asarray(tip_pos[0]) + tip_offset

        Jv = J[:3, :]
        Jang = J[3:, :]
        Jv_tip = Jv - _skew_matrix(tip_offset) @ Jang
        elbow_col_v = Jv_tip[:, elbow_dof_index]
        speed_per_unit_omega = float(np.linalg.norm(elbow_col_v))
        along_target_speed_per_unit_omega = float(np.dot(elbow_col_v, target_direction))
        alignment = along_target_speed_per_unit_omega / speed_per_unit_omega if speed_per_unit_omega > 1e-6 else -1.0

        print(f"[bridge] required_pan={np.degrees(required_pan):.2f}°  alignment={alignment:.5f}  tip_height={local_tip_position[2]:.4f}", end="  ")

        if alignment < _MIN_ALIGNMENT:
            print("未達對齊門檻，跳過")
            continue

        omega_elbow_needed_here = required_tip_speed / along_target_speed_per_unit_omega
        feasible = bool(omega_elbow_needed_here <= elbow_limit + 1e-9)
        print(f"omega_elbow_needed={omega_elbow_needed_here:.4f}  feasible={feasible}")

        if not feasible:
            continue

        best = {
            "joints": joints.copy(),
            "alignment": alignment,
            "omega_elbow_needed": omega_elbow_needed_here,
            "local_tip_position": local_tip_position,
        }
        break  # Stage 1 已經依 z_tilt_error/夠域排序，第一個通過的就是目前最佳選擇

    if best is None:
        print(f"[bridge] [WARN] Stage 1 前 10 名都無法在 Stage 2 通過對齊/可行性檢查——需要跟使用者討論（放寬範圍、換測試點）")
        return

    joints_contact = best["joints"]
    omega_elbow_needed = best["omega_elbow_needed"]
    local_tip_position = best["local_tip_position"]

    # ---- 把基座平移到「這個姿態桿尖剛好落在目標接觸點」的位置 ----
    base_position = (target_wrist_position - local_tip_position).tolist()
    print("")
    print(f"[bridge] 選定姿態 joints_contact={joints_contact.tolist()}")
    print(f"[bridge] local_tip_position(基座在原點時)={local_tip_position.tolist()}")
    print(f"[bridge] 換算 base_position={base_position}")
    if base_position[2] < -1.0 or base_position[2] > 0.5:
        # 安全防呆：WAM7 既有 base_z≈-0.6 是這個專案唯一驗證過「手臂本體
        # 不會陷進地板/桌面」的參考值，這裡的高度篩選（_MIN/MAX_TIP_
        # HEIGHT_M）理論上該把 base_z 限制在鄰近範圍——第一版就是沒有這個
        # 檢查，選出 base_z=-1.7（陷進地板 1m 以上）才讓整支手臂物理爆掉
        # （關節角瞬間飆到上萬 rad）。這裡不是把問題修好，是在問題重演時
        # 提早中止並如實回報，不要浪費時間跑一次注定爆炸的物理模擬。
        raise RuntimeError(
            f"base_position Z={base_position[2]:.4f} 超出合理範圍（預期接近 WAM7 參考值 -0.6 附近），"
            "很可能會讓手臂陷進地板或懸空過高，中止執行，需要檢查 Stage 1 的高度篩選範圍"
        )
    stage_api.set_prim_translate(robot_prim_path, *base_position)
    for _ in range(5):
        simulation_app.update()

    # ---- 平移後驗證：桿尖世界座標應該等於 target_wrist_position ----
    _set_pose(joints_contact)
    verify_tip_pos, verify_tip_orient = end_effector_rigid_prim.get_world_poses()
    verify_tip_orient = np.asarray(verify_tip_orient[0])
    verify_direction_guess = _rotate_vector_by_quat(verify_tip_orient, np.array([0.0, 0.0, 1.0]))
    verify_direction_guess = verify_direction_guess / np.linalg.norm(verify_direction_guess)
    verify_tip_offset = CUE_STICK_GRIP_TO_TIP * verify_direction_guess
    verify_world_tip = np.asarray(verify_tip_pos[0]) + verify_tip_offset
    position_error = float(np.linalg.norm(verify_world_tip - target_wrist_position))
    print(f"[bridge] 平移後實測桿尖位置={verify_world_tip.tolist()}  跟目標誤差={position_error:.5f} m（應該接近 0，純平移不該改變關節構型的相對幾何）")

    # ---- 建球檯（真正的碰撞幾何）----
    table_base_path = "/World/BridgeUR3eTableTest"
    table = BilliardTable(table_base_path, stage_api, material_api, rigid_body_api, (0.0, 0.0))
    ball_positions = {i: (5.0 + i * 0.2, 5.0) for i in range(10)}
    ball_positions[0] = _CUE_BALL
    table.get_table_ball_set().build(ball_positions)
    ball_prim_path = table.get_table_ball_set().get_ball_prim_paths()[0]
    ball_rigid_prim = RigidPrim(paths=ball_prim_path)

    # ---- 掛球桿（比照 TableRobotManager 的做法，因為 UR3e 目前沒有實作
    # RobotArm 介面，不能直接用 TableRobotManager，手動複刻同一段邏輯）----
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
    # 手臂本體也開碰撞回報——高架橋案例最早在 WAM7 上出過「手臂本體（不是
    # 桿頭）掃過庫邊」的問題（見 cue_pose_calculator.py 的歷史說明），這裡
    # 一併檢查，不只看球桿。
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_prim_path)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
            physics_api.enable_contact_reporting(str(prim.GetPath()))
    physics_api.subscribe_contact_events(lambda e: contacts.append((_step_counter["value"], e)))

    # ---- 重新讀 Jacobian 確認基座平移後結果不變（純平移不該影響關節
    # 空間的相對幾何/Jacobian，這裡當一個交叉驗證，不是必要步驟）----
    jac_all = np.asarray(articulation.get_jacobian_matrices().numpy())[0]
    J = jac_all[jac_link_index]
    Jv = J[:3, :]
    Jang = J[3:, :]
    Jv_tip = Jv - _skew_matrix(verify_tip_offset) @ Jang
    elbow_col_v = Jv_tip[:, elbow_dof_index]
    direction = elbow_col_v / np.linalg.norm(elbow_col_v)
    print(f"[bridge] 平移後 direction_from_elbow={direction.tolist()}  target_direction={target_direction.tolist()}  dot={float(np.dot(direction, target_direction)):.5f}")

    # ---- Backswing Pose：只有肘關節角度往回轉 sweep_rad ----
    sweep_rad = np.radians(_SWEEP_DEG)
    joints_backswing = joints_contact.copy()
    joints_backswing[elbow_dof_index] -= sweep_rad
    print(f"[bridge] sweep_rad={sweep_rad:.4f} ({_SWEEP_DEG}°)  joints_backswing={joints_backswing.tolist()}")

    q0 = float(joints_backswing[elbow_dof_index])
    q1 = float(joints_contact[elbow_dof_index])
    v1 = float(omega_elbow_needed)
    T = max(abs(q1 - q0) / max(v1, 1e-6), 0.05)
    for _attempt in range(50):
        c3, c4, c5 = _solve_quintic_coeffs(q0, q1, v1, T)
        peak_velocity = _peak_abs_velocity(c3, c4, c5, T)
        if peak_velocity <= elbow_limit + 1e-9:
            break
        T *= (peak_velocity / elbow_limit) * 1.05
    print(f"[bridge] quintic T={T:.4f}s  peak_elbow_velocity={peak_velocity:.4f} rad/s")

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
    print(f"[bridge] 瞬移＋穩定後實際關節角={live_joints_before.tolist()}")
    ball_vel_before, _ = ball_rigid_prim.get_velocities()
    print(f"[bridge] settle 後母球速度={np.asarray(ball_vel_before[0]).tolist()}（應接近 0，代表 settle 階段沒有碰到球）")

    # ---- velocity-mode 逐 tick 餵 q̇(t)：只有肘關節非零 ----
    physics_dt = 1.0 / 60.0
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
            print(f"[bridge] step={step} t={t:.3f}  qdot_ref_elbow={qdot_ref[elbow_dof_index]:.4f}  elbow_angle_error={elbow_angle_error:.4f}")

    history.sort(key=lambda h: h[0])
    _, elbow_angle_error_at_contact, tip_velocity = history[0]
    tip_speed_along_direction = float(np.dot(tip_velocity, target_direction))
    tip_speed_total = float(np.linalg.norm(tip_velocity))

    print("")
    print(f"[bridge] 接觸瞬間肘關節角度誤差={elbow_angle_error_at_contact:.5f} rad")
    print(f"[bridge] 接觸瞬間桿尖速度向量={tip_velocity.tolist()}")
    print(f"[bridge] 沿 target_direction 分量={tip_speed_along_direction:.4f} m/s  總速度={tip_speed_total:.4f} m/s")
    print(f"[bridge] required_tip_speed={required_tip_speed:.4f} m/s  達成率(沿方向)={100 * tip_speed_along_direction / required_tip_speed:.1f}%")

    ball_vel_after, _ = ball_rigid_prim.get_velocities()
    print(f"[bridge] 揮桿結束後母球速度={np.asarray(ball_vel_after[0]).tolist()}")

    # ---- 碰撞回報彙總：分開列「碰到母球」vs「碰到其他東西（庫邊/桌面/
    # 手臂自己）」，後者才是要檢查的「撞到庫邊」問題 ----
    ball_contacts = [(s, e) for s, e in contacts if ball_prim_path in (e.actor_path_a, e.actor_path_b)]
    other_contacts = [(s, e) for s, e in contacts if ball_prim_path not in (e.actor_path_a, e.actor_path_b)]
    print("")
    print(f"[bridge] 母球碰撞事件數={len(ball_contacts)}")
    for step_idx, e in ball_contacts:
        phase = "settle" if step_idx < _SETTLE_STEPS else "swing"
        print(f"    [{phase}] step={step_idx}  {e.actor_path_a} <-> {e.actor_path_b}  impulse={e.impulse:.4f}")
    print(f"[bridge] 其他碰撞事件數（庫邊/桌面/手臂自撞）={len(other_contacts)}")
    for step_idx, e in other_contacts:
        phase = "settle" if step_idx < _SETTLE_STEPS else "swing"
        print(f"    [{phase}] step={step_idx}  {e.actor_path_a} <-> {e.actor_path_b}  impulse={e.impulse:.4f}")

    if other_contacts:
        print("[bridge] [WARN] 揮桿過程中偵測到跟球以外的東西碰撞——需要檢查是不是撞到庫邊/桌面")
    else:
        print("[bridge] [OK] 全程沒有偵測到跟球以外的東西碰撞")


if __name__ == "__main__":
    import traceback

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    try:
        _run()
    except BaseException:
        # ⚠️ 上一輪執行在 Stage 2 一個 print() 之後、下一行還沒印出來就
        # 整個 process 結束，log 裡完全沒有 Python traceback（`try/finally`
        # 照理說 unhandled exception 也會印 traceback 才對）——懷疑是
        # Kit/PhysX 原生層級的問題，不是單純 Python 例外，這裡改成明確
        # `except BaseException` 把任何例外都印出完整 traceback 再重新拋出，
        # 排除「例外被吞掉沒印出來」這個可能性，方便下次重現時定位。
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
