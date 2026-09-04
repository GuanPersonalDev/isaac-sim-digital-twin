"""UR10e closed-form（解析解）逆向運動學。

2026-09-04 新增，動機見 `ur10e_placement_calculator.compute_roll_minimizing_
reorientation()` 的除錯記錄：RMPflow＋收尾差動 IK 是反應式/疊代式方法，在
某些姿態附近會卡在運動學奇異點（實測：wrist_2 附近方向誤差卡在 0.0294
rad 不再收斂，同時把已經收斂的位置也拖差），這是 DLS 類方法在奇異點附近
的已知結構性限制（精度與數值穩定互相拖累），不是調參能解決的。

UR 家族手臂（含 UR10e）幾何上屬於「三個手腕軸交於一點」的 6R 手臂，存在
標準的 closed-form 解（不需要疊代，也就不會有「卡在奇異點附近收斂不完」
這種問題）。本檔案是這個 closed-form 演算法的移植版——演算法結構本身
（DH 參數表的形式、8 組解的推導順序）沿用業界廣泛使用、經過驗證的公開
實作（Ryan Keating, Johns Hopkins University；同一套結構也是 ROS-Industrial
`ur_kinematics` 套件的基礎），只替換成這個專案 `assets/rmpflow_config/
ur10e_cue/ur10e.urdf` 裡的 UR10e 實際 DH 數值（d1/a2/a3/d4/d5/d6，直接從
各關節 `<origin>` 的位移量讀出，UR10e 版跟舊版 UR10 的實際數值不同）。

⚠️ 這裡算出的關節角，是相對於這個演算法自己的 DH 慣例座標系（base 端
「frame 0」／末端「frame 6」），不保證直接等於 Isaac Sim `base_link`／
`wrist_3_link` 的座標系——URDF 本身就註記了 `base_link` 到 `base_link_
inertia`（DH frame 0 的起點）之間有一個繞 Z 軸 180 度的旋轉（REP-103
對齊 vs. 控制器內部慣例）。是否還有其他偏移，靠
`scripts/verify_ur10e_analytic_ik.py` 對 RMPflow 正向運動學做批次數值驗證
才能確認，不能只靠讀 URDF 猜——這也是為什麼這個模組只提供「解出候選解」
的能力，呼叫端（`ur10e_placement_calculator.py`）要負責用驗證過的座標系
轉換去對接。
"""

import numpy as np

# UR10e 官方 DH 參數（Standard DH，d/a 對照 assets/rmpflow_config/ur10e_cue/
# ur10e.urdf 各關節 <origin> 的位移量：d1=shoulder_pan_joint 的 z 位移、
# a2=elbow_joint 的 x 位移、a3=wrist_1_joint 的 x 位移、d4=wrist_1_joint 的
# z 位移、d5=wrist_2_joint 的 y 位移量值、d6=wrist_3_joint 的 y 位移量值）。
_D1 = 0.1807
_A2 = -0.6127
_A3 = -0.57155
_D4 = 0.17415
_D5 = 0.11985
_D6 = 0.11655

# `forward_kinematics()`/`inverse_kinematics()` 用的 DH frame 0（base）/
# frame 6（末端）跟 Isaac Sim 的 base_link/wrist_3_link 之間，經
# `scripts/verify_ur10e_analytic_ik.py` 對 RMPflow 正向運動學做過 27 組
# 隨機關節角的批次數值驗證（2026-09-04），確認只差一個固定的繞 Z 軸 180
# 度旋轉（R_offset=diag(-1,-1,1)，平移偏移量為 0，跟 ur10e.urdf 裡「
# base_link 對齊 REP-103（X+ 朝前），控制器內部座標系 X+ 朝後，兩者差
# 180 度」的註記完全吻合，誤差在機器精度等級 1e-10~1e-11）。這個旋轉是
# 自身的反旋轉（轉兩次 180 度等於沒轉），所以同一個矩陣可以雙向使用。
_ISAAC_FRAME_ROTATION = np.diag([-1.0, -1.0, 1.0])


def dh_to_isaac_frame(position: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把 `forward_kinematics()` 算出的 DH frame 0/frame 6 位姿，轉成
    Isaac Sim base_link/wrist_3_link 座標系下的位姿。"""
    return _ISAAC_FRAME_ROTATION @ position, _ISAAC_FRAME_ROTATION @ rotation


def isaac_to_dh_frame(position: np.ndarray, rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """反方向：Isaac Sim base_link/wrist_3_link 座標系下的位姿，轉成
    `inverse_kinematics()` 需要的 DH frame 0/frame 6 慣例——跟
    `dh_to_isaac_frame()` 是同一個矩陣（見 `_ISAAC_FRAME_ROTATION` 說明：
    180 度旋轉自身互逆）。"""
    return _ISAAC_FRAME_ROTATION @ position, _ISAAC_FRAME_ROTATION @ rotation


_DH_D = (_D1, 0.0, 0.0, _D4, _D5, _D6)
_DH_A = (0.0, _A2, _A3, 0.0, 0.0, 0.0)
_DH_ALPHA = (np.pi / 2.0, 0.0, 0.0, np.pi / 2.0, -np.pi / 2.0, 0.0)


def _link_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """單一 DH link 的齊次轉換矩陣：Trans_z(d) · Rot_z(theta) · Trans_x(a) ·
    Rot_x(alpha)（standard DH；Trans_z 與 Rot_z 同軸可交換，跟 Rot_z(theta)·
    Trans_z(d)·Trans_x(a)·Rot_x(alpha) 等價）。"""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    t_d = np.eye(4)
    t_d[2, 3] = d
    rz = np.array([[ct, -st, 0.0, 0.0], [st, ct, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    t_a = np.eye(4)
    t_a[0, 3] = a
    rx = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, ca, -sa, 0.0], [0.0, sa, ca, 0.0], [0.0, 0.0, 0.0, 1.0]])
    return t_d @ rz @ t_a @ rx


def _joint_transform(joint_index: int, theta: float) -> np.ndarray:
    return _link_transform(theta, _DH_D[joint_index], _DH_A[joint_index], _DH_ALPHA[joint_index])


def forward_kinematics(joint_angles) -> tuple[np.ndarray, np.ndarray]:
    """六個關節角（[shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2,
    wrist_3]）→ (position, rotation_matrix)，DH frame 0（base）到 frame 6
    （末端）。"""
    t = np.eye(4)
    for i in range(6):
        t = t @ _joint_transform(i, float(joint_angles[i]))
    return t[:3, 3].copy(), t[:3, :3].copy()


def inverse_kinematics(target_position, target_rotation) -> list[np.ndarray]:
    """closed-form 逆向運動學，回傳最多 8 組候選解（每組是長度 6 的關節角
    陣列），DH frame 0/frame 6 慣例跟 `forward_kinematics()` 相同。幾何上
    不可達或落在奇異點本身（`sin(theta5)==0`，此時 theta6 數學上無定義）
    的分支會被跳過，不會出現在回傳的清單裡——呼叫端不需要另外過濾 NaN。

    演算法結構沿用 Ryan Keating (Johns Hopkins) 公開發表的 UR5/UR10
    closed-form IK 推導（theta1 由腕心投影角度解出、theta5 由末端 Z 軸
    在 frame 1 座標系下的分量解出、theta6/theta3/theta2/theta4 依序用
    相鄰 frame 之間的相對變換解出），只替換成本檔案開頭的 UR10e 官方 DH
    數值，並把數值穩定性相關的處理（clip 到 [-1,1] 再呼叫 arccos，避免
    浮點誤差讓引數略超出定義域時直接丟例外）明確加上。
    """
    target_position = np.asarray(target_position, dtype=float)
    target_rotation = np.asarray(target_rotation, dtype=float)
    t = np.eye(4)
    t[:3, :3] = target_rotation
    t[:3, 3] = target_position

    th = np.full((6, 8), np.nan)
    valid = [True] * 8

    # theta1：腕心（frame 5 原點，沿末端 -Z 方向退開 d6）投影到 XY 平面
    # 的角度，加減一個由 d4 決定的偏移角，對應「肩膀在左邊或右邊」兩種解。
    p05 = t @ np.array([0.0, 0.0, -_D6, 1.0]) - np.array([0.0, 0.0, 0.0, 1.0])
    psi = np.arctan2(p05[1], p05[0])
    r = float(np.hypot(p05[1], p05[0]))
    if r < abs(_D4) - 1e-9:
        return []  # 腕心投影半徑比 d4 還短，theta1 無實數解，目標完全不可達
    phi = np.arccos(np.clip(_D4 / r, -1.0, 1.0))
    th[0, 0:4] = np.pi / 2.0 + psi + phi
    th[0, 4:8] = np.pi / 2.0 + psi - phi

    # theta5
    for c in (0, 4):
        a1 = _joint_transform(0, th[0, c])
        t10 = np.linalg.inv(a1)
        t16 = t10 @ t
        arg = (t16[2, 3] - _D4) / _D6
        if abs(arg) > 1.0 + 1e-6:
            for k in range(c, c + 4):
                valid[k] = False
            continue
        arg = np.clip(arg, -1.0, 1.0)
        th[4, c:c + 2] = np.arccos(arg)
        th[4, c + 2:c + 4] = -np.arccos(arg)

    # theta6——sin(theta5)==0（正好落在手腕奇異點上）時數學上無定義，
    # 這個分支直接標記無效，不強行給一個任意值。
    for c in (0, 2, 4, 6):
        if not valid[c]:
            continue
        a1 = _joint_transform(0, th[0, c])
        t10 = np.linalg.inv(a1)
        t16 = np.linalg.inv(t10 @ t)
        s5 = np.sin(th[4, c])
        if abs(s5) < 1e-8:
            valid[c] = False
            valid[c + 1] = False
            continue
        th[5, c:c + 2] = np.arctan2(-t16[1, 2] / s5, t16[0, 2] / s5)

    # theta3：肘部用餘弦定理解，「肘部向上/向下」兩種解。
    for c in (0, 2, 4, 6):
        if not valid[c]:
            continue
        a1 = _joint_transform(0, th[0, c])
        t10 = np.linalg.inv(a1)
        t65 = _joint_transform(5, th[5, c])
        t54 = _joint_transform(4, th[4, c])
        t14 = (t10 @ t) @ np.linalg.inv(t54 @ t65)
        p13 = (t14 @ np.array([0.0, -_D4, 0.0, 1.0]) - np.array([0.0, 0.0, 0.0, 1.0]))[:3]
        p13_norm = float(np.linalg.norm(p13))
        arg = (p13_norm ** 2 - _A2 ** 2 - _A3 ** 2) / (2.0 * _A2 * _A3)
        if abs(arg) > 1.0 + 1e-6:
            valid[c] = False
            valid[c + 1] = False
            continue
        t3 = np.arccos(np.clip(arg, -1.0, 1.0))
        th[2, c] = t3
        th[2, c + 1] = -t3

    # theta2 / theta4
    for c in range(8):
        if not valid[c]:
            continue
        a1 = _joint_transform(0, th[0, c])
        t10 = np.linalg.inv(a1)
        t65 = np.linalg.inv(_joint_transform(5, th[5, c]))
        t54 = np.linalg.inv(_joint_transform(4, th[4, c]))
        t14 = (t10 @ t) @ t65 @ t54
        p13 = (t14 @ np.array([0.0, -_D4, 0.0, 1.0]) - np.array([0.0, 0.0, 0.0, 1.0]))[:3]
        p13_norm = float(np.linalg.norm(p13))

        th[1, c] = -np.arctan2(p13[1], -p13[0]) + np.arcsin(
            np.clip(_A3 * np.sin(th[2, c]) / p13_norm, -1.0, 1.0)
        )

        t32 = np.linalg.inv(_joint_transform(2, th[2, c]))
        t21 = np.linalg.inv(_joint_transform(1, th[1, c]))
        t34 = t32 @ t21 @ t14
        th[3, c] = np.arctan2(t34[1, 0], t34[0, 0])

    return [th[:, c].copy() for c in range(8) if valid[c] and not np.any(np.isnan(th[:, c]))]


def quat_wxyz_to_rotation_matrix(quat_wxyz) -> np.ndarray:
    """四元數（wxyz）轉 3x3 旋轉矩陣，標準公式。供呼叫端把
    `cue_pose_calculator` 算出來的 wrist_orientation（四元數）轉成
    `inverse_kinematics()`／`isaac_to_dh_frame()` 需要的旋轉矩陣。"""
    w, x, y, z = (float(c) for c in quat_wxyz)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ])


def wrist_singularity_margin(joint_angles) -> float:
    """UR 系列手臂手腕奇異點的標準判定條件是 `sin(theta5)==0`（見
    `inverse_kinematics()` theta6 那段：`sin(theta5)=0` 時 theta6 數學上
    無定義，官方文件也指出 wrist_2≈0 或 π 時 wrist_1/wrist_3 兩軸共平面）。
    回傳 `abs(sin(theta5))`（theta5＝關節陣列的第 5 個分量＝wrist_2_joint），
    數值越小代表越接近奇異點，0 代表正好卡在奇異點上；數值越大（上限 1，
    在 wrist_2=±90° 時取得）代表離奇異點越遠、操作性越好。
    """
    return float(abs(np.sin(joint_angles[4])))
