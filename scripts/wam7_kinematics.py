"""
scripts/wam7_kinematics.py — WAM7 正向/逆向運動學（不跑物理模擬），供
scripts/search_ik_reachability.py 這類需要在短時間內測試大量候選姿態的
搜尋工具使用。運動學鏈直接從 assets/barrett_wam/wam7.urdf 的 <joint>
標籤讀出來的 origin xyz/rpy + axis（全部 revolute joint 的 axis 都是各自
局部座標系的 (0,0,1)，寫死在下面 `_CHAIN` 裡，不做 URDF XML 解析，避免
引入額外相依）。

見 docs/issue-180-reachability-analysis.md 第十四節：手動/半系統化試誤
（跑真實物理模擬，每組候選要 1-2 分鐘）已經證實窮舉不完，這支模組提供
不跑模擬、純數值的 FK/IK，讓大規模搜尋在合理時間內可行。

⚠️ 用 `_validate_against_known_constants()` 驗證過這條鏈跟真實 Isaac Sim
量出來的 `_LOCAL_TIP_RADIUS`／`_LOCAL_TIP_HEIGHT`／`CANONICAL_FLAT_
ORIENTATION` 一致，才能信任這支模組算出來的結果——任何修改運動學鏈定義
之後都要重新跑這個驗證。
"""

import math

import numpy as np

# (joint_name, origin_xyz, origin_rpy, is_revolute, (lower, upper) or None)
# 直接照 assets/barrett_wam/wam7.urdf 的 <joint> 標籤抄，順序是
# world -> base_link -> shoulder_yaw_link -> shoulder_pitch_link ->
# upper_arm_link -> forearm_link -> wrist_yaw_link -> wrist_pitch_link ->
# wrist_palm_link -> wrist_palm_stump_link。
_CHAIN = [
    ("wam_wam_fixed_joint", (0.0, 0.0, 1.0), (0.0, 0.0, 0.0), False, None),
    ("wam_base_yaw_joint", (0.0, 0.0, 0.346), (0.0, 0.0, 0.0), True, (-2.6, 2.6)),
    ("wam_shoulder_pitch_joint", (0.0, 0.0, 0.0), (-math.pi / 2, 0.0, 0.0), True, (-1.985, 1.985)),
    ("wam_shoulder_yaw_joint", (0.0, 0.0, 0.0), (math.pi / 2, 0.0, 0.0), True, (-2.8, 2.8)),
    ("wam_elbow_pitch_joint", (0.045, 0.0, 0.55), (-math.pi / 2, 0.0, 0.0), True, (-0.9, math.pi)),
    ("wam_wrist_yaw_joint", (-0.045, -0.3, 0.0), (math.pi / 2, 0.0, 0.0), True, (-4.55, 1.25)),
    ("wam_wrist_pitch_joint", (0.0, 0.0, 0.0), (-math.pi / 2, 0.0, 0.0), True, (-1.5707, 1.5707)),
    ("wam_palm_yaw_joint", (0.0, 0.0, 0.0), (math.pi / 2, 0.0, 0.0), True, (-3.0, 3.0)),
    ("wam_wrist_palm_stump_joint", (0.0, 0.0, 0.06), (0.0, 0.0, 0.0), False, None),
]

JOINT_NAMES = [name for name, *_ in _CHAIN if _[2]] if False else None  # placeholder, 見下方 JOINT_LIMITS
JOINT_LIMITS = [limits for _, _, _, is_rev, limits in _CHAIN if is_rev]
# [base_yaw, shoulder_pitch, shoulder_yaw, elbow_pitch, wrist_yaw, wrist_pitch, palm_yaw]
NUM_JOINTS = len(JOINT_LIMITS)


def _rpy_to_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    """URDF 慣例：R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。"""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _transform(xyz, rpy) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy_to_matrix(rpy)
    T[:3, 3] = xyz
    return T


def _rot_z(theta: float) -> np.ndarray:
    T = np.eye(4)
    c, s = math.cos(theta), math.sin(theta)
    T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return T


def _matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """旋轉矩陣轉四元數（wxyz），標準 Shepperd 方法。"""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    if q[0] < 0:
        q = -q
    return q


def forward_kinematics(joint_angles, base_position=(0.0, 0.0, 0.0), base_yaw_rad=0.0):
    """回傳 (position(3,), orientation_wxyz(4,))。`joint_angles` 是 7 個
    revolute joint 的角度，順序跟 `JOINT_LIMITS` 一致。`base_position`／
    `base_yaw_rad` 對應 `TableRobotManager`/`BarrettWamRobot.reposition()`
    設的機器人基座世界座標與（這條鏈本身不含的）額外基座朝向——目前專案
    裡機器人基座本身不會繞 Z 額外旋轉（`reposition()` 只設位置），所以
    `base_yaw_rad` 預設 0，保留參數只是為了未來擴充。
    """
    T = np.eye(4)
    T[:3, 3] = base_position
    if base_yaw_rad != 0.0:
        T = T @ _rot_z(base_yaw_rad)
    angle_idx = 0
    for _name, xyz, rpy, is_revolute, _limits in _CHAIN:
        T = T @ _transform(xyz, rpy)
        if is_revolute:
            T = T @ _rot_z(joint_angles[angle_idx])
            angle_idx += 1
    position = T[:3, 3]
    orientation = _matrix_to_quat_wxyz(T[:3, :3])
    return position, orientation


def _quat_error(current_wxyz: np.ndarray, target_wxyz: np.ndarray) -> np.ndarray:
    """跟 ArticulationAPIImpl._quat_error() 完全同一個公式：
    q_error = q_target * q_current⁻¹，w<0 時整體取反走最短路徑。"""
    cw, cx, cy, cz = current_wxyz
    tw, tx, ty, tz = target_wxyz
    current_conj = np.array([cw, -cx, -cy, -cz])
    aw, ax, ay, az = tw, tx, ty, tz
    bw, bx, by, bz = current_conj
    q_error = np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])
    if q_error[0] < 0:
        q_error = -q_error
    return q_error


def _numerical_jacobian(joint_angles, base_position, epsilon=1e-6) -> np.ndarray:
    """6×7 數值 Jacobian（有限差分），前 3 列是位置對關節角的偏微分，後 3
    列是「小角度姿態誤差」對關節角的偏微分（跟 ArticulationAPIImpl 的
    _compute_pose_tracking_twist() 用同一種線性化方式，差動量測不是解析
    推導，換取實作簡單可靠）。"""
    pos0, quat0 = forward_kinematics(joint_angles, base_position)
    J = np.zeros((6, NUM_JOINTS))
    for i in range(NUM_JOINTS):
        perturbed = list(joint_angles)
        perturbed[i] += epsilon
        pos1, quat1 = forward_kinematics(perturbed, base_position)
        J[:3, i] = (pos1 - pos0) / epsilon
        q_err = _quat_error(quat0, quat1)
        J[3:, i] = (2.0 * q_err[1:]) / epsilon
    return J


def solve_ik(
    target_position,
    target_orientation,
    initial_joints,
    base_position=(0.0, 0.0, 0.0),
    max_iters=300,
    damping=0.05,
    step_size=0.5,
    position_tolerance=0.003,
    orientation_tolerance=0.015,
    clamp_to_limits=True,
):
    """跟 ArticulationAPIImpl._step_motion() 同一種阻尼最小二乘法，但這裡是
    純數值迭代（不跑物理），每次迭代後把關節角 clamp 回 JOINT_LIMITS
    （模擬 PhysX 關節硬限位會擋住超出範圍的移動——這個簡化不是精確重現
    PhysX 動力學，但足夠用來快速篩「這組起點附近有沒有機會收斂」）。

    回傳 (final_joints, converged: bool, final_pos_error, final_orient_error)。
    """
    q = np.array(initial_joints, dtype=float)
    target_position = np.asarray(target_position, dtype=float)
    target_orientation = np.asarray(target_orientation, dtype=float)
    lowers = np.array([lo for lo, hi in JOINT_LIMITS])
    uppers = np.array([hi for lo, hi in JOINT_LIMITS])

    pos_error_norm = orient_error_norm = float("inf")
    for _ in range(max_iters):
        pos, quat = forward_kinematics(q, base_position)
        pos_error = target_position - pos
        q_err = _quat_error(quat, target_orientation)
        orient_error = 2.0 * q_err[1:]
        pos_error_norm = float(np.linalg.norm(pos_error))
        orient_error_norm = float(np.linalg.norm(orient_error))
        if pos_error_norm < position_tolerance and orient_error_norm < orientation_tolerance:
            return q, True, pos_error_norm, orient_error_norm

        twist = np.concatenate([pos_error, orient_error])
        J = _numerical_jacobian(q, base_position)
        JJt = J @ J.T + (damping ** 2) * np.eye(6)
        qdot = J.T @ np.linalg.solve(JJt, twist)
        q = q + step_size * qdot
        if clamp_to_limits:
            q = np.clip(q, lowers, uppers)

    return q, False, pos_error_norm, orient_error_norm


def _validate_against_known_constants() -> None:
    """核對這條鏈的 FK 算出來的值，跟真實 Isaac Sim 量出的
    `_LOCAL_TIP_RADIUS`(0.38511)／`_LOCAL_TIP_HEIGHT`(0.78731)／
    `CANONICAL_FLAT_ORIENTATION`((0,0.68216,0.73120,0)) 是否一致。任何
    修改 `_CHAIN` 之後都要重新跑這個函式確認沒有壞掉。"""
    canonical_rest_joints = (1.9, 0.0, 1.8, 0.0, -0.5585, 1.5010)
    joint_angles = [0.0, *canonical_rest_joints]  # base_yaw=0
    position, orientation = forward_kinematics(joint_angles, base_position=(0.0, 0.0, 0.0))

    horizontal_dist = math.hypot(position[0], position[1])
    height = position[2]

    expected_radius = 0.38511
    expected_height = 0.78731
    expected_orientation = np.array([0.0, 0.68216, 0.73120, 0.0])

    radius_diff = abs(horizontal_dist - expected_radius)
    height_diff = abs(height - expected_height)
    # 四元數可能整體差一個符號（q 和 -q 代表同一個旋轉），兩種都要比對。
    orient_diff = min(
        float(np.linalg.norm(orientation - expected_orientation)),
        float(np.linalg.norm(orientation + expected_orientation)),
    )

    print(f"[validate] FK position={position.tolist()}  horizontal_dist={horizontal_dist:.5f} (期望 {expected_radius}, 差 {radius_diff:.5f})")
    print(f"[validate] FK height={height:.5f} (期望 {expected_height}, 差 {height_diff:.5f})")
    print(f"[validate] FK orientation={orientation.tolist()} (期望 {expected_orientation.tolist()}, 差 {orient_diff:.5f})")

    ok = radius_diff < 0.002 and height_diff < 0.002 and orient_diff < 0.01
    print(f"[validate] {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise AssertionError("FK 驗證失敗，_CHAIN 定義可能有誤，不能拿去做搜尋")


if __name__ == "__main__":
    _validate_against_known_constants()
