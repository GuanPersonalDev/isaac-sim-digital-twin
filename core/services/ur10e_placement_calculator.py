import logging

import numpy as np

from . import cue_pose_calculator, ur10e_analytic_ik

logger = logging.getLogger(__name__)

_BASE_STANDOFF_M = 0.5
"""UR10e 基座沿擊球方向水平反方向、從 wrist 目標退開的距離。

決策 4 原本假設 UR10e 固定基座位置（1.3m 可達距離夠大，不需要像
WAM7/UR3e 每一擊都重算基座）。實測發現：wrist 目標本身就在離母球
CUE_STICK_GRIP_TO_TIP=1.35m 遠的地方（球桿本身的長度），加上球檯各處
位置差異，固定基座（沿用 WAM7 的 TableRobotManager._ROBOT_OFFSET_
FROM_TABLE_CENTER=(1.5,0,0)）對某些母球位置離目標遠達 2.6m，遠超過
UR10e 的可達距離——純幾何上到不了，不是 RMPflow 參數能調出來的。

修正：改回 per-shot 重新計算基座位置（推翻決策 4 的固定基座假設），但
比 WAM7/UR3e 簡單很多——UR10e 靠 RMPflow 自己解完整 6-DOF IK，不需要
像 UR3e 那樣搜尋特定關節組合，也不需要像 WAM7 那樣算 base_yaw 關節
目標，只需要確保 wrist 目標落在舒適的可達範圍內（留一些操作餘裕，不要
让手臂伸到接近完全打直的邊界姿態）。做法：從 wrist 目標沿擊球方向的
水平反方向（握把那一側）退開 _BASE_STANDOFF_M，Z 維持跟桌面同高——
UR10e 可達距離 1.3m，0.5m 標準站距對任何 tilt_rad 案例都留有充足餘裕。
"""


def compute_base_position(
    wrist_position: tuple[float, float, float],
    direction_unit: tuple[float, float, float],
    table_z: float,
) -> tuple[float, float, float]:
    """由目標 wrist 位置（cue_pose_calculator.compute_tilted_wrist_pose()
    算出來的握把位置）與擊球方向反推 UR10e 基座位置。

    direction_unit: cue_pose_calculator.compute_tilted_direction() 的回傳值
    （單位向量，水平案例只有 X/Y 分量，高架橋案例可能帶 Z 分量——這裡只取
    水平分量決定基座退開方向，Z 固定跟桌面同高，不隨 tilt 變化，避免基座
    本身陷進地板或懸空）。
    """
    horizontal = np.array([direction_unit[0], direction_unit[1], 0.0])
    norm = float(np.linalg.norm(horizontal))
    if norm > 1e-9:
        horizontal = horizontal / norm

    base_x = wrist_position[0] - _BASE_STANDOFF_M * horizontal[0]
    base_y = wrist_position[1] - _BASE_STANDOFF_M * horizontal[1]
    base_z = table_z

    return (base_x, base_y, base_z)


_ROLL_SEARCH_STEP_DEG = 2.0
"""compute_roll_minimizing_reorientation() 搜尋 roll_rad 的角度解析度。

2026-09-03 除錯發現：UR10e 從 HOME 出發做 AIM 時，某些目標（尤其 flat
案例）需要接近 180 度的姿態翻轉，RMPflow 反應式求解在這種大角度翻轉
下容易卡在局部穩定點（實測：wrist_1_joint 短短幾個 waypoint 內衝過
-π 又折返，殘留誤差高達 0.1-0.6m）。

roll_rad（cue_pose_calculator.compute_tilted_wrist_pose() 的參數）是
球桿繞自身軸的冗餘自由度——不影響桿頭實際指向或位置，純粹是「同一個
指向，用哪個繞軸滾動角度表示」的選擇。固定用 roll_rad=0（_shortest_arc_
quat() 的預設「最短弧」慣例）算出來的姿態，對某些起始姿態剛好是最壞的
選擇（跟起始姿態接近正反面，需要接近 180 度翻轉）；改用能讓最終姿態
盡量接近目前姿態的 roll_rad，同一個指向可以只需要小得多的翻轉角度
（實測：flat 案例從 180 度降到 90 度）。這個函式搜尋讓翻轉角度最小化的
roll_rad。

⚠️ 2026-09-04 補充：只用「翻轉角度最小」當唯一準則，找到的 roll_rad
可能剛好逼近 UR10e 手腕的運動學奇異點（實測：flat 案例最終姿態卡在
wrist_2≈1.70 rad 附近的差動 IK 收尾，方向誤差穩定卡在 0.0294 rad
（容許值 0.02 rad）不再收斂，換算成桿尖偏移超過球半徑，STRIKE 完全打不
到球——這是 DLS 類疊代方法在奇異點附近精度/穩定性互相拖累的已知結構性
限制，不是加大步數/調參能解決的）。改成用 `core/services/
ur10e_analytic_ik.py` 的 UR10e closed-form 逆向運動學，直接算出每個候選
roll_rad 對應姿態的實際可達關節解，用 `wrist_singularity_margin()` 篩掉
太靠近奇異點的候選，只在「安全」的候選裡挑翻轉角度最小的——不再是盲目
只看翻轉角度，這樣選出來的目標姿態本身就遠離奇異點，RMPflow／差動 IK
收尾自然不會卡住（"避開奇異點"比"卡住後想辦法補救"更直接）。
"""

_WRIST_SINGULARITY_MARGIN_THRESHOLD = 0.3
"""`ur10e_analytic_ik.wrist_singularity_margin()` 的安全門檻——對應
wrist_2 離 0/π 至少約 17.5 度，留了不小的餘裕，不是卡在剛好過門檻的邊緣。
搜尋 roll_rad 時只在候選解的 margin 超過這個值才視為「安全」。"""


def _best_wrist_singularity_margin(
    wrist_position_in_base: np.ndarray, wrist_orientation_wxyz: np.ndarray,
) -> float:
    """回傳這個 wrist 目標位姿（機器人底座座標系）在 UR10e 幾何下，所有
    可達 closed-form 解析解分支裡 manipulability margin 最好的那個值——
    不管實際會落在哪個分支，只要存在一個安全的分支，就代表這個目標本身
    不是結構性卡死在奇異點上（目標本身不可達，回傳 0.0，視同不安全）。"""
    rotation_matrix = ur10e_analytic_ik.quat_wxyz_to_rotation_matrix(wrist_orientation_wxyz)
    dh_position, dh_rotation = ur10e_analytic_ik.isaac_to_dh_frame(
        np.asarray(wrist_position_in_base, dtype=float), rotation_matrix
    )
    solutions = ur10e_analytic_ik.inverse_kinematics(dh_position, dh_rotation)
    if not solutions:
        return 0.0
    return max(ur10e_analytic_ik.wrist_singularity_margin(solution) for solution in solutions)


def compute_roll_minimizing_reorientation(
    cue_ball: tuple[float, float],
    shot_angle_deg: float,
    table_z: float,
    ball_radius: float,
    position_offset: list[float],
    current_orientation: tuple[float, float, float, float],
    base_position: tuple[float, float, float],
) -> float:
    """搜尋讓 wrist 目標姿態跟 current_orientation 之間旋轉角度最小、且
    遠離 UR10e 手腕奇異點的 roll_rad（見模組說明的除錯發現）。純數學
    搜尋，不需要模擬，對每個候選 roll_rad 呼叫 cue_pose_calculator.
    compute_tilted_wrist_pose() 算出對應姿態，比較跟 current_orientation
    的四元數夾角，並用 ur10e_analytic_ik 算出實際可達關節解評估離奇異點
    多遠，在「安全」的候選裡挑夾角最小的。

    current_orientation: [qw, qx, qy, qz]，通常是手臂目前（AIM 前，例如
    RESET 後的 HOME）的實際末端朝向。
    base_position: 機器人底座目前的世界座標（`compute_base_position()`
    算出來的值）——wrist 目標本身不受 roll_rad 影響，呼叫端應該在搜尋前
    就先算好、傳進來，不用每個候選都重算一次。用來把 wrist 目標從世界
    座標轉成 `ur10e_analytic_ik` 需要的機器人底座座標系（機器人底座朝向
    固定是單位四元數，見 Ur10eSwingStrategy._BASE_ORIENTATION，所以只需要
    平移，不需要旋轉）。
    """
    current_orientation_arr = np.asarray(current_orientation, dtype=float)
    base_position_arr = np.asarray(base_position, dtype=float)

    best_roll_rad = 0.0
    best_dot = -1.0
    best_safe_roll_rad = None
    best_safe_dot = -1.0
    num_candidates = int(round(360.0 / _ROLL_SEARCH_STEP_DEG))
    for i in range(num_candidates):
        roll_rad = np.radians(i * _ROLL_SEARCH_STEP_DEG)
        wrist_position, orientation, tilt_rad, _ = cue_pose_calculator.compute_tilted_wrist_pose(
            cue_ball, shot_angle_deg, table_z, ball_radius, position_offset, roll_rad=roll_rad
        )
        if orientation is None:
            continue
        dot = float(np.clip(np.abs(np.dot(current_orientation_arr, orientation)), -1.0, 1.0))
        if dot > best_dot:
            best_dot = dot
            best_roll_rad = roll_rad

        margin = _best_wrist_singularity_margin(
            np.asarray(wrist_position, dtype=float) - base_position_arr, orientation
        )
        if margin >= _WRIST_SINGULARITY_MARGIN_THRESHOLD and dot > best_safe_dot:
            best_safe_dot = dot
            best_safe_roll_rad = roll_rad

    if best_safe_roll_rad is not None:
        return best_safe_roll_rad

    logger.warning(
        "compute_roll_minimizing_reorientation：360 度網格內找不到 wrist_singularity_margin "
        ">= %.2f 的安全候選，退回只看翻轉角度最小的舊準則（roll_rad=%.4f），"
        "這個 AIM 目標可能結構性地逼近奇異點，建議另外檢查 cue_ball=%s shot_angle_deg=%.2f",
        _WRIST_SINGULARITY_MARGIN_THRESHOLD, best_roll_rad, cue_ball, shot_angle_deg,
    )
    return best_roll_rad
