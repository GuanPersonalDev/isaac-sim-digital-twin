import numpy as np

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
