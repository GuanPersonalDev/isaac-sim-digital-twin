"""揮桿軌跡計算：從瞄準姿態＋`Action.cue_ball_speed` 算出「後擺→接觸」兩個
Cartesian waypoint（供 `DemoTableOrchestrator._execute_strike()` 使用）。

見 Issue #181。動量傳遞公式與三個物理常數跟 `core/models/action_bounds.py`
的 `CUE_BALL_SPEED` 上界推導同一套（球桿 0.5kg、母球 0.163kg、恢復係數
e=0.75），這裡是單一事實來源。
"""

import numpy as np

from ..models.pose_waypoint import PoseWaypoint

CUE_STICK_MASS_KG = 0.5
CUE_BALL_MASS_KG = 0.163
RESTITUTION_COEFFICIENT = 0.75

DEFAULT_BACKSWING_DISTANCE_M = 0.15
"""後擺距離，沿桿身方向反向退開的距離。初始值取自
docs/issue-180-reachability-analysis.md 第九節「後擺走廊 L≈0.1~0.15m」的
粗估——那個數字原本只是為了估算臂展需求的假設，未經真正揮桿動力學驗證，
需要靠 scripts/verify_swing_trajectory.py 實測校準（後擺方向若在高架橋
姿態下會斜向後上方，還要確認不會撞到前臂本體）。"""

_FOLLOW_THROUGH_COEFFICIENT = 0.02
_FOLLOW_THROUGH_MIN_M = 0.01
_FOLLOW_THROUGH_MAX_M = 0.06
"""隨揮距離的係數/上下限，見 compute_follow_through_distance() 說明。同樣是
未經實測校準的初始值。"""


def compute_required_tip_speed(cue_ball_speed: float) -> float:
    """`v_ball = v_cue * (1+e) * M/(M+m)`（action_bounds.py CUE_BALL_SPEED
    上界推導公式）的反函式：由目標母球初速反推桿尖接觸瞬間需要的速度。"""
    momentum_ratio = (1.0 + RESTITUTION_COEFFICIENT) * CUE_STICK_MASS_KG / (
        CUE_STICK_MASS_KG + CUE_BALL_MASS_KG
    )
    return cue_ball_speed / momentum_ratio


def compute_follow_through_distance(required_tip_speed: float) -> float:
    """隨揮距離不能是常數：`_compute_pose_tracking_twist()` 的 P 控制器項
    會疊加在 feedforward 速度之上（不是位置誤差趨近 0 才生效），固定距離
    會讓低速擊球被 P 項殘留貢獻系統性超速。用跟目標速度成正比、並裁進
    [_FOLLOW_THROUGH_MIN_M, _FOLLOW_THROUGH_MAX_M] 的距離，讓 P 項貢獻
    （`POSITION_GAIN × distance`）相對目標速度維持在驗收容許值內。
    係數需要靠 scripts/verify_swing_trajectory.py 實測校準，這裡只是
    起始值。"""
    return float(
        np.clip(
            _FOLLOW_THROUGH_COEFFICIENT * required_tip_speed,
            _FOLLOW_THROUGH_MIN_M,
            _FOLLOW_THROUGH_MAX_M,
        )
    )


def compute_backswing_position(
    contact_position: np.ndarray, direction_unit: np.ndarray, backswing_distance: float
) -> np.ndarray:
    """後擺位置：沿桿身方向反向退開 backswing_distance，姿態不變。"""
    return contact_position - backswing_distance * direction_unit


def compute_swing_waypoints(
    contact_position: list[float],
    contact_orientation: list[float],
    direction_unit: list[float],
    cue_ball_speed: float,
    backswing_distance: float = DEFAULT_BACKSWING_DISTANCE_M,
) -> list[PoseWaypoint]:
    """回傳 [後擺 waypoint(v=0), 隨揮終點 waypoint(v=目標桿尖速度)]。

    第二個 waypoint 的 position 是 `contact_position` 前方
    `compute_follow_through_distance()` 距離的隨揮終點，**不是**
    `contact_position` 本身——這是刻意的：P 控制器在球心處還有殘留位置
    誤差貢獻，把終點設在球心之後，桿尖通過球心當下才會仍在接近目標速度
    全速前進，而不是在球心就已經開始減速（對應真人打撞球的隨揮技巧）。

    兩個 waypoint 的 orientation 相同：揮桿全程桿身指向不變，只有沿桿身
    軸方向的位置變化（見 docs/WAM_IK_implementation_and_verification.md
    2.1 節「後擺/擊球姿態同一個朝向」）。
    """
    contact = np.array(contact_position)
    direction = np.array(direction_unit)
    direction = direction / np.linalg.norm(direction)

    required_tip_speed = compute_required_tip_speed(cue_ball_speed)
    follow_through_distance = compute_follow_through_distance(required_tip_speed)

    backswing_position = compute_backswing_position(contact, direction, backswing_distance)
    follow_through_position = contact + follow_through_distance * direction
    contact_linear_velocity = (required_tip_speed * direction).tolist()

    return [
        PoseWaypoint(
            position=backswing_position.tolist(),
            orientation=list(contact_orientation),
            linear_velocity=[0.0, 0.0, 0.0],
        ),
        PoseWaypoint(
            position=follow_through_position.tolist(),
            orientation=list(contact_orientation),
            linear_velocity=contact_linear_velocity,
        ),
    ]
