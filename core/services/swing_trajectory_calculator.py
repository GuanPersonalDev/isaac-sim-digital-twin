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
姿態下會斜向後上方，還要確認不會撞到前臂本體）。

⚠️ 2026-09-01：正式的高架橋路徑（`DemoTableOrchestrator._execute_strike()`
的 `tilt_rad>1e-6` 分支）已改用
`cue_pose_calculator.lookup_backswing_distance_m()`——用 IK 可達邊界法
（`scripts/search_backswing_distance_ik.py`）對每個 Kitchen 案例反推出的
後擺距離，遠大於這個常數（0.34~0.35m vs 0.15m），跟關節實際能提供的加速
能力掛鉤，不再是這個粗估常數。這個常數現在只服務 flat 案例（第十八節
「待處理 B」明文決定不套用這套統一）與離線工具
（`scripts/search_roll_for_full_swing.py`／`scripts/search_backswing_
distance.py`），不要刪除。"""

_FOLLOW_THROUGH_COEFFICIENT = 0.02
_FOLLOW_THROUGH_MIN_M = 0.01
_FOLLOW_THROUGH_MAX_M = 0.06
"""隨揮距離的係數/上下限，見 compute_follow_through_distance() 說明。同樣是
未經實測校準的初始值。"""


def compute_required_tip_speed(cue_ball_speed: float) -> float:
    """`v_ball = v_cue * (1+e) * M/(M+m)`（action_bounds.py CUE_BALL_SPEED
    上界推導公式）的反函式：由目標母球初速反推桿尖接觸瞬間需要的速度。"""
    momentum_ratio = (1 + RESTITUTION_COEFFICIENT) * CUE_STICK_MASS_KG / (CUE_STICK_MASS_KG + CUE_BALL_MASS_KG)
    return cue_ball_speed / momentum_ratio


CUE_SLIDE_MEASURED_SPEED_RATIO = 1.7333
"""UR10e 線性滑軌推桿機構實測的「母球初速 ÷ 指令桿尖速度」端到端比值。

`compute_required_tip_speed()` 那套 `(1+e)·M/(M+m)`＝1.3197 的理論比值
**不適用於這個機構**，實測（`scripts/test_ur10e_table_flat.py`：指令桿尖
速度 1.5116 m/s → 母球初速 2.6200 m/s）差了 31%，兩個原因疊加：

1. 公式假設球桿是質量 0.5kg 的自由物體，但滑軌關節的 drive stiffness
   是 1e5，撞擊瞬間球桿是被驅動器硬撐住的，等效質量遠大於 0.5kg
   （M→∞ 時比值上限就是 1+e=1.75）。
2. `Ur10eCueSlideController` 的 quintic 邊界條件是「固定後擺距離、終點
   速度＝指令速度」，行程中段速度必然高於終點速度（平均速度要等於終點
   速度，中段就得超過），而母球實際是在 q≈0 前一點就被打到，吃到的是
   還沒降回終點值的中段速度。

兩者都跟指令速度成正比（quintic 的正規化剖面與 v1 無關，見
`Ur10eCueSlideController._step_backswing()` 的 `T=|q0|/v1`），所以用單一
線性比值校準即可，不需要拆開建模。這是端到端量測結果，改動後擺距離、
drive 增益或 quintic 邊界條件之後都要重新量。"""


def compute_required_tip_speed_for_cue_slide(cue_ball_speed: float) -> float:
    """UR10e 線性滑軌推桿專用：由目標母球初速反推該下達的桿尖速度指令。

    跟 `compute_required_tip_speed()` 的差別只在改用實測的端到端比值
    `CUE_SLIDE_MEASURED_SPEED_RATIO`（見該常數說明），不是動量傳遞理論
    值——WAM7／UR3e 兩套揮桿機構不適用這個比值，仍走原本的函式。
    """
    return cue_ball_speed / CUE_SLIDE_MEASURED_SPEED_RATIO


def compute_follow_through_distance(required_tip_speed: float) -> float:
    """隨揮距離不能是常數：`_compute_pose_tracking_twist()` 的 P 控制器項
    會疊加在 feedforward 速度之上（不是位置誤差趨近 0 才生效），固定距離
    會讓低速擊球被 P 項殘留貢獻系統性超速。用跟目標速度成正比、並裁進
    [_FOLLOW_THROUGH_MIN_M, _FOLLOW_THROUGH_MAX_M] 的距離，讓 P 項貢獻
    （`POSITION_GAIN × distance`）相對目標速度維持在驗收容許值內。
    係數需要靠 scripts/verify_swing_trajectory.py 實測校準，這裡只是
    起始值。"""
    distance = _FOLLOW_THROUGH_COEFFICIENT * required_tip_speed
    return float(np.clip(distance, _FOLLOW_THROUGH_MIN_M, _FOLLOW_THROUGH_MAX_M))


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

    做法：`direction_unit` 正規化 → `compute_required_tip_speed()` 算目標
    桿尖速度 → `compute_follow_through_distance()` 算隨揮距離 →
    `compute_backswing_position()` 算後擺點 → 隨揮終點 =
    `contact_position + follow_through_distance * direction` → 兩個
    waypoint 的 linear_velocity 分別是 `[0,0,0]` 與
    `required_tip_speed * direction`。
    """
    direction = np.array(direction_unit)
    direction = direction / np.linalg.norm(direction_unit)

    required_tip_speed = compute_required_tip_speed(cue_ball_speed)
    follow_through_distance = compute_follow_through_distance(required_tip_speed)

    contact_position = np.array(contact_position)
    backswing_position = compute_backswing_position(contact_position, direction, backswing_distance)
    follow_through_position = contact_position + follow_through_distance * direction

    waypoint_backswing = PoseWaypoint(position=backswing_position.tolist(), orientation=contact_orientation, linear_velocity=[0,0,0])
    waypoint_follow_through = PoseWaypoint(position=follow_through_position.tolist(), orientation=contact_orientation, linear_velocity=(required_tip_speed * direction).tolist())

    return [waypoint_backswing, waypoint_follow_through]
