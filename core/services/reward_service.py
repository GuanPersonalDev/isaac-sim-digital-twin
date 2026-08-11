import math

from ..models.break_foul_result import BreakFoulResult
from ..models.shot_result import ShotResult
from .aim_shaping_calculator import closest_approach_to_reward
from .break_foul_evaluator import (
    FIRST_CONTACT_FOUL_PENALTY,
    INSUFFICIENT_RAIL_CONTACT_PENALTY,
    NO_CONTACT_FOUL_PENALTY,
)
from .cue_ball_pocketed_penalty_calculator import (
    calculate_cue_ball_pocketed_penalty,
)
from .nine_ball_pocketed_bonus_calculator import (
    calculate_nine_ball_pocketed_bonus,
)
from .spread_score_calculator import spread_score_to_reward


# 引用 break_foul_evaluator 的常數而不是重打數字：兩邊各寫一份的話，那裡加
# 一種犯規（如 #124 的 NO_CONTACT）這裡就會把它判成「不支援的狀態」並拋錯，
# 而錯誤訊息完全不會提到真正的原因。
_VALID_BREAK_FOUL_STATES = frozenset(
    {
        (0.0, False),
        (INSUFFICIENT_RAIL_CONTACT_PENALTY, False),
        (FIRST_CONTACT_FOUL_PENALTY, True),
        (NO_CONTACT_FOUL_PENALTY, True),
    }
)


def calculate_reward(
    shot_result: ShotResult,
    break_foul_result: BreakFoulResult,
) -> float:
    """整合散開分數、進袋獎懲與開球犯規，回傳訓練 reward。"""
    _validate_spread_score(shot_result.spread_score)
    _validate_break_foul_result(break_foul_result)

    # ⚠️ dense shaping 必須**跨過** should_reset 分支（#124）。犯規重置涵蓋了
    #    「沒碰到球」與「碰到錯球」兩種情形，而那正是訓練初期的壓倒性多數——
    #    塑形若跟其他項一樣被這個分支吃掉，就等於只在已經打好的那 4.5% 局
    #    給塑形，完全沒有把 policy 拉出平原的作用。
    aim_reward = closest_approach_to_reward(shot_result.closest_approach)

    if break_foul_result.should_reset:
        return break_foul_result.penalty + aim_reward

    cue_ball_penalty = calculate_cue_ball_pocketed_penalty(
        shot_result.cue_ball_pocketed
    )
    nine_ball_bonus = _calculate_nine_ball_bonus(
        shot_result,
        break_foul_result,
    )

    # ⚠️ 不是直接加 spread_score。RunPod 實測控制式開球相對 rack 的原始增量
    #    平均只有約 0.030，跟 0.5~3.5 的其他項差 17~116 倍（#123）。轉換的
    #    定義與理由見 spread_score_calculator.spread_score_to_reward()。
    return (
        spread_score_to_reward(shot_result.spread_score)
        + cue_ball_penalty
        + break_foul_result.penalty
        + nine_ball_bonus
        + aim_reward
    )


def _validate_spread_score(spread_score: float) -> None:
    if not math.isfinite(spread_score) or not 0.0 <= spread_score <= 1.0:
        raise ValueError("spread_score must be finite and between 0.0 and 1.0")


def _validate_break_foul_result(
    break_foul_result: BreakFoulResult,
) -> None:
    state = (
        break_foul_result.penalty,
        break_foul_result.should_reset,
    )
    if state not in _VALID_BREAK_FOUL_STATES:
        raise ValueError("break_foul_result contains an unsupported state")


def _calculate_nine_ball_bonus(
    shot_result: ShotResult,
    break_foul_result: BreakFoulResult,
) -> float:
    has_foul = (
        shot_result.cue_ball_pocketed
        or break_foul_result.penalty < 0.0
    )
    if has_foul:
        return 0.0

    return calculate_nine_ball_pocketed_bonus(
        shot_result.nine_ball_pocketed,
        shot_result.cue_ball_pocketed,
    )
