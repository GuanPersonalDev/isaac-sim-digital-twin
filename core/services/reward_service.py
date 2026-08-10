import math

from ..models.break_foul_result import BreakFoulResult
from ..models.shot_result import ShotResult
from .cue_ball_pocketed_penalty_calculator import (
    calculate_cue_ball_pocketed_penalty,
)
from .nine_ball_pocketed_bonus_calculator import (
    calculate_nine_ball_pocketed_bonus,
)
from .spread_score_calculator import spread_score_to_reward


_VALID_BREAK_FOUL_STATES = frozenset(
    {
        (0.0, False),
        (-0.5, False),
        (-1.5, True),
    }
)


def calculate_reward(
    shot_result: ShotResult,
    break_foul_result: BreakFoulResult,
) -> float:
    """整合散開分數、進袋獎懲與開球犯規，回傳訓練 reward。"""
    _validate_spread_score(shot_result.spread_score)
    _validate_break_foul_result(break_foul_result)

    if break_foul_result.should_reset:
        return break_foul_result.penalty

    cue_ball_penalty = calculate_cue_ball_pocketed_penalty(
        shot_result.cue_ball_pocketed
    )
    nine_ball_bonus = _calculate_nine_ball_bonus(
        shot_result,
        break_foul_result,
    )

    # ⚠️ 不是直接加 spread_score。原始分數的除數是整張桌，實際可達區間只有
    #    0.012~0.34，跟 ±3.5 的其他項差 25~175 倍（#123）。轉換的定義與理由見
    #    spread_score_calculator.spread_score_to_reward()。
    return (
        spread_score_to_reward(shot_result.spread_score)
        + cue_ball_penalty
        + break_foul_result.penalty
        + nine_ball_bonus
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
