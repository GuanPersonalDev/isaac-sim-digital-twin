import math

import pytest

from core.models.break_foul_result import BreakFoulResult
from core.models.shot_result import ShotResult
from core.services.reward_service import calculate_reward
from core.services.spread_score_calculator import spread_score_to_reward

# 期望值一律用 spread_score_to_reward() 表示，不寫死數字：#123 明文要求
# SPREAD_REF 開跑後要用實際 rollout 重新量一次，寫死的話那次重新校準會變成
# 「順手改測試」而不是「改定義」。
_SPREAD_04 = spread_score_to_reward(0.4)


@pytest.fixture
def final_ball_positions() -> list[list[float]]:
    return [[0.0, 0.0] for _ in range(10)]


@pytest.fixture
def legal_shot_result(
    final_ball_positions: list[list[float]],
) -> ShotResult:
    return ShotResult(
        final_ball_positions=final_ball_positions,
        cue_ball_pocketed=False,
        nine_ball_pocketed=False,
        spread_score=0.4,
    )


@pytest.fixture
def nine_ball_pocketed_shot_result(
    final_ball_positions: list[list[float]],
) -> ShotResult:
    return ShotResult(
        final_ball_positions=final_ball_positions,
        cue_ball_pocketed=False,
        nine_ball_pocketed=True,
        spread_score=0.4,
    )


@pytest.fixture
def cue_ball_pocketed_shot_result(
    final_ball_positions: list[list[float]],
) -> ShotResult:
    return ShotResult(
        final_ball_positions=final_ball_positions,
        cue_ball_pocketed=True,
        nine_ball_pocketed=False,
        spread_score=0.4,
    )


@pytest.fixture
def cue_and_nine_ball_pocketed_shot_result(
    final_ball_positions: list[list[float]],
) -> ShotResult:
    return ShotResult(
        final_ball_positions=final_ball_positions,
        cue_ball_pocketed=True,
        nine_ball_pocketed=True,
        spread_score=0.4,
    )


@pytest.fixture
def all_positive_components_shot_result(
    final_ball_positions: list[list[float]],
) -> ShotResult:
    return ShotResult(
        final_ball_positions=final_ball_positions,
        cue_ball_pocketed=False,
        nine_ball_pocketed=True,
        spread_score=1.0,
    )


@pytest.fixture(params=[0.0, 1.0])
def boundary_spread_shot_result(
    request,
    final_ball_positions: list[list[float]],
) -> ShotResult:
    return ShotResult(
        final_ball_positions=final_ball_positions,
        cue_ball_pocketed=False,
        nine_ball_pocketed=False,
        spread_score=request.param,
    )


@pytest.fixture(params=[-0.1, 1.1, math.nan, math.inf, -math.inf])
def invalid_spread_shot_result(
    request,
    final_ball_positions: list[list[float]],
) -> ShotResult:
    return ShotResult(
        final_ball_positions=final_ball_positions,
        cue_ball_pocketed=False,
        nine_ball_pocketed=False,
        spread_score=request.param,
    )


@pytest.fixture
def no_break_foul_result() -> BreakFoulResult:
    return BreakFoulResult(penalty=0.0, should_reset=False)


@pytest.fixture
def insufficient_rail_foul_result() -> BreakFoulResult:
    return BreakFoulResult(penalty=-0.5, should_reset=False)


@pytest.fixture
def first_contact_foul_result() -> BreakFoulResult:
    return BreakFoulResult(penalty=-1.5, should_reset=True)


@pytest.fixture(
    params=[
        (0.0, True),
        (-0.5, True),
        (-1.5, False),
        (-2.0, False),
    ]
)
def invalid_break_foul_result(request) -> BreakFoulResult:
    penalty, should_reset = request.param
    return BreakFoulResult(
        penalty=penalty,
        should_reset=should_reset,
    )


class TestCalculateReward:
    def test_returns_spread_score_for_legal_shot(
        self,
        legal_shot_result: ShotResult,
        no_break_foul_result: BreakFoulResult,
    ):
        reward = calculate_reward(legal_shot_result, no_break_foul_result)

        assert reward == pytest.approx(_SPREAD_04)

    def test_adds_nine_ball_bonus_for_foul_free_shot(
        self,
        nine_ball_pocketed_shot_result: ShotResult,
        no_break_foul_result: BreakFoulResult,
    ):
        reward = calculate_reward(
            nine_ball_pocketed_shot_result,
            no_break_foul_result,
        )

        assert reward == pytest.approx(_SPREAD_04 + 3.0)

    def test_applies_cue_ball_penalty(
        self,
        cue_ball_pocketed_shot_result: ShotResult,
        no_break_foul_result: BreakFoulResult,
    ):
        reward = calculate_reward(
            cue_ball_pocketed_shot_result,
            no_break_foul_result,
        )

        assert reward == pytest.approx(_SPREAD_04 - 3.5)

    def test_cue_ball_pocketed_cancels_nine_ball_bonus(
        self,
        cue_and_nine_ball_pocketed_shot_result: ShotResult,
        no_break_foul_result: BreakFoulResult,
    ):
        reward = calculate_reward(
            cue_and_nine_ball_pocketed_shot_result,
            no_break_foul_result,
        )

        assert reward == pytest.approx(_SPREAD_04 - 3.5)

    def test_applies_insufficient_rail_penalty(
        self,
        legal_shot_result: ShotResult,
        insufficient_rail_foul_result: BreakFoulResult,
    ):
        reward = calculate_reward(
            legal_shot_result,
            insufficient_rail_foul_result,
        )

        assert reward == pytest.approx(_SPREAD_04 - 0.5)

    def test_accumulates_cue_ball_and_rail_penalties(
        self,
        cue_ball_pocketed_shot_result: ShotResult,
        insufficient_rail_foul_result: BreakFoulResult,
    ):
        reward = calculate_reward(
            cue_ball_pocketed_shot_result,
            insufficient_rail_foul_result,
        )

        assert reward == pytest.approx(_SPREAD_04 - 3.5 - 0.5)

    def test_break_foul_cancels_nine_ball_bonus(
        self,
        nine_ball_pocketed_shot_result: ShotResult,
        insufficient_rail_foul_result: BreakFoulResult,
    ):
        reward = calculate_reward(
            nine_ball_pocketed_shot_result,
            insufficient_rail_foul_result,
        )

        assert reward == pytest.approx(_SPREAD_04 - 0.5)

    def test_first_contact_foul_returns_only_terminal_penalty(
        self,
        all_positive_components_shot_result: ShotResult,
        first_contact_foul_result: BreakFoulResult,
    ):
        reward = calculate_reward(
            all_positive_components_shot_result,
            first_contact_foul_result,
        )

        assert reward == pytest.approx(-1.5)

    def test_accepts_spread_score_boundaries(
        self,
        boundary_spread_shot_result: ShotResult,
        no_break_foul_result: BreakFoulResult,
    ):
        reward = calculate_reward(
            boundary_spread_shot_result,
            no_break_foul_result,
        )

        assert reward == pytest.approx(
            spread_score_to_reward(boundary_spread_shot_result.spread_score)
        )

    def test_rejects_invalid_spread_score(
        self,
        invalid_spread_shot_result: ShotResult,
        no_break_foul_result: BreakFoulResult,
    ):
        with pytest.raises(ValueError):
            calculate_reward(
                invalid_spread_shot_result,
                no_break_foul_result,
            )

    def test_rejects_invalid_break_foul_result(
        self,
        legal_shot_result: ShotResult,
        invalid_break_foul_result: BreakFoulResult,
    ):
        with pytest.raises(ValueError):
            calculate_reward(
                legal_shot_result,
                invalid_break_foul_result,
            )
