import math

import pytest

from core.models.break_foul_result import BreakFoulResult
from core.models.shot_result import ShotResult
from core.services.aim_shaping_calculator import (
    AIM_REFERENCE_GAP,
    AIM_REWARD_SCALE,
)
from core.services.break_foul_evaluator import (
    FIRST_CONTACT_FOUL_PENALTY,
    NO_CONTACT_FOUL_PENALTY,
)
from core.services.reward_service import calculate_reward
from core.services.spread_score_calculator import spread_score_to_reward

# 期望值一律用 spread_score_to_reward() 表示，不寫死數字：#123 已依 RunPod
# rollout 校準 SPREAD_REF，未來若再校準，測試應跟著正式定義而不是複製常數。
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


class TestAimShaping:
    """#124：dense shaping 必須跨過 `should_reset` 分支。"""

    def test_defaults_to_no_shaping(
        self,
        legal_shot_result: ShotResult,
        no_break_foul_result: BreakFoulResult,
    ):
        """沒帶 closest_approach 的舊呼叫端維持改動前的數值。"""
        assert legal_shot_result.closest_approach == math.inf

        reward = calculate_reward(legal_shot_result, no_break_foul_result)

        assert reward == pytest.approx(_SPREAD_04)

    def test_shaping_is_paid_on_first_contact_foul(
        self,
        all_positive_components_shot_result: ShotResult,
        first_contact_foul_result: BreakFoulResult,
    ):
        """碰到錯球（should_reset）也要拿塑形。

        這是整個改動的重點：訓練初期壓倒性多數的 episode 都走這個分支，
        塑形若被它吃掉就等於沒加。
        """
        all_positive_components_shot_result.closest_approach = 0.0

        reward = calculate_reward(
            all_positive_components_shot_result,
            first_contact_foul_result,
        )

        assert reward == pytest.approx(-1.5 + AIM_REWARD_SCALE)

    def test_shaping_is_paid_on_no_contact_foul(
        self,
        all_positive_components_shot_result: ShotResult,
    ):
        """整局沒碰到球也要拿塑形——不然這一項對「亂打」完全沒有作用力。"""
        all_positive_components_shot_result.closest_approach = 0.0
        no_contact = BreakFoulResult(
            penalty=NO_CONTACT_FOUL_PENALTY, should_reset=True
        )

        reward = calculate_reward(all_positive_components_shot_result, no_contact)

        assert reward == pytest.approx(NO_CONTACT_FOUL_PENALTY + AIM_REWARD_SCALE)

    def test_shaping_adds_to_a_legal_shot(
        self,
        legal_shot_result: ShotResult,
        no_break_foul_result: BreakFoulResult,
    ):
        legal_shot_result.closest_approach = 0.0

        reward = calculate_reward(legal_shot_result, no_break_foul_result)

        assert reward == pytest.approx(_SPREAD_04 + AIM_REWARD_SCALE)

    def test_shaping_is_monotone_in_the_approach(
        self,
        all_positive_components_shot_result: ShotResult,
        first_contact_foul_result: BreakFoulResult,
    ):
        rewards = []
        for gap in (0.0, 0.25, 0.5, 1.0, AIM_REFERENCE_GAP):
            all_positive_components_shot_result.closest_approach = gap
            rewards.append(
                calculate_reward(
                    all_positive_components_shot_result,
                    first_contact_foul_result,
                )
            )

        assert rewards == sorted(rewards, reverse=True)

    def test_missing_can_never_beat_contacting(
        self,
        all_positive_components_shot_result: ShotResult,
    ):
        """塑形不得反轉犯規排序：瞄最準的「沒碰到」仍輸給最爛的「碰到錯球」。"""
        all_positive_components_shot_result.closest_approach = 0.0
        best_miss = calculate_reward(
            all_positive_components_shot_result,
            BreakFoulResult(penalty=NO_CONTACT_FOUL_PENALTY, should_reset=True),
        )

        all_positive_components_shot_result.closest_approach = AIM_REFERENCE_GAP
        worst_wrong_contact = calculate_reward(
            all_positive_components_shot_result,
            BreakFoulResult(
                penalty=FIRST_CONTACT_FOUL_PENALTY, should_reset=True
            ),
        )

        assert best_miss < worst_wrong_contact
