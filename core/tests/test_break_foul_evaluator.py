from dataclasses import FrozenInstanceError

import pytest

from core.models.break_foul_result import BreakFoulResult
from core.services.break_foul_evaluator import (
    FIRST_CONTACT_FOUL_PENALTY,
    INSUFFICIENT_RAIL_CONTACT_PENALTY,
    MIN_RAIL_CONTACTED_OBJECT_BALLS,
    NO_CONTACT_FOUL_PENALTY,
    evaluate_break_foul,
)


@pytest.fixture
def legal_first_contact_ball_id() -> int:
    return 1


@pytest.fixture
def wrong_first_contact_ball_id() -> int:
    return 2


@pytest.fixture
def no_first_contact_ball_id() -> None:
    return None


@pytest.fixture
def no_pocketed_object_ball_ids() -> set[int]:
    return set()


@pytest.fixture
def one_pocketed_object_ball_id() -> set[int]:
    return {2}


@pytest.fixture
def no_rail_contacted_object_ball_ids() -> set[int]:
    return set()


@pytest.fixture
def three_rail_contacted_object_ball_ids() -> set[int]:
    return {2, 3, 4}


@pytest.fixture
def four_rail_contacted_object_ball_ids() -> set[int]:
    return {2, 3, 4, 5}


@pytest.fixture
def repeated_rail_contacted_object_ball_ids() -> set[int]:
    rail_contact_events = [2, 2, 3, 4]
    return set(rail_contact_events)


@pytest.fixture
def overlapping_pocketed_and_rail_ball_ids() -> tuple[set[int], set[int]]:
    return {2}, {2}


@pytest.fixture(params=[0, -1, 10])
def invalid_first_contact_ball_id(request) -> int:
    return request.param


@pytest.fixture(params=[{0}, {-1}, {10}])
def invalid_pocketed_object_ball_ids(request) -> set[int]:
    return request.param


@pytest.fixture(params=[{0}, {-1}, {10}])
def invalid_rail_contacted_object_ball_ids(request) -> set[int]:
    return request.param


@pytest.fixture
def legal_break_foul_result() -> BreakFoulResult:
    return BreakFoulResult(penalty=0.0, should_reset=False)


class TestEvaluateBreakFoul:
    def test_returns_first_contact_penalty_when_first_ball_is_not_one(
        self,
        wrong_first_contact_ball_id: int,
        no_pocketed_object_ball_ids: set[int],
        four_rail_contacted_object_ball_ids: set[int],
    ):
        result = evaluate_break_foul(
            first_contacted_ball_id=wrong_first_contact_ball_id,
            pocketed_object_ball_ids=no_pocketed_object_ball_ids,
            rail_contacted_object_ball_ids=four_rail_contacted_object_ball_ids,
        )

        assert result == BreakFoulResult(
            penalty=FIRST_CONTACT_FOUL_PENALTY,
            should_reset=True,
        )

    def test_returns_no_contact_penalty_when_no_ball_contacted(
        self,
        no_first_contact_ball_id: None,
        no_pocketed_object_ball_ids: set[int],
        four_rail_contacted_object_ball_ids: set[int],
    ):
        """整局沒碰到球比碰到錯球更差（#124）。

        兩者原本同為 -1.5，導致 policy 起點附近的 reward 地形完全是平的。
        """
        result = evaluate_break_foul(
            first_contacted_ball_id=no_first_contact_ball_id,
            pocketed_object_ball_ids=no_pocketed_object_ball_ids,
            rail_contacted_object_ball_ids=four_rail_contacted_object_ball_ids,
        )

        assert result == BreakFoulResult(
            penalty=NO_CONTACT_FOUL_PENALTY,
            should_reset=True,
        )

    def test_foul_penalties_form_a_strictly_increasing_ladder(self):
        """犯規罰分必須嚴格遞增，這是 dense shaping 能運作的前提。

        任兩級相等就會產生一片沒有梯度的平原——#124 第一輪訓練就是踩在
        「沒碰到球」與「碰到錯球」同為 -1.5 的那片平原上收斂到亂打。
        """
        ladder = [
            NO_CONTACT_FOUL_PENALTY,
            FIRST_CONTACT_FOUL_PENALTY,
            INSUFFICIENT_RAIL_CONTACT_PENALTY,
            0.0,
        ]

        assert ladder == sorted(ladder)
        assert len(set(ladder)) == len(ladder)

    def test_first_contact_foul_takes_precedence_over_rail_foul(
        self,
        wrong_first_contact_ball_id: int,
        no_pocketed_object_ball_ids: set[int],
        no_rail_contacted_object_ball_ids: set[int],
    ):
        result = evaluate_break_foul(
            first_contacted_ball_id=wrong_first_contact_ball_id,
            pocketed_object_ball_ids=no_pocketed_object_ball_ids,
            rail_contacted_object_ball_ids=no_rail_contacted_object_ball_ids,
        )

        assert result.penalty == pytest.approx(FIRST_CONTACT_FOUL_PENALTY)
        assert result.penalty != pytest.approx(
            FIRST_CONTACT_FOUL_PENALTY
            + INSUFFICIENT_RAIL_CONTACT_PENALTY
        )
        assert result.should_reset is True

    def test_returns_rail_penalty_when_no_ball_pocketed_and_three_balls_hit_rail(
        self,
        legal_first_contact_ball_id: int,
        no_pocketed_object_ball_ids: set[int],
        three_rail_contacted_object_ball_ids: set[int],
    ):
        result = evaluate_break_foul(
            first_contacted_ball_id=legal_first_contact_ball_id,
            pocketed_object_ball_ids=no_pocketed_object_ball_ids,
            rail_contacted_object_ball_ids=three_rail_contacted_object_ball_ids,
        )

        assert result == BreakFoulResult(
            penalty=INSUFFICIENT_RAIL_CONTACT_PENALTY,
            should_reset=False,
        )

    def test_returns_no_penalty_when_four_balls_hit_rail(
        self,
        legal_first_contact_ball_id: int,
        no_pocketed_object_ball_ids: set[int],
        four_rail_contacted_object_ball_ids: set[int],
    ):
        result = evaluate_break_foul(
            first_contacted_ball_id=legal_first_contact_ball_id,
            pocketed_object_ball_ids=no_pocketed_object_ball_ids,
            rail_contacted_object_ball_ids=four_rail_contacted_object_ball_ids,
        )

        assert result == BreakFoulResult(penalty=0.0, should_reset=False)

    def test_returns_no_penalty_when_object_ball_pocketed(
        self,
        legal_first_contact_ball_id: int,
        one_pocketed_object_ball_id: set[int],
        no_rail_contacted_object_ball_ids: set[int],
    ):
        result = evaluate_break_foul(
            first_contacted_ball_id=legal_first_contact_ball_id,
            pocketed_object_ball_ids=one_pocketed_object_ball_id,
            rail_contacted_object_ball_ids=no_rail_contacted_object_ball_ids,
        )

        assert result == BreakFoulResult(penalty=0.0, should_reset=False)

    def test_counts_each_rail_contacted_ball_once(
        self,
        legal_first_contact_ball_id: int,
        no_pocketed_object_ball_ids: set[int],
        repeated_rail_contacted_object_ball_ids: set[int],
    ):
        result = evaluate_break_foul(
            first_contacted_ball_id=legal_first_contact_ball_id,
            pocketed_object_ball_ids=no_pocketed_object_ball_ids,
            rail_contacted_object_ball_ids=repeated_rail_contacted_object_ball_ids,
        )

        assert len(repeated_rail_contacted_object_ball_ids) == 3
        assert result == BreakFoulResult(
            penalty=INSUFFICIENT_RAIL_CONTACT_PENALTY,
            should_reset=False,
        )

    def test_allows_ball_in_both_pocketed_and_rail_sets(
        self,
        legal_first_contact_ball_id: int,
        overlapping_pocketed_and_rail_ball_ids: tuple[set[int], set[int]],
    ):
        pocketed_ball_ids, rail_contacted_ball_ids = (
            overlapping_pocketed_and_rail_ball_ids
        )

        result = evaluate_break_foul(
            first_contacted_ball_id=legal_first_contact_ball_id,
            pocketed_object_ball_ids=pocketed_ball_ids,
            rail_contacted_object_ball_ids=rail_contacted_ball_ids,
        )

        assert result == BreakFoulResult(penalty=0.0, should_reset=False)

    def test_raises_value_error_for_invalid_first_contact_ball_id(
        self,
        invalid_first_contact_ball_id: int,
        no_pocketed_object_ball_ids: set[int],
        no_rail_contacted_object_ball_ids: set[int],
    ):
        with pytest.raises(ValueError):
            evaluate_break_foul(
                first_contacted_ball_id=invalid_first_contact_ball_id,
                pocketed_object_ball_ids=no_pocketed_object_ball_ids,
                rail_contacted_object_ball_ids=no_rail_contacted_object_ball_ids,
            )

    def test_raises_value_error_for_invalid_pocketed_ball_id(
        self,
        legal_first_contact_ball_id: int,
        invalid_pocketed_object_ball_ids: set[int],
        no_rail_contacted_object_ball_ids: set[int],
    ):
        with pytest.raises(ValueError):
            evaluate_break_foul(
                first_contacted_ball_id=legal_first_contact_ball_id,
                pocketed_object_ball_ids=invalid_pocketed_object_ball_ids,
                rail_contacted_object_ball_ids=no_rail_contacted_object_ball_ids,
            )

    def test_raises_value_error_for_invalid_rail_contacted_ball_id(
        self,
        legal_first_contact_ball_id: int,
        no_pocketed_object_ball_ids: set[int],
        invalid_rail_contacted_object_ball_ids: set[int],
    ):
        with pytest.raises(ValueError):
            evaluate_break_foul(
                first_contacted_ball_id=legal_first_contact_ball_id,
                pocketed_object_ball_ids=no_pocketed_object_ball_ids,
                rail_contacted_object_ball_ids=invalid_rail_contacted_object_ball_ids,
            )

    def test_constants_match_reward_specification(self):
        assert FIRST_CONTACT_FOUL_PENALTY == pytest.approx(-1.5)
        assert INSUFFICIENT_RAIL_CONTACT_PENALTY == pytest.approx(-0.5)
        assert MIN_RAIL_CONTACTED_OBJECT_BALLS == 4


class TestBreakFoulResult:
    def test_is_immutable(
        self,
        legal_break_foul_result: BreakFoulResult,
    ):
        with pytest.raises(FrozenInstanceError):
            legal_break_foul_result.penalty = -1.5
