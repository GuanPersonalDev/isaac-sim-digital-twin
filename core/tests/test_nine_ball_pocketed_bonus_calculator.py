import pytest

from core.services.nine_ball_pocketed_bonus_calculator import (
    NINE_BALL_POCKETED_BONUS,
    calculate_nine_ball_pocketed_bonus,
)


@pytest.fixture
def nine_ball_only_pocketed() -> tuple[bool, bool]:
    return True, False


@pytest.fixture
def nine_ball_and_cue_ball_pocketed() -> tuple[bool, bool]:
    return True, True


@pytest.fixture
def cue_ball_only_pocketed() -> tuple[bool, bool]:
    return False, True


@pytest.fixture
def neither_ball_pocketed() -> tuple[bool, bool]:
    return False, False


class TestCalculateNineBallPocketedBonus:
    def test_returns_bonus_when_nine_ball_pocketed_without_cue_ball(
        self,
        nine_ball_only_pocketed: tuple[bool, bool],
    ):
        nine_ball_pocketed, cue_ball_pocketed = nine_ball_only_pocketed

        bonus = calculate_nine_ball_pocketed_bonus(
            nine_ball_pocketed=nine_ball_pocketed,
            cue_ball_pocketed=cue_ball_pocketed,
        )

        assert bonus == pytest.approx(NINE_BALL_POCKETED_BONUS)
        assert bonus == pytest.approx(3.0)

    def test_returns_zero_when_nine_ball_and_cue_ball_both_pocketed(
        self,
        nine_ball_and_cue_ball_pocketed: tuple[bool, bool],
    ):
        nine_ball_pocketed, cue_ball_pocketed = (
            nine_ball_and_cue_ball_pocketed
        )

        bonus = calculate_nine_ball_pocketed_bonus(
            nine_ball_pocketed=nine_ball_pocketed,
            cue_ball_pocketed=cue_ball_pocketed,
        )

        assert bonus == pytest.approx(0.0)

    def test_returns_zero_when_only_cue_ball_pocketed(
        self,
        cue_ball_only_pocketed: tuple[bool, bool],
    ):
        nine_ball_pocketed, cue_ball_pocketed = cue_ball_only_pocketed

        bonus = calculate_nine_ball_pocketed_bonus(
            nine_ball_pocketed=nine_ball_pocketed,
            cue_ball_pocketed=cue_ball_pocketed,
        )

        assert bonus == pytest.approx(0.0)

    def test_returns_zero_when_neither_ball_pocketed(
        self,
        neither_ball_pocketed: tuple[bool, bool],
    ):
        nine_ball_pocketed, cue_ball_pocketed = neither_ball_pocketed

        bonus = calculate_nine_ball_pocketed_bonus(
            nine_ball_pocketed=nine_ball_pocketed,
            cue_ball_pocketed=cue_ball_pocketed,
        )

        assert bonus == pytest.approx(0.0)

    def test_bonus_constant_is_three(self):
        assert NINE_BALL_POCKETED_BONUS == pytest.approx(3.0)
