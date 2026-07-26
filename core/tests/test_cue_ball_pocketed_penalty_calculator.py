import pytest

from core.services.cue_ball_pocketed_penalty_calculator import (
    CUE_BALL_POCKETED_PENALTY,
    calculate_cue_ball_pocketed_penalty,
)


class TestCalculateCueBallPocketedPenalty:
    def test_returns_penalty_when_cue_ball_pocketed(self):
        penalty = calculate_cue_ball_pocketed_penalty(cue_ball_pocketed=True)

        assert penalty == pytest.approx(CUE_BALL_POCKETED_PENALTY)
        assert penalty == pytest.approx(-3.5)

    def test_returns_zero_when_cue_ball_not_pocketed(self):
        penalty = calculate_cue_ball_pocketed_penalty(cue_ball_pocketed=False)

        assert penalty == pytest.approx(0.0)
