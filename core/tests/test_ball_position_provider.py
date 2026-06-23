from collections.abc import Sequence
from math import isclose

import pytest

from core.services import BallPositionProvider, BreakShotPositionProvider


class IncompleteBallPositionProvider(BallPositionProvider):
    pass


@pytest.fixture
def break_shot_provider() -> BreakShotPositionProvider:
    return BreakShotPositionProvider()


@pytest.fixture
def break_shot_positions(break_shot_provider: BreakShotPositionProvider) -> dict:
    return break_shot_provider.get_positions()


class TestBallPositionProvider:
    def test_ball_position_provider_cannot_be_instantiated_directly(self):
        # Arrange / Act / Assert
        with pytest.raises(TypeError):
            BallPositionProvider()

    def test_incomplete_subclass_cannot_be_instantiated(self):
        # Arrange / Act / Assert
        with pytest.raises(TypeError):
            IncompleteBallPositionProvider()


class TestBreakShotPositionProvider:
    def test_break_shot_provider_is_ball_position_provider_subclass(self):
        assert issubclass(BreakShotPositionProvider, BallPositionProvider)

    def test_break_shot_provider_instance_is_ball_position_provider(
        self,
        break_shot_provider: BreakShotPositionProvider,
    ):
        assert isinstance(break_shot_provider, BallPositionProvider)

    def test_get_positions_returns_dict(self, break_shot_positions: dict):
        assert isinstance(break_shot_positions, dict)

    def test_get_positions_contains_zero_to_nine_ball_keys(
        self,
        break_shot_positions: dict,
    ):
        expected_keys = set(range(10))

        assert set(break_shot_positions.keys()) == expected_keys

    def test_each_position_is_xy_sequence(self, break_shot_positions: dict):
        for position in break_shot_positions.values():
            assert isinstance(position, Sequence)
            assert not isinstance(position, (str, bytes))
            assert len(position) == 2
            assert all(isinstance(value, (int, float)) for value in position)

    def test_one_ball_is_on_foot_spot(self, break_shot_positions: dict):
        one_ball_x, one_ball_y = break_shot_positions[1]

        assert isclose(one_ball_x, 0.0,   abs_tol=1e-6)
        assert isclose(one_ball_y, 0.635, abs_tol=1e-3)

    def test_nine_ball_is_on_diamond_center_line(self, break_shot_positions: dict):
        nine_ball_x, _ = break_shot_positions[9]

        assert isclose(nine_ball_x, 0.0, abs_tol=1e-6)

    def test_cue_ball_is_in_kitchen(self, break_shot_positions: dict):
        _, cue_ball_y = break_shot_positions[0]

        assert cue_ball_y < -0.635
