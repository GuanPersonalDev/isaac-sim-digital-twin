import math
from collections.abc import Callable

import pytest

from core.models.observation import Observation
from core.services.rl_observation_encoder import encode_rl_observation


@pytest.fixture
def observation_factory() -> Callable[..., Observation]:
    def create(
        ball_positions: list[list[float]],
        *,
        is_init_state: bool = False,
        is_ball_moving: bool = False,
        is_motion_complete: bool = True,
        has_error: bool = False,
    ) -> Observation:
        return Observation(
            ball_positions=ball_positions,
            cue_ball_position=(
                ball_positions[0] if ball_positions else [0.0, 0.0]
            ),
            is_init_state=is_init_state,
            is_ball_moving=is_ball_moving,
            is_motion_complete=is_motion_complete,
            has_error=has_error,
        )

    return create


@pytest.fixture
def table_position() -> tuple[float, float]:
    return (100.0, 200.0)


@pytest.fixture
def max_offset() -> float:
    return 0.4


@pytest.fixture
def world_ball_positions() -> list[list[float]]:
    return [
        [100.0 + ball_id, 200.0 + ball_id, 0.75 + ball_id]
        for ball_id in range(10)
    ]


@pytest.fixture
def observation(
    observation_factory: Callable[..., Observation],
    world_ball_positions: list[list[float]],
) -> Observation:
    return observation_factory(world_ball_positions)


@pytest.fixture
def expected_rl_observation(max_offset: float) -> list[float]:
    object_ball_positions = [
        coordinate
        for ball_id in range(1, 10)
        for coordinate in (float(ball_id), float(ball_id))
    ]
    return object_ball_positions + [0.0, 0.0] + [max_offset]


@pytest.fixture(params=[math.nan, math.inf, -math.inf])
def non_finite_value(request) -> float:
    return request.param


@pytest.fixture(params=[True, "invalid", None])
def non_numeric_value(request):
    return request.param


class TestEncodeRlObservation:
    def test_returns_object_balls_then_cue_ball_then_max_offset(
        self,
        observation: Observation,
        table_position: tuple[float, float],
        max_offset: float,
        expected_rl_observation: list[float],
    ):
        encoded = encode_rl_observation(
            observation,
            table_position,
            max_offset,
        )

        assert encoded == pytest.approx(expected_rl_observation)
        assert len(encoded) == 21

    def test_subtracts_table_world_position_from_each_ball(
        self,
        observation_factory: Callable[..., Observation],
        max_offset: float,
    ):
        ball_positions = [
            [10.5 + ball_id, -4.25 - ball_id, 7.0]
            for ball_id in range(10)
        ]
        observation = observation_factory(ball_positions)

        encoded = encode_rl_observation(
            observation,
            table_position=(10.5, -4.25),
            max_offset=max_offset,
        )

        assert encoded[:2] == pytest.approx([1.0, -1.0])
        # 母球 XY 位於第 19、20 維，最後一維是 max_offset。
        assert encoded[-3:-1] == pytest.approx([0.0, 0.0])

    def test_ignores_z_coordinate(
        self,
        observation_factory: Callable[..., Observation],
        table_position: tuple[float, float],
        max_offset: float,
        world_ball_positions: list[list[float]],
        expected_rl_observation: list[float],
    ):
        changed_z_positions = [
            [position[0], position[1], position[2] + 1000.0]
            for position in world_ball_positions
        ]
        original = observation_factory(world_ball_positions)
        changed_z = observation_factory(changed_z_positions)

        original_encoded = encode_rl_observation(
            original,
            table_position,
            max_offset,
        )
        changed_z_encoded = encode_rl_observation(
            changed_z,
            table_position,
            max_offset,
        )

        assert original_encoded == pytest.approx(expected_rl_observation)
        assert changed_z_encoded == pytest.approx(expected_rl_observation)

    def test_runtime_flags_do_not_affect_encoded_values(
        self,
        observation_factory: Callable[..., Observation],
        table_position: tuple[float, float],
        max_offset: float,
        world_ball_positions: list[list[float]],
        expected_rl_observation: list[float],
    ):
        normal = observation_factory(world_ball_positions)
        changed_flags = observation_factory(
            world_ball_positions,
            is_init_state=True,
            is_ball_moving=True,
            is_motion_complete=False,
            has_error=True,
        )

        normal_encoded = encode_rl_observation(
            normal,
            table_position,
            max_offset,
        )
        changed_flags_encoded = encode_rl_observation(
            changed_flags,
            table_position,
            max_offset,
        )

        assert normal_encoded == pytest.approx(expected_rl_observation)
        assert changed_flags_encoded == pytest.approx(expected_rl_observation)

    def test_uses_current_xy_for_pocketed_or_hidden_ball(
        self,
        observation_factory: Callable[..., Observation],
        table_position: tuple[float, float],
        max_offset: float,
        world_ball_positions: list[list[float]],
    ):
        pocketed_ball_positions = [
            position.copy() for position in world_ball_positions
        ]
        pocketed_ball_positions[4] = [101.27, 200.635, -10.0]
        observation = observation_factory(pocketed_ball_positions)

        encoded = encode_rl_observation(
            observation,
            table_position,
            max_offset,
        )

        assert encoded[6:8] == pytest.approx([1.27, 0.635])

    @pytest.mark.parametrize("ball_count", [0, 9, 11])
    def test_rejects_ball_count_other_than_ten(
        self,
        observation_factory: Callable[..., Observation],
        table_position: tuple[float, float],
        max_offset: float,
        ball_count: int,
    ):
        ball_positions = [[0.0, 0.0, 0.0] for _ in range(ball_count)]
        observation = observation_factory(ball_positions)

        with pytest.raises(ValueError):
            encode_rl_observation(observation, table_position, max_offset)

    def test_rejects_ball_position_without_xy(
        self,
        observation_factory: Callable[..., Observation],
        table_position: tuple[float, float],
        max_offset: float,
        world_ball_positions: list[list[float]],
    ):
        invalid_positions = [
            position.copy() for position in world_ball_positions
        ]
        invalid_positions[3] = [1.0]
        observation = observation_factory(invalid_positions)

        with pytest.raises(ValueError):
            encode_rl_observation(observation, table_position, max_offset)

    def test_rejects_non_finite_ball_xy(
        self,
        observation_factory: Callable[..., Observation],
        table_position: tuple[float, float],
        max_offset: float,
        world_ball_positions: list[list[float]],
        non_finite_value: float,
    ):
        invalid_positions = [
            position.copy() for position in world_ball_positions
        ]
        invalid_positions[5][0] = non_finite_value
        observation = observation_factory(invalid_positions)

        with pytest.raises(ValueError):
            encode_rl_observation(observation, table_position, max_offset)

    def test_rejects_non_numeric_ball_xy(
        self,
        observation_factory: Callable[..., Observation],
        table_position: tuple[float, float],
        max_offset: float,
        world_ball_positions: list[list[float]],
        non_numeric_value,
    ):
        invalid_positions = [
            position.copy() for position in world_ball_positions
        ]
        invalid_positions[5][0] = non_numeric_value
        observation = observation_factory(invalid_positions)

        with pytest.raises(ValueError):
            encode_rl_observation(observation, table_position, max_offset)

    @pytest.mark.parametrize(
        "invalid_table_position",
        [
            (),
            (0.0,),
            (0.0, 0.0, 0.0),
        ],
    )
    def test_rejects_table_position_without_two_values(
        self,
        observation: Observation,
        max_offset: float,
        invalid_table_position,
    ):
        with pytest.raises(ValueError):
            encode_rl_observation(
                observation,
                invalid_table_position,
                max_offset,
            )

    def test_rejects_non_finite_table_position(
        self,
        observation: Observation,
        max_offset: float,
        non_finite_value: float,
    ):
        with pytest.raises(ValueError):
            encode_rl_observation(
                observation,
                (non_finite_value, 0.0),
                max_offset,
            )


class TestMaxOffsetDimension:
    def test_appends_max_offset_as_last_dimension(
        self,
        observation: Observation,
        table_position: tuple[float, float],
    ):
        encoded = encode_rl_observation(observation, table_position, 0.42)

        assert len(encoded) == 21
        assert encoded[-1] == pytest.approx(0.42)

    @pytest.mark.parametrize("boundary_max_offset", [0.0, 1.0])
    def test_accepts_range_boundaries(
        self,
        observation: Observation,
        table_position: tuple[float, float],
        boundary_max_offset: float,
    ):
        encoded = encode_rl_observation(
            observation,
            table_position,
            boundary_max_offset,
        )

        assert encoded[-1] == pytest.approx(boundary_max_offset)

    def test_does_not_affect_ball_position_dimensions(
        self,
        observation: Observation,
        table_position: tuple[float, float],
    ):
        low = encode_rl_observation(observation, table_position, 0.0)
        high = encode_rl_observation(observation, table_position, 1.0)

        assert low[:20] == pytest.approx(high[:20])
        assert low[-1] != high[-1]

    @pytest.mark.parametrize(
        "out_of_range_max_offset",
        [-0.1, 1.1, -1.0, 2.0],
    )
    def test_rejects_value_out_of_range(
        self,
        observation: Observation,
        table_position: tuple[float, float],
        out_of_range_max_offset: float,
    ):
        with pytest.raises(ValueError):
            encode_rl_observation(
                observation,
                table_position,
                out_of_range_max_offset,
            )

    def test_rejects_non_finite_value(
        self,
        observation: Observation,
        table_position: tuple[float, float],
        non_finite_value: float,
    ):
        with pytest.raises(ValueError):
            encode_rl_observation(
                observation,
                table_position,
                non_finite_value,
            )

    def test_rejects_non_numeric_value(
        self,
        observation: Observation,
        table_position: tuple[float, float],
        non_numeric_value,
    ):
        with pytest.raises(ValueError):
            encode_rl_observation(
                observation,
                table_position,
                non_numeric_value,
            )
