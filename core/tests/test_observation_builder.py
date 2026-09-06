import inspect
from unittest.mock import MagicMock

import pytest

from core.services.observation_builder import (
    DemoTableObservationBuilder,
    ObservationBuilder,
    TrainingTableObservationBuilder,
)

_BALL_0 = "/World/BilliardTable_0/Balls/Ball_0"
_BALL_1 = "/World/BilliardTable_0/Balls/Ball_1"
_BALL_2 = "/World/BilliardTable_0/Balls/Ball_2"


@pytest.fixture
def table_ball_set():
    ball_set = MagicMock()
    ball_set.get_ball_prim_paths.return_value = [_BALL_0, _BALL_1, _BALL_2]
    ball_set.get_table_x_y.return_value = (0.0, 0.0)
    return ball_set


@pytest.fixture
def rigid_body_api():
    return MagicMock()


@pytest.fixture
def ball_motion_monitor():
    monitor = MagicMock()
    monitor.is_any_ball_moving.return_value = False
    return monitor


@pytest.fixture
def error_state():
    state = MagicMock()
    state.has_error.return_value = False
    return state


@pytest.fixture
def robot_arm():
    return MagicMock()


@pytest.fixture
def ball_position_provider():
    provider = MagicMock()
    provider.get_positions.return_value = {
        0: (0.0, 0.0),
        1: (0.1, 0.2),
        2: (-0.1, -0.2),
    }
    return provider


@pytest.fixture
def training_builder(
    table_ball_set, rigid_body_api, ball_motion_monitor, error_state, ball_position_provider
):
    return TrainingTableObservationBuilder(
        table_ball_set, rigid_body_api, ball_motion_monitor, error_state, ball_position_provider
    )


@pytest.fixture
def demo_builder(
    table_ball_set,
    rigid_body_api,
    ball_motion_monitor,
    error_state,
    ball_position_provider,
    robot_arm,
):
    return DemoTableObservationBuilder(
        table_ball_set,
        rigid_body_api,
        ball_motion_monitor,
        error_state,
        ball_position_provider,
        robot_arm,
    )


def _set_ball_world_positions(rigid_body_api, positions_by_prim_path):
    # ObservationBuilder 改成一次批次讀取（get_positions，見
    # core/ports/rigid_body_api.py 的效能說明），測試資料仍以 prim path 對應
    # 座標的形式描述比較好讀。
    rigid_body_api.get_position.side_effect = lambda prim_path: positions_by_prim_path[prim_path]
    rigid_body_api.get_positions.side_effect = lambda prim_paths: [
        positions_by_prim_path[prim_path] for prim_path in prim_paths
    ]


class TestObservationBuilderBallPositions:
    def test_observation_builder_builds_ball_positions_from_rigid_body_api(
        self, training_builder, rigid_body_api
    ):
        _set_ball_world_positions(
            rigid_body_api,
            {
                _BALL_0: [0.0, 0.0, 0.75],
                _BALL_1: [0.1, 0.2, 0.75],
                _BALL_2: [-0.1, -0.2, 0.75],
            },
        )

        observation = training_builder.build()

        assert observation.ball_positions == [
            [0.0, 0.0, 0.75],
            [0.1, 0.2, 0.75],
            [-0.1, -0.2, 0.75],
        ]

    def test_observation_builder_cue_ball_position_is_ball_zero(
        self, training_builder, rigid_body_api
    ):
        _set_ball_world_positions(
            rigid_body_api,
            {
                _BALL_0: [0.0, 0.0, 0.75],
                _BALL_1: [0.1, 0.2, 0.75],
                _BALL_2: [-0.1, -0.2, 0.75],
            },
        )

        observation = training_builder.build()

        assert observation.cue_ball_position == observation.ball_positions[0]


class TestObservationBuilderInitState:
    def test_observation_builder_is_init_state_true_within_tolerance(
        self, training_builder, rigid_body_api
    ):
        # 球 0 相對預設擺位偏移 3mm（< 5mm 容許誤差）
        _set_ball_world_positions(
            rigid_body_api,
            {
                _BALL_0: [0.003, 0.0, 0.75],
                _BALL_1: [0.1, 0.2, 0.75],
                _BALL_2: [-0.1, -0.2, 0.75],
            },
        )

        observation = training_builder.build()

        assert observation.is_init_state is True

    def test_observation_builder_is_init_state_false_outside_tolerance(
        self, training_builder, rigid_body_api
    ):
        # 球 1 相對預設擺位偏移 6mm（> 5mm 容許誤差）
        _set_ball_world_positions(
            rigid_body_api,
            {
                _BALL_0: [0.0, 0.0, 0.75],
                _BALL_1: [0.106, 0.2, 0.75],
                _BALL_2: [-0.1, -0.2, 0.75],
            },
        )

        observation = training_builder.build()

        assert observation.is_init_state is False


class TestObservationBuilderErrorState:
    def test_observation_builder_has_error_reflects_error_state(
        self, training_builder, rigid_body_api, error_state
    ):
        _set_ball_world_positions(
            rigid_body_api,
            {
                _BALL_0: [0.0, 0.0, 0.75],
                _BALL_1: [0.1, 0.2, 0.75],
                _BALL_2: [-0.1, -0.2, 0.75],
            },
        )
        error_state.has_error.return_value = True

        observation = training_builder.build()

        assert observation.has_error is True


class TestObservationBuilderInterface:
    def test_observation_builder_does_not_accept_previous_observation(self):
        signature = inspect.signature(ObservationBuilder.build)

        assert list(signature.parameters) == ["self"]


class TestDemoTableObservationBuilderMotionComplete:
    @pytest.mark.parametrize("reset_complete", [True, False])
    def test_demo_observation_builder_motion_complete_uses_ur5_reset_complete(
        self, demo_builder, robot_arm, reset_complete
    ):
        robot_arm.is_reset_complete.return_value = reset_complete

        assert demo_builder._is_motion_complete() is reset_complete


class TestTrainingTableObservationBuilderMotionComplete:
    def test_training_observation_builder_motion_complete_always_true(
        self, training_builder
    ):
        assert training_builder._is_motion_complete() is True
