from unittest.mock import MagicMock, call

import pytest

from core.models.billiard_state import BilliardStatus
from core.models.observation import Observation
from core.services.table_runtime import TableRuntime


@pytest.fixture
def observation_builder() -> MagicMock:
    return MagicMock()


@pytest.fixture
def orchestrator() -> MagicMock:
    return MagicMock()


@pytest.fixture
def observation() -> Observation:
    return Observation(
        ball_positions=[[1.0, 2.0, 3.0]],
        cue_ball_position=[0.0, 0.0, 0.0],
        is_init_state=False,
        is_ball_moving=False,
        is_motion_complete=True,
        has_error=False,
    )


@pytest.fixture
def observations() -> list[Observation]:
    return [
        Observation(
            ball_positions=[[1.0, 2.0, 3.0]],
            cue_ball_position=[0.0, 0.0, 0.0],
            is_init_state=False,
            is_ball_moving=False,
            is_motion_complete=True,
            has_error=False,
        ),
        Observation(
            ball_positions=[[4.0, 5.0, 6.0]],
            cue_ball_position=[1.0, 1.0, 1.0],
            is_init_state=True,
            is_ball_moving=False,
            is_motion_complete=False,
            has_error=False,
        ),
        Observation(
            ball_positions=[[7.0, 8.0, 9.0]],
            cue_ball_position=[2.0, 2.0, 2.0],
            is_init_state=False,
            is_ball_moving=True,
            is_motion_complete=False,
            has_error=True,
        ),
    ]


@pytest.fixture
def table_runtime(
    observation_builder: MagicMock, orchestrator: MagicMock
) -> TableRuntime:
    return TableRuntime(
        observation_builder=observation_builder,
        orchestrator=orchestrator,
    )


class TestTableRuntime:
    def test_table_runtime_tick_builds_observation_then_steps_orchestrator(
        self,
        table_runtime: TableRuntime,
        observation_builder: MagicMock,
        orchestrator: MagicMock,
        observation: Observation,
    ):
        calls = []

        def build_observation() -> Observation:
            calls.append("build")
            return observation

        def step_orchestrator(received_observation: Observation) -> None:
            calls.append("step")
            assert received_observation is observation

        observation_builder.build.side_effect = build_observation
        orchestrator.step.side_effect = step_orchestrator

        table_runtime.tick()

        observation_builder.build.assert_called_once_with()
        orchestrator.step.assert_called_once_with(observation)
        assert calls == ["build", "step"]

    def test_table_runtime_forwards_each_tick_observation_to_orchestrator_step(
        self,
        table_runtime: TableRuntime,
        observation_builder: MagicMock,
        orchestrator: MagicMock,
        observations: list[Observation],
    ):
        observation_builder.build.side_effect = observations

        for _ in observations:
            table_runtime.tick()

        assert observation_builder.build.call_count == len(observations)
        assert orchestrator.step.call_args_list == [
            call(observations[0]),
            call(observations[1]),
            call(observations[2]),
        ]

        for index, step_call in enumerate(orchestrator.step.call_args_list):
            assert step_call.args[0] is observations[index]

    def test_get_last_observation_returns_none_before_first_tick(
        self, table_runtime: TableRuntime
    ):
        assert table_runtime.get_last_observation() is None

    def test_get_last_observation_returns_most_recent_built_observation(
        self,
        table_runtime: TableRuntime,
        observation_builder: MagicMock,
        observations: list[Observation],
    ):
        observation_builder.build.side_effect = observations

        for expected in observations:
            table_runtime.tick()
            assert table_runtime.get_last_observation() is expected

    def test_get_current_state_delegates_to_orchestrator(
        self,
        table_runtime: TableRuntime,
        orchestrator: MagicMock,
    ):
        orchestrator.get_current_state.return_value = BilliardStatus.AIMING

        assert table_runtime.get_current_state() == BilliardStatus.AIMING

    def test_request_full_reset_resets_the_state_machine_immediately(
        self,
        table_runtime: TableRuntime,
        orchestrator: MagicMock,
    ):
        table_runtime.request_full_reset()

        orchestrator.reset.assert_called_once_with()
        orchestrator.full_reset.assert_not_called()

    def test_request_full_reset_drops_the_observation_left_over_from_the_previous_run(
        self,
        table_runtime: TableRuntime,
        observation_builder: MagicMock,
        observation: Observation,
    ):
        observation_builder.build.return_value = observation
        table_runtime.tick()
        assert table_runtime.get_last_observation() is observation

        table_runtime.request_full_reset()

        assert table_runtime.get_last_observation() is None

    def test_request_full_reset_defers_the_scene_reset_to_the_next_tick(
        self,
        table_runtime: TableRuntime,
        observation_builder: MagicMock,
        orchestrator: MagicMock,
        observation: Observation,
    ):
        observation_builder.build.return_value = observation
        calls = []
        orchestrator.full_reset.side_effect = lambda: calls.append("full_reset")
        observation_builder.build.side_effect = lambda: calls.append("build") or observation
        orchestrator.step.side_effect = lambda _: calls.append("step")

        table_runtime.request_full_reset()
        table_runtime.tick()

        # 場景重置必須在建 Observation 之前，否則這一 tick 讀到的是重置前的球位
        assert calls == ["full_reset", "build", "step"]

    def test_pending_full_reset_runs_only_once(
        self,
        table_runtime: TableRuntime,
        observation_builder: MagicMock,
        orchestrator: MagicMock,
        observation: Observation,
    ):
        observation_builder.build.return_value = observation

        table_runtime.request_full_reset()
        table_runtime.tick()
        table_runtime.tick()

        orchestrator.full_reset.assert_called_once_with()

    def test_tick_without_a_pending_request_does_not_reset(
        self,
        table_runtime: TableRuntime,
        observation_builder: MagicMock,
        orchestrator: MagicMock,
        observation: Observation,
    ):
        observation_builder.build.return_value = observation

        table_runtime.tick()

        orchestrator.full_reset.assert_not_called()
