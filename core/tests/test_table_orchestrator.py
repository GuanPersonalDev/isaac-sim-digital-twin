from unittest.mock import MagicMock

import pytest

from core.models.action import Action
from core.models.billiard_state import BilliardStatus
from core.models.observation import Observation
from core.services.error_state import ErrorState
from core.services.table_orchestrator import (
    DemoTableOrchestrator,
    TrainingTableOrchestrator,
)


def _observation() -> Observation:
    return Observation(
        ball_positions=[],
        cue_ball_position=[0.0, 0.0, 0.0],
        is_init_state=False,
        is_ball_moving=False,
        is_motion_complete=False,
        has_error=False,
    )


def _action(should_execute_action: bool) -> Action:
    return Action(
        cue_speed=0.0,
        shot_angle=0.0,
        position_offset=[0.0, 0.0],
        cue_ball_placement=[0.0, 0.0],
        should_execute_action=should_execute_action,
    )


@pytest.fixture
def script_controller() -> MagicMock:
    return MagicMock()


@pytest.fixture
def table_ball_set() -> MagicMock:
    return MagicMock()


@pytest.fixture
def ball_position_provider() -> MagicMock:
    return MagicMock()


@pytest.fixture
def ur5_robot() -> MagicMock:
    return MagicMock()


@pytest.fixture
def articulation_api() -> MagicMock:
    return MagicMock()


@pytest.fixture
def impulse_striking_service() -> MagicMock:
    return MagicMock()


@pytest.fixture
def error_state() -> MagicMock:
    # wraps 真正的 ErrorState：既能斷言呼叫參數，has_error()/get_last_exception() 也反映真實狀態
    return MagicMock(wraps=ErrorState())


@pytest.fixture
def demo_orchestrator(
    script_controller: MagicMock,
    table_ball_set: MagicMock,
    ball_position_provider: MagicMock,
    ur5_robot: MagicMock,
    articulation_api: MagicMock,
    error_state: MagicMock,
) -> DemoTableOrchestrator:
    return DemoTableOrchestrator(
        script_controller=script_controller,
        table_ball_set=table_ball_set,
        ball_position_provider=ball_position_provider,
        ur5_robot=ur5_robot,
        articulation_api=articulation_api,
        error_state=error_state,
    )


@pytest.fixture
def training_orchestrator(
    script_controller: MagicMock,
    table_ball_set: MagicMock,
    ball_position_provider: MagicMock,
    impulse_striking_service: MagicMock,
    error_state: MagicMock,
) -> TrainingTableOrchestrator:
    return TrainingTableOrchestrator(
        script_controller=script_controller,
        table_ball_set=table_ball_set,
        ball_position_provider=ball_position_provider,
        impulse_striking_service=impulse_striking_service,
        error_state=error_state,
    )


class TestStepDispatch:
    def test_step_dispatches_reset_when_should_execute_action_true(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        script_controller: MagicMock,
        table_ball_set: MagicMock,
        ball_position_provider: MagicMock,
        ur5_robot: MagicMock,
    ):
        script_controller.get_action.return_value = _action(should_execute_action=True)
        script_controller.get_current_state.return_value = BilliardStatus.RESET
        ball_position_provider.get_positions.return_value = {0: (0.0, 0.0)}

        demo_orchestrator.step(_observation())

        ball_position_provider.get_positions.assert_called_once_with()
        table_ball_set.reset.assert_called_once_with({0: (0.0, 0.0)})
        ur5_robot.reset.assert_called_once_with()

    def test_step_dispatches_aiming(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        script_controller: MagicMock,
    ):
        action = _action(should_execute_action=True)
        script_controller.get_action.return_value = action
        script_controller.get_current_state.return_value = BilliardStatus.AIMING
        demo_orchestrator._execute_aim = MagicMock()

        demo_orchestrator.step(_observation())

        demo_orchestrator._execute_aim.assert_called_once_with(action)

    def test_step_dispatches_striking(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        script_controller: MagicMock,
    ):
        action = _action(should_execute_action=True)
        script_controller.get_action.return_value = action
        script_controller.get_current_state.return_value = BilliardStatus.STRIKING
        demo_orchestrator._execute_strike = MagicMock()

        demo_orchestrator.step(_observation())

        demo_orchestrator._execute_strike.assert_called_once_with(action)

    def test_step_skips_downstream_when_should_execute_action_false(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        script_controller: MagicMock,
        table_ball_set: MagicMock,
        ur5_robot: MagicMock,
        ball_position_provider: MagicMock,
    ):
        script_controller.get_action.return_value = _action(should_execute_action=False)
        script_controller.get_current_state.return_value = BilliardStatus.RESET

        demo_orchestrator.step(_observation())

        ball_position_provider.get_positions.assert_not_called()
        table_ball_set.reset.assert_not_called()
        ur5_robot.reset.assert_not_called()

    @pytest.mark.parametrize("state", [BilliardStatus.WAITING, BilliardStatus.IDLE])
    def test_step_has_no_downstream_action_for_waiting_or_idle(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        script_controller: MagicMock,
        table_ball_set: MagicMock,
        ur5_robot: MagicMock,
        state: BilliardStatus,
    ):
        script_controller.get_action.return_value = _action(should_execute_action=True)
        script_controller.get_current_state.return_value = state

        demo_orchestrator.step(_observation())

        table_ball_set.reset.assert_not_called()
        ur5_robot.reset.assert_not_called()


class TestStepErrorHandling:
    def test_downstream_exception_is_recorded_and_not_reraised(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        script_controller: MagicMock,
        error_state: MagicMock,
    ):
        script_controller.get_action.return_value = _action(should_execute_action=True)
        script_controller.get_current_state.return_value = BilliardStatus.AIMING
        boom = RuntimeError("boom")
        demo_orchestrator._execute_aim = MagicMock(side_effect=boom)

        demo_orchestrator.step(_observation())  # 不應往外拋

        error_state.mark_error.assert_called_once_with(boom)
        assert error_state.has_error() is True
        assert error_state.get_last_exception() is boom

    def test_no_exception_leaves_error_state_untouched(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        script_controller: MagicMock,
        ball_position_provider: MagicMock,
        error_state: MagicMock,
    ):
        script_controller.get_action.return_value = _action(should_execute_action=True)
        script_controller.get_current_state.return_value = BilliardStatus.RESET
        ball_position_provider.get_positions.return_value = {0: (0.0, 0.0)}

        demo_orchestrator.step(_observation())

        error_state.mark_error.assert_not_called()
        assert error_state.has_error() is False


class TestReset:
    def test_reset_clears_error_state_and_resets_script_controller(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        script_controller: MagicMock,
        error_state: MagicMock,
    ):
        demo_orchestrator.reset()

        error_state.clear.assert_called_once_with()
        script_controller.reset.assert_called_once_with()

    def test_reset_clears_a_previously_recorded_error(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        error_state: MagicMock,
    ):
        error_state.mark_error(RuntimeError("boom"))
        assert error_state.has_error() is True

        demo_orchestrator.reset()

        assert error_state.has_error() is False
        assert error_state.get_last_exception() is None


class TestResetBalls:
    def test_reset_balls_uses_position_provider_positions(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        table_ball_set: MagicMock,
        ball_position_provider: MagicMock,
    ):
        ball_position_provider.get_positions.return_value = {1: (0.1, 0.2)}

        demo_orchestrator._reset_balls()

        table_ball_set.reset.assert_called_once_with({1: (0.1, 0.2)})


class TestDemoTableOrchestrator:
    def test_reset_downstream_calls_ur5_robot_reset(
        self,
        demo_orchestrator: DemoTableOrchestrator,
        ur5_robot: MagicMock,
    ):
        demo_orchestrator._reset_downstream()

        ur5_robot.reset.assert_called_once_with()


class TestTrainingTableOrchestrator:
    def test_step_dispatches_reset_balls_only(
        self,
        training_orchestrator: TrainingTableOrchestrator,
        script_controller: MagicMock,
        table_ball_set: MagicMock,
        ball_position_provider: MagicMock,
    ):
        script_controller.get_action.return_value = _action(should_execute_action=True)
        script_controller.get_current_state.return_value = BilliardStatus.RESET
        ball_position_provider.get_positions.return_value = {0: (0.0, 0.0)}

        training_orchestrator.step(_observation())

        table_ball_set.reset.assert_called_once_with({0: (0.0, 0.0)})

    def test_reset_downstream_is_noop(
        self, training_orchestrator: TrainingTableOrchestrator
    ):
        training_orchestrator._reset_downstream()

    def test_execute_aim_is_noop(
        self, training_orchestrator: TrainingTableOrchestrator
    ):
        training_orchestrator._execute_aim(_action(should_execute_action=True))

    def test_execute_strike_calls_impulse_service_with_table_z(
        self,
        training_orchestrator: TrainingTableOrchestrator,
        impulse_striking_service: MagicMock,
        table_ball_set: MagicMock,
    ):
        table_ball_set.get_table_z.return_value = 0.75
        table_ball_set.get_table_x_y.return_value = (5.0, 3.0)
        action = _action(should_execute_action=True)

        training_orchestrator._execute_strike(action)

        impulse_striking_service.strike.assert_called_once_with(action, 5.0, 3.0, table_z=0.75)
