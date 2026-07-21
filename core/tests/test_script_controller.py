import pytest

from core.controllers.script_controller import ScriptController
from core.models.billiard_state import BilliardStatus
from core.models.observation import Observation


def _observation(
    is_init_state: bool = False,
    is_ball_moving: bool = False,
    is_motion_complete: bool = False,
    has_error: bool = False,
) -> Observation:
    return Observation(
        ball_positions=[[0.0, 0.0, 0.0]],
        cue_ball_position=[-0.3, 0.0, 0.0],
        is_init_state=is_init_state,
        is_ball_moving=is_ball_moving,
        is_motion_complete=is_motion_complete,
        has_error=has_error,
    )


@pytest.fixture
def controller() -> ScriptController:
    return ScriptController()


def _advance_to_idle(controller: ScriptController) -> None:
    controller.get_action(_observation(is_motion_complete=True))


def _advance_to_aiming(controller: ScriptController) -> None:
    _advance_to_idle(controller)
    controller.get_action(_observation(is_init_state=True, is_ball_moving=False))


def _advance_to_striking(controller: ScriptController) -> None:
    _advance_to_aiming(controller)
    controller.get_action(_observation(is_motion_complete=True))


def _advance_to_waiting(controller: ScriptController) -> None:
    _advance_to_striking(controller)
    controller.get_action(_observation(is_motion_complete=True))


class TestInitialState:
    def test_starts_in_reset_state(self, controller: ScriptController):
        # Assert
        assert controller._current_state == BilliardStatus.RESET


class TestGetCurrentState:
    def test_get_current_state_returns_reset_initially(self, controller: ScriptController):
        # Assert
        assert controller.get_current_state() == BilliardStatus.RESET

    def test_get_current_state_reflects_transitions(self, controller: ScriptController):
        # Act
        _advance_to_idle(controller)

        # Assert
        assert controller.get_current_state() == BilliardStatus.IDLE


class TestResetToIdle:
    def test_stays_in_reset_when_motion_not_complete(self, controller: ScriptController):
        # Act
        controller.get_action(_observation(is_motion_complete=False))

        # Assert
        assert controller._current_state == BilliardStatus.RESET

    def test_transitions_to_idle_when_motion_complete(self, controller: ScriptController):
        # Act
        controller.get_action(_observation(is_motion_complete=True))

        # Assert
        assert controller._current_state == BilliardStatus.IDLE


class TestIdleToAiming:
    def test_stays_idle_when_table_not_at_initial_position(self, controller: ScriptController):
        # Arrange
        _advance_to_idle(controller)

        # Act
        controller.get_action(_observation(is_init_state=False, is_ball_moving=False))

        # Assert
        assert controller._current_state == BilliardStatus.IDLE

    def test_stays_idle_when_ball_still_moving(self, controller: ScriptController):
        # Arrange
        _advance_to_idle(controller)

        # Act
        controller.get_action(_observation(is_init_state=True, is_ball_moving=True))

        # Assert
        assert controller._current_state == BilliardStatus.IDLE

    def test_transitions_to_aiming_when_table_reset_and_balls_still(
        self, controller: ScriptController
    ):
        # Arrange
        _advance_to_idle(controller)

        # Act
        controller.get_action(_observation(is_init_state=True, is_ball_moving=False))

        # Assert
        assert controller._current_state == BilliardStatus.AIMING


class TestAimingToStriking:
    def test_waits_until_motion_complete(self, controller: ScriptController):
        # Arrange
        _advance_to_aiming(controller)

        # Act
        controller.get_action(_observation(is_motion_complete=False))

        # Assert
        assert controller._current_state == BilliardStatus.AIMING

    def test_transitions_to_striking_when_motion_complete(self, controller: ScriptController):
        # Arrange
        _advance_to_aiming(controller)

        # Act
        controller.get_action(_observation(is_motion_complete=True))

        # Assert
        assert controller._current_state == BilliardStatus.STRIKING


class TestStrikingToWaiting:
    def test_waits_until_motion_complete(self, controller: ScriptController):
        # Arrange
        _advance_to_striking(controller)

        # Act
        controller.get_action(_observation(is_motion_complete=False))

        # Assert
        assert controller._current_state == BilliardStatus.STRIKING

    def test_transitions_to_waiting_when_motion_complete(self, controller: ScriptController):
        # Arrange
        _advance_to_striking(controller)

        # Act
        controller.get_action(_observation(is_motion_complete=True))

        # Assert
        assert controller._current_state == BilliardStatus.WAITING

    def test_striking_action_uses_fixed_speed_with_no_angle_or_offset(
        self, controller: ScriptController
    ):
        # Arrange
        _advance_to_aiming(controller)

        # Act
        action = controller.get_action(_observation(is_motion_complete=True))

        # Assert
        assert controller._current_state == BilliardStatus.STRIKING
        assert action.cue_speed == ScriptController.MAX_ARM_SPEED
        assert action.shot_angle == 0
        assert action.position_offset == [0.0, 0.0]


class TestWaitingToReset:
    def test_stays_waiting_while_balls_moving(self, controller: ScriptController):
        # Arrange
        _advance_to_waiting(controller)

        # Act
        controller.get_action(_observation(is_ball_moving=True))

        # Assert
        assert controller._current_state == BilliardStatus.WAITING

    def test_transitions_to_reset_when_balls_stop(self, controller: ScriptController):
        # Arrange
        _advance_to_waiting(controller)

        # Act
        controller.get_action(_observation(is_ball_moving=False))

        # Assert
        assert controller._current_state == BilliardStatus.RESET


class TestErrorHandling:
    def test_any_state_transitions_to_error_when_has_error(self, controller: ScriptController):
        # Act
        controller.get_action(_observation(has_error=True))

        # Assert
        assert controller._current_state == BilliardStatus.ERROR

    def test_error_takes_priority_over_normal_transition(self, controller: ScriptController):
        # Arrange
        _advance_to_aiming(controller)

        # Act
        controller.get_action(_observation(is_motion_complete=True, has_error=True))

        # Assert
        assert controller._current_state == BilliardStatus.ERROR

    def test_error_state_does_not_auto_recover(self, controller: ScriptController):
        # Arrange
        controller.get_action(_observation(has_error=True))

        # Act
        controller.get_action(
            _observation(is_init_state=True, is_ball_moving=False, is_motion_complete=True)
        )

        # Assert
        assert controller._current_state == BilliardStatus.ERROR


class TestReset:
    def test_reset_returns_controller_to_reset_state(self, controller: ScriptController):
        # Arrange
        _advance_to_idle(controller)

        # Act
        controller.reset()

        # Assert
        assert controller._current_state == BilliardStatus.RESET
