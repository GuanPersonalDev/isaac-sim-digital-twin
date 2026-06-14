import importlib.util
from pathlib import Path

import pytest

from core.models.action import Action
from core.models.observation import Observation


_CONTROLLER_BASE_PATH = (
    Path(__file__).resolve().parents[1] / "controllers" / "controller_base.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "controller_base_for_test",
    _CONTROLLER_BASE_PATH,
)
_CONTROLLER_BASE_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_CONTROLLER_BASE_MODULE)
ControllerBase = _CONTROLLER_BASE_MODULE.ControllerBase


class IncompleteController(ControllerBase):
    pass


class ConcreteController(ControllerBase):
    def __init__(self):
        self.reset_called = False

    def get_action(self, observation: Observation) -> Action:
        return Action(
            cue_speed=observation.shot_params[0],
            shot_angle=observation.shot_params[1],
            position_offset=[observation.shot_params[2], 0.0, 0.0],
        )

    def reset(self) -> None:
        self.reset_called = True


@pytest.fixture
def observation() -> Observation:
    return Observation(
        ball_positions=[
            [0.0, 0.0, 0.0],
            [0.1, 0.2, 0.0],
        ],
        cue_ball_position=[-0.3, 0.0, 0.0],
        joint_angles=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        shot_params=[2.5, 42.0, 0.01],
    )


class TestControllerBase:
    def test_incomplete_subclass_cannot_be_instantiated(self):
        # Arrange / Act / Assert
        with pytest.raises(TypeError):
            IncompleteController()

    def test_concrete_subclass_can_be_instantiated(self):
        # Arrange / Act
        controller = ConcreteController()

        # Assert
        assert isinstance(controller, ControllerBase)

    def test_get_action_accepts_observation_and_returns_action(
        self,
        observation: Observation,
    ):
        # Arrange
        controller = ConcreteController()

        # Act
        action = controller.get_action(observation)

        # Assert
        assert isinstance(action, Action)
        assert action.cue_speed == observation.shot_params[0]
        assert action.shot_angle == observation.shot_params[1]
        assert action.position_offset == [observation.shot_params[2], 0.0, 0.0]

    def test_reset_can_be_called(self):
        # Arrange
        controller = ConcreteController()

        # Act
        controller.reset()

        # Assert
        assert controller.reset_called is True
