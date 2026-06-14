import pytest

from core.controllers.controller_base import ControllerBase
from core.models.action import Action
from core.models.observation import Observation


class IncompleteController(ControllerBase):
    pass


class ConcreteController(ControllerBase):
    def __init__(self):
        self.reset_called = False

    def get_action(self, observation: Observation) -> Action:
        return Action(
            target_end_effector_position=observation.target_position,
            gripper_open=observation.gripper_open,
        )

    def reset(self) -> None:
        self.reset_called = True


def create_observation() -> Observation:
    return Observation(
        joint_positions=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        end_effector_position=[0.4, 0.0, 0.2],
        gripper_open=True,
        target_position=[0.6, 0.1, 0.2],
    )


class TestControllerBase:
    def test_incomplete_subclass_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            IncompleteController()

    def test_concrete_subclass_can_be_instantiated(self):
        controller = ConcreteController()

        assert isinstance(controller, ControllerBase)

    def test_get_action_accepts_observation_and_returns_action(self):
        controller = ConcreteController()
        observation = create_observation()

        action = controller.get_action(observation)

        assert isinstance(action, Action)
        assert action.target_end_effector_position == observation.target_position
        assert action.gripper_open is observation.gripper_open

    def test_reset_can_be_called(self):
        controller = ConcreteController()

        controller.reset()

        assert controller.reset_called is True
