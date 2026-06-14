from core.models.action import Action
from core.models.machine_state import MachineState, MachineStatus
from core.models.observation import Observation


class TestMachineState:
    def test_create_valid_machine_state_preserves_status_and_gripper_state(self):
        machine_state = MachineState(
            status=MachineStatus.RUNNING,
            joint_positions=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            end_effector_position=[0.4, 0.0, 0.2],
            gripper_open=True,
        )

        assert machine_state.status is MachineStatus.RUNNING
        assert machine_state.gripper_open is True

    def test_machine_status_enum_values_are_correct(self):
        assert MachineStatus.IDLE.value == "idle"
        assert MachineStatus.RUNNING.value == "running"
        assert MachineStatus.WARNING.value == "warning"
        assert MachineStatus.ERROR.value == "error"


class TestObservation:
    def test_create_valid_observation_has_six_joint_positions_and_bool_gripper_state(self):
        observation = Observation(
            joint_positions=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            end_effector_position=[0.4, 0.0, 0.2],
            gripper_open=False,
            target_position=[0.6, 0.1, 0.2],
        )

        assert len(observation.joint_positions) == 6
        assert isinstance(observation.gripper_open, bool)


class TestAction:
    def test_create_valid_action_has_three_target_position_values_and_bool_gripper_state(self):
        action = Action(
            target_end_effector_position=[0.5, 0.1, 0.3],
            gripper_open=True,
        )

        assert len(action.target_end_effector_position) == 3
        assert isinstance(action.gripper_open, bool)
