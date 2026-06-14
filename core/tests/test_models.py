from core.models.action import Action
from core.models.billiard_state import BilliardState, BilliardStatus
from core.models.observation import Observation
from core.models.shot_result import ShotResult


class TestBilliardState:
    def test_create_valid_billiard_state_preserves_status_and_ball_state(self):
        billiard_state = BilliardState(
            status=BilliardStatus.AIMING,
            ball_positions=[
                [0.0, 0.0, 0.0],
                [0.1, 0.2, 0.0],
            ],
            cue_ball_position=[-0.3, 0.0, 0.0],
            joint_angles=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        )

        assert billiard_state.status is BilliardStatus.AIMING
        assert billiard_state.ball_positions[1] == [0.1, 0.2, 0.0]
        assert billiard_state.cue_ball_position == [-0.3, 0.0, 0.0]

    def test_billiard_status_enum_values_are_correct(self):
        assert BilliardStatus.IDLE.value == "idle"
        assert BilliardStatus.AIMING.value == "aiming"
        assert BilliardStatus.SHOOTING.value == "shooting"
        assert BilliardStatus.RESETTING.value == "resetting"


class TestObservation:
    def test_create_valid_observation_preserves_billiard_inputs(self):
        observation = Observation(
            ball_positions=[
                [0.0, 0.0, 0.0],
                [0.1, 0.2, 0.0],
            ],
            cue_ball_position=[-0.3, 0.0, 0.0],
            joint_angles=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            shot_params=[1.5, 35.0, 0.02],
        )

        assert len(observation.ball_positions) == 2
        assert len(observation.joint_angles) == 6
        assert observation.shot_params == [1.5, 35.0, 0.02]


class TestAction:
    def test_create_valid_action_preserves_shot_command(self):
        action = Action(
            cue_speed=2.5,
            shot_angle=42.0,
            position_offset=[0.01, -0.02, 0.0],
        )

        assert action.cue_speed == 2.5
        assert action.shot_angle == 42.0
        assert len(action.position_offset) == 3


class TestShotResult:
    def test_create_valid_shot_result_preserves_result_metrics(self):
        shot_result = ShotResult(
            final_ball_positions=[
                [0.0, 0.0, 0.0],
                [0.4, 0.2, 0.0],
            ],
            cue_ball_pocketed=False,
            nine_ball_pocketed=True,
            spread_score=0.87,
        )

        assert len(shot_result.final_ball_positions) == 2
        assert shot_result.cue_ball_pocketed is False
        assert shot_result.nine_ball_pocketed is True
        assert shot_result.spread_score == 0.87
