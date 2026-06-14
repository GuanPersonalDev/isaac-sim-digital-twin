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

    def test_create_with_empty_ball_positions_preserves_empty_list(self):
        billiard_state = BilliardState(
            status=BilliardStatus.IDLE,
            ball_positions=[],
            cue_ball_position=[0.0, 0.0, 0.0],
            joint_angles=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        )

        assert billiard_state.ball_positions == []

    def test_create_with_six_and_seven_joint_angles_preserves_input_lengths(self):
        six_axis_state = BilliardState(
            status=BilliardStatus.AIMING,
            ball_positions=[],
            cue_ball_position=[0.0, 0.0, 0.0],
            joint_angles=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        )
        seven_axis_state = BilliardState(
            status=BilliardStatus.AIMING,
            ball_positions=[],
            cue_ball_position=[0.0, 0.0, 0.0],
            joint_angles=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        )

        assert len(six_axis_state.joint_angles) == 6
        assert len(seven_axis_state.joint_angles) == 7


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

    def test_create_with_empty_shot_params_preserves_empty_list(self):
        observation = Observation(
            ball_positions=[[0.0, 0.0, 0.0]],
            cue_ball_position=[-0.3, 0.0, 0.0],
            joint_angles=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            shot_params=[],
        )

        assert observation.shot_params == []

    def test_create_with_empty_ball_positions_preserves_empty_list(self):
        observation = Observation(
            ball_positions=[],
            cue_ball_position=[-0.3, 0.0, 0.0],
            joint_angles=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            shot_params=[1.5, 35.0, 0.02],
        )

        assert observation.ball_positions == []


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

    def test_create_with_position_offset_preserves_three_values(self):
        action = Action(
            cue_speed=2.5,
            shot_angle=42.0,
            position_offset=[0.01, -0.02, 0.0],
        )

        assert action.position_offset == [0.01, -0.02, 0.0]

    def test_create_with_zero_cue_speed_preserves_boundary_value(self):
        action = Action(
            cue_speed=0.0,
            shot_angle=42.0,
            position_offset=[0.0, 0.0, 0.0],
        )

        assert action.cue_speed == 0.0


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

    def test_create_with_zero_spread_score_preserves_boundary_value(self):
        shot_result = ShotResult(
            final_ball_positions=[[0.0, 0.0, 0.0]],
            cue_ball_pocketed=False,
            nine_ball_pocketed=False,
            spread_score=0.0,
        )

        assert shot_result.spread_score == 0.0

    def test_create_with_full_spread_score_preserves_boundary_value(self):
        shot_result = ShotResult(
            final_ball_positions=[[0.0, 0.0, 0.0]],
            cue_ball_pocketed=False,
            nine_ball_pocketed=True,
            spread_score=1.0,
        )

        assert shot_result.spread_score == 1.0

    def test_create_with_empty_final_ball_positions_preserves_empty_list(self):
        shot_result = ShotResult(
            final_ball_positions=[],
            cue_ball_pocketed=False,
            nine_ball_pocketed=False,
            spread_score=0.5,
        )

        assert shot_result.final_ball_positions == []
