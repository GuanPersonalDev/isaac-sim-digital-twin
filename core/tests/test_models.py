from unittest.mock import MagicMock, patch

from core.models.action import Action
from core.models.billiard_state import BilliardState, BilliardStatus
from core.models.billiard_table import BilliardTable
from core.models.observation import Observation
from core.models.shot_result import ShotResult
from core.models.table_ball_set import TableBallSet


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
        assert BilliardStatus.STRIKING.value == "striking"
        assert BilliardStatus.WAITING.value == "waiting"
        assert BilliardStatus.RESET.value == "reset"
        assert BilliardStatus.ERROR.value == "error"

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
            is_init_state=False,
            is_ball_moving=False,
            is_motion_complete=False,
            has_error=False,
        )

        assert len(observation.ball_positions) == 2

    def test_create_with_empty_ball_positions_preserves_empty_list(self):
        observation = Observation(
            ball_positions=[],
            cue_ball_position=[-0.3, 0.0, 0.0],
            is_init_state=False,
            is_ball_moving=False,
            is_motion_complete=False,
            has_error=False,
        )

        assert observation.ball_positions == []


class TestAction:
    def test_create_valid_action_preserves_shot_command(self):
        action = Action(
            cue_speed=2.5,
            shot_angle=42.0,
            position_offset=[0.01, -0.02],
            cue_ball_placement=[0.1, -0.2],
            should_execute_action=False,
        )

        assert action.cue_speed == 2.5
        assert action.shot_angle == 42.0
        assert len(action.position_offset) == 2

    def test_create_with_position_offset_preserves_two_values(self):
        action = Action(
            cue_speed=2.5,
            shot_angle=42.0,
            position_offset=[0.01, -0.02],
            cue_ball_placement=[0.1, -0.2],
            should_execute_action=False,
        )

        assert action.position_offset == [0.01, -0.02]

    def test_create_with_zero_cue_speed_preserves_boundary_value(self):
        action = Action(
            cue_speed=0.0,
            shot_angle=42.0,
            position_offset=[0.0, 0.0],
            cue_ball_placement=[0.0, 0.0],
            should_execute_action=False,
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


class TestTableBallSetBallRadius:
    def test_get_ball_radius_returns_constructed_value(self):
        table_ball_set = TableBallSet(
            stage_api=MagicMock(),
            material_api=MagicMock(),
            rigid_body_api=MagicMock(),
            table_z=0.75,
            base_path="/World/BilliardTable_0",
            ball_radius=0.03,
        )

        assert table_ball_set.get_ball_radius() == 0.03


class TestBilliardTableGetTableBallSet:
    def test_billiard_table_get_table_ball_set_returns_internal_instance(self):
        with (
            patch("core.models.billiard_table.TableBallSet") as table_ball_set_class,
            patch(
                "core.models.billiard_table.BreakShotPositionProvider"
            ) as position_provider_class,
        ):
            position_provider_class.return_value.get_positions.return_value = {
                ball_id: (0.0, 0.0) for ball_id in range(10)
            }

            billiard_table = BilliardTable(
                base_path="/World/BilliardTable",
                stage_api=MagicMock(),
                material_api=MagicMock(),
                rigid_body_api=MagicMock(),
                position=(0.0, 0.0),
            )

            assert billiard_table.get_table_ball_set() is table_ball_set_class.return_value
