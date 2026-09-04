from unittest.mock import MagicMock, patch

import pytest

from core.models.action import Action
from core.services import swing_trajectory_calculator
from core.services.ur10e_swing_strategy import Ur10eSwingStrategy


def _action(should_execute_action: bool = True) -> Action:
    return Action(
        cue_ball_speed=0.0,
        shot_angle=0.0,
        position_offset=[0.0, 0.0],
        cue_ball_placement=[0.0, 0.0],
        should_execute_action=should_execute_action,
    )


@pytest.fixture
def robot_arm() -> MagicMock:
    return MagicMock()


@pytest.fixture
def articulation_api() -> MagicMock:
    return MagicMock()


@pytest.fixture
def strategy(robot_arm: MagicMock, articulation_api: MagicMock) -> Ur10eSwingStrategy:
    return Ur10eSwingStrategy(robot_arm, articulation_api)


TABLE_Z = 0.0
BALL_RADIUS = 0.028575


class TestUr10eSwingStrategyExecuteAim:
    def test_repositions_base_and_moves_to_pose_via_rmpflow(
        self,
        strategy: Ur10eSwingStrategy,
        robot_arm: MagicMock,
        articulation_api: MagicMock,
    ):
        action = _action()
        action.shot_angle = 12.0
        cue_ball = (0.0, -0.752)
        wrist = (-0.036, -2.093, 0.181)
        orientation = (0.998, -0.057, 0.0, 0.0)
        direction = (0.0, -0.985, 0.172)
        base_position = (-0.036, -2.593, 0.0)
        current_orientation = (1.0, 0.0, 0.0, 0.0)
        best_roll_rad = 3.14159

        articulation_api.get_end_effector_orientation.return_value = current_orientation

        with patch(
            "core.services.ur10e_swing_strategy.cue_pose_calculator.compute_tilted_wrist_pose",
            return_value=(wrist, orientation, 0.1134, None),
        ) as mock_compute_wrist, patch(
            "core.services.ur10e_swing_strategy.cue_pose_calculator.compute_tilted_direction",
            return_value=direction,
        ), patch(
            "core.services.ur10e_swing_strategy.ur10e_placement_calculator.compute_roll_minimizing_reorientation",
            return_value=best_roll_rad,
        ) as mock_compute_roll, patch(
            "core.services.ur10e_swing_strategy.ur10e_placement_calculator.compute_base_position",
            return_value=base_position,
        ) as mock_compute_base:
            strategy.execute_aim(action, cue_ball, TABLE_Z, BALL_RADIUS)

        mock_compute_roll.assert_called_once_with(
            cue_ball, 12.0, TABLE_Z, BALL_RADIUS, action.position_offset, current_orientation, base_position
        )
        # roll_rad 是找到最貼近目前姿態的旋轉自由度後，重算一次
        # compute_tilted_wrist_pose() 套用（見 execute_aim() 除錯註解），
        # 呼叫兩次：第一次（roll_rad 預設 0）只為了拿 tilt_rad 判斷幾何
        # 是否有解，第二次才是真正套用 roll_rad 的版本。
        assert mock_compute_wrist.call_count == 2
        second_call_kwargs = mock_compute_wrist.call_args_list[1].kwargs
        assert second_call_kwargs["roll_rad"] == best_roll_rad
        mock_compute_base.assert_called_once_with(wrist, direction, TABLE_Z)
        robot_arm.reposition.assert_called_once_with(base_position)
        articulation_api.set_robot_base_pose.assert_called_once_with(
            list(base_position), [1.0, 0.0, 0.0, 0.0]
        )
        articulation_api.move_to_pose.assert_called_once_with(list(wrist), list(orientation))

    def test_infeasible_geometry_raises(
        self, strategy: Ur10eSwingStrategy, articulation_api: MagicMock
    ):
        action = _action()
        cue_ball = (0.0, 0.0)
        articulation_api.get_end_effector_orientation.return_value = (1.0, 0.0, 0.0, 0.0)

        with patch(
            "core.services.ur10e_swing_strategy.cue_pose_calculator.compute_tilted_wrist_pose",
            return_value=(None, None, None, (0.0, -1.295)),
        ):
            with pytest.raises(ValueError):
                strategy.execute_aim(action, cue_ball, TABLE_Z, BALL_RADIUS)


class TestUr10eSwingStrategyExecuteStrike:
    def test_strikes_via_cue_slide_with_backswing_and_required_tip_speed(
        self,
        strategy: Ur10eSwingStrategy,
        articulation_api: MagicMock,
    ):
        action = _action()
        action.cue_ball_speed = 1.995
        cue_ball = (0.0, -0.752)

        with patch(
            "core.services.ur10e_swing_strategy.swing_trajectory_calculator.compute_required_tip_speed",
            return_value=1.5116,
        ) as mock_required_speed:
            strategy.execute_strike(action, cue_ball, TABLE_Z, BALL_RADIUS)

        mock_required_speed.assert_called_once_with(1.995)
        articulation_api.move_cue_slide_stroke.assert_called_once_with(
            -swing_trajectory_calculator.DEFAULT_BACKSWING_DISTANCE_M, 1.5116
        )

    def test_does_not_recompute_wrist_geometry(
        self,
        strategy: Ur10eSwingStrategy,
        articulation_api: MagicMock,
    ):
        """UR10e 揮桿不像 WAM7/UR3e 需要重新算 wrist/方向幾何——滑軌關節軸向
        就是桿尖速度方向，AIM 收斂後手臂完全靜止，不需要槓桿臂換算。"""
        action = _action()
        cue_ball = (0.0, -0.752)

        with patch(
            "core.services.ur10e_swing_strategy.cue_pose_calculator.compute_tilted_wrist_pose"
        ) as mock_compute_pose:
            strategy.execute_strike(action, cue_ball, TABLE_Z, BALL_RADIUS)

        mock_compute_pose.assert_not_called()
