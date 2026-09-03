from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.models.action import Action
from core.models.pose_waypoint import PoseWaypoint
from core.services.wam7_swing_strategy import Wam7SwingStrategy


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
    api = MagicMock()
    api.did_last_motion_timeout.return_value = False
    return api


@pytest.fixture
def strategy(robot_arm: MagicMock, articulation_api: MagicMock) -> Wam7SwingStrategy:
    return Wam7SwingStrategy(robot_arm, articulation_api)


TABLE_Z = 0.0
BALL_RADIUS = 0.028575


class TestWam7SwingStrategyExecuteAim:
    """`execute_aim` 依 `cue_pose_calculator.compute_tilted_wrist_pose()` 判定
    的 tilt_rad 分支：flat（<=1e-6）走 joint-space，高架橋（>0）走
    `move_through_poses`，職責分離——策略不自己算幾何，只轉交 calculator
    算好的結果給 port。搬移自 core/services/table_orchestrator.py 舊版
    _execute_aim() 的 WAM7 分支測試（見 Strategy 重構，零行為變化）。"""

    def test_flat_case_calls_move_to_joint_position_not_move_through_poses(
        self,
        strategy: Wam7SwingStrategy,
        robot_arm: MagicMock,
        articulation_api: MagicMock,
    ):
        action = _action()
        action.shot_angle = 0.0
        cue_ball = (0.0, 0.3)

        with patch(
            "core.services.wam7_swing_strategy.cue_pose_calculator.compute_tilted_wrist_pose",
            return_value=(None, None, 0.0, None),
        ):
            strategy.execute_aim(action, cue_ball, TABLE_Z, BALL_RADIUS)

        robot_arm.reposition.assert_called_once()
        articulation_api.move_to_joint_position.assert_called_once()
        articulation_api.move_through_poses.assert_not_called()

    def test_bridge_case_calls_move_through_poses_with_preceding_joint_targets(
        self,
        strategy: Wam7SwingStrategy,
        robot_arm: MagicMock,
        articulation_api: MagicMock,
    ):
        articulation_api.get_end_effector_position.return_value = [0.0, 0.0, 1.0]
        articulation_api.get_end_effector_orientation.return_value = [1.0, 0.0, 0.0, 0.0]
        action = _action()
        action.shot_angle = 0.0
        cue_ball = (0.0, 0.0)
        bridge_waypoints = [
            PoseWaypoint(position=[0.0, 0.0, 1.0], orientation=[1.0, 0.0, 0.0, 0.0]),
            PoseWaypoint(position=[0.0, 0.0, 0.5], orientation=[1.0, 0.0, 0.0, 0.0]),
        ]

        with patch(
            "core.services.wam7_swing_strategy.cue_pose_calculator.compute_tilted_wrist_pose",
            return_value=(None, None, 0.05, None),
        ), patch(
            "core.services.wam7_swing_strategy.cue_pose_calculator.compute_elevated_bridge_waypoints",
            return_value=bridge_waypoints,
        ):
            strategy.execute_aim(action, cue_ball, TABLE_Z, BALL_RADIUS)

        robot_arm.reposition.assert_called_once()
        articulation_api.move_to_joint_position.assert_not_called()
        articulation_api.move_through_poses.assert_called_once()
        call_kwargs = articulation_api.move_through_poses.call_args
        assert call_kwargs.args[0] == bridge_waypoints
        assert call_kwargs.kwargs["preceding_joint_targets"] is not None

    def test_infeasible_geometry_raises(self, strategy: Wam7SwingStrategy):
        action = _action()
        cue_ball = (0.0, 0.0)

        with patch(
            "core.services.wam7_swing_strategy.cue_pose_calculator.compute_tilted_wrist_pose",
            return_value=(None, None, None, (0.0, -1.295)),
        ):
            with pytest.raises(ValueError):
                strategy.execute_aim(action, cue_ball, TABLE_Z, BALL_RADIUS)

    def test_infeasible_bridge_geometry_raises(
        self,
        strategy: Wam7SwingStrategy,
        articulation_api: MagicMock,
    ):
        articulation_api.get_end_effector_position.return_value = [0.0, 0.0, 1.0]
        articulation_api.get_end_effector_orientation.return_value = [1.0, 0.0, 0.0, 0.0]
        action = _action()
        cue_ball = (0.0, 0.0)

        with patch(
            "core.services.wam7_swing_strategy.cue_pose_calculator.compute_tilted_wrist_pose",
            return_value=(None, None, 0.05, None),
        ), patch(
            "core.services.wam7_swing_strategy.cue_pose_calculator.compute_elevated_bridge_waypoints",
            return_value=None,
        ):
            with pytest.raises(ValueError):
                strategy.execute_aim(action, cue_ball, TABLE_Z, BALL_RADIUS)


class TestWam7SwingStrategyExecuteStrike:
    def test_calls_move_swing_with_calculator_backswing_and_follow_through(
        self,
        strategy: Wam7SwingStrategy,
        articulation_api: MagicMock,
    ):
        action = _action()
        action.cue_ball_speed = 2.0
        cue_ball = (0.0, 0.0)
        wrist = np.array([0.0, -1.35, 0.028575])
        orientation = np.array([1.0, 0.0, 0.0, 0.0])
        direction = np.array([0.0, 1.0, 0.0])
        backswing = np.array([0.0, -1.5, 0.028575])

        with patch(
            "core.services.wam7_swing_strategy.cue_pose_calculator.compute_tilted_wrist_pose",
            return_value=(wrist, orientation, 0.0, None),
        ), patch(
            "core.services.wam7_swing_strategy.cue_pose_calculator.compute_tilted_direction",
            return_value=direction,
        ), patch(
            "core.services.wam7_swing_strategy.swing_trajectory_calculator.compute_required_tip_speed",
            return_value=1.5,
        ), patch(
            "core.services.wam7_swing_strategy.swing_trajectory_calculator.compute_follow_through_distance",
            return_value=0.03,
        ), patch(
            "core.services.wam7_swing_strategy.swing_trajectory_calculator.compute_backswing_position",
            return_value=backswing,
        ) as mock_backswing:
            strategy.execute_strike(action, cue_ball, TABLE_Z, BALL_RADIUS)

        mock_backswing.assert_called_once()
        articulation_api.move_swing.assert_called_once()
        call = articulation_api.move_swing.call_args
        assert call.args[0] == pytest.approx(backswing.tolist())
        assert call.args[1] == pytest.approx(orientation.tolist())
        # follow_through = wrist + follow_through_distance(0.03) * direction([0,1,0])
        assert call.args[2] == pytest.approx([0.0, -1.32, 0.028575])
        assert call.kwargs["orientation_gain"] == pytest.approx(1.0)
        assert call.kwargs["max_angular_speed"] == pytest.approx(1.0)

    def test_bridge_case_uses_looked_up_backswing_distance(
        self,
        strategy: Wam7SwingStrategy,
        articulation_api: MagicMock,
    ):
        """tilt_rad>1e-6（高架橋案例）必須用
        cue_pose_calculator.lookup_backswing_distance_m() 查表算後擺距離，
        不是 flat 案例才用的 DEFAULT_BACKSWING_DISTANCE_M——見 2026-09-01
        的 IK 可達邊界法修法（docs/issue-180-reachability-analysis.md
        第十八節「待處理 B」）。"""
        action = _action()
        action.cue_ball_speed = 2.0
        cue_ball = (0.0, -0.9382125)
        wrist = np.array([0.0, -1.35, 0.18])
        orientation = np.array([0.998, -0.057, 0.0, 0.0])
        direction = np.array([0.0, 0.985, -0.172])
        backswing = np.array([0.0, -1.5, 0.5])

        with patch(
            "core.services.wam7_swing_strategy.cue_pose_calculator.compute_tilted_wrist_pose",
            return_value=(wrist, orientation, 0.173, None),
        ), patch(
            "core.services.wam7_swing_strategy.cue_pose_calculator.lookup_roll_rad",
            return_value=0.0,
        ), patch(
            "core.services.wam7_swing_strategy.cue_pose_calculator.lookup_backswing_distance_m",
            return_value=0.35,
        ) as mock_lookup_distance, patch(
            "core.services.wam7_swing_strategy.cue_pose_calculator.compute_tilted_direction",
            return_value=direction,
        ), patch(
            "core.services.wam7_swing_strategy.swing_trajectory_calculator.compute_required_tip_speed",
            return_value=1.5,
        ), patch(
            "core.services.wam7_swing_strategy.swing_trajectory_calculator.compute_follow_through_distance",
            return_value=0.03,
        ), patch(
            "core.services.wam7_swing_strategy.swing_trajectory_calculator.compute_backswing_position",
            return_value=backswing,
        ) as mock_backswing:
            strategy.execute_strike(action, cue_ball, TABLE_Z, BALL_RADIUS)

        mock_lookup_distance.assert_called_once_with((0.0, -0.9382125))
        mock_backswing.assert_called_once()
        assert mock_backswing.call_args.args[2] == pytest.approx(0.35)

    def test_infeasible_geometry_raises(self, strategy: Wam7SwingStrategy):
        action = _action()
        cue_ball = (0.0, 0.0)

        with patch(
            "core.services.wam7_swing_strategy.cue_pose_calculator.compute_tilted_wrist_pose",
            return_value=(None, None, None, (0.0, -1.295)),
        ):
            with pytest.raises(ValueError):
                strategy.execute_strike(action, cue_ball, TABLE_Z, BALL_RADIUS)
