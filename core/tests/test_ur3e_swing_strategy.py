from unittest.mock import MagicMock, patch

import pytest

from core.models.action import Action
from core.services.ur3e_swing_strategy import Ur3eSwingStrategy


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
def strategy(robot_arm: MagicMock, articulation_api: MagicMock) -> Ur3eSwingStrategy:
    return Ur3eSwingStrategy(robot_arm, articulation_api)


TABLE_Z = 0.0
BALL_RADIUS = 0.028575


class TestUr3eSwingStrategyExecuteAim:
    """搬移自 core/services/table_orchestrator.py 舊版
    TestDemoTableOrchestratorUr3eDispatch（`isinstance(self._robot_arm,
    UR3eRobot)` 分流到 ur3e_placement_calculator.py／
    move_swing_elbow_pivot()），跟 WAM7 既有路徑完全分開，見 Strategy
    重構（零行為變化）。"""

    def test_aim_flat_case_uses_flat_placement_calculator(
        self,
        strategy: Ur3eSwingStrategy,
        robot_arm: MagicMock,
        articulation_api: MagicMock,
    ):
        action = _action()
        action.shot_angle = 12.0
        cue_ball = (0.0, 0.0)
        wrist = (0.0, -1.35, 0.028575)
        base_position = (0.3, -2.0, -0.6)
        joint_targets = [1.0, -0.4, -0.8, -0.7, 0.25, 0.0]

        with patch(
            "core.services.ur3e_swing_strategy.cue_pose_calculator.compute_tilted_wrist_pose",
            return_value=(wrist, None, 0.0, None),
        ), patch(
            "core.services.ur3e_swing_strategy.ur3e_placement_calculator.compute_flat_base_position_and_joint_targets",
            return_value=(base_position, joint_targets),
        ) as mock_flat, patch(
            "core.services.ur3e_swing_strategy.ur3e_placement_calculator.compute_bridge_base_position_and_joint_targets",
        ) as mock_bridge:
            strategy.execute_aim(action, cue_ball, TABLE_Z, BALL_RADIUS)

        mock_flat.assert_called_once_with(wrist, 12.0)
        mock_bridge.assert_not_called()
        robot_arm.reposition.assert_called_once_with(base_position)
        articulation_api.move_to_joint_position.assert_called_once_with(joint_targets, list(wrist))

    def test_aim_bridge_case_uses_bridge_placement_calculator(
        self,
        strategy: Ur3eSwingStrategy,
        robot_arm: MagicMock,
        articulation_api: MagicMock,
    ):
        action = _action()
        cue_ball = (0.0, -0.635)
        wrist = (0.0, -1.98, 0.15)
        direction = (0.0, 0.996, -0.093)
        base_position = (0.1, -2.3, -0.7)
        joint_targets = [1.57, -0.4, -0.8, -0.7, 0.25, 0.0]

        with patch(
            "core.services.ur3e_swing_strategy.cue_pose_calculator.compute_tilted_wrist_pose",
            return_value=(wrist, None, 0.093, None),
        ), patch(
            "core.services.ur3e_swing_strategy.cue_pose_calculator.compute_tilted_direction",
            return_value=direction,
        ), patch(
            "core.services.ur3e_swing_strategy.ur3e_placement_calculator.compute_bridge_base_position_and_joint_targets",
            return_value=(base_position, joint_targets),
        ) as mock_bridge, patch(
            "core.services.ur3e_swing_strategy.ur3e_placement_calculator.compute_flat_base_position_and_joint_targets",
        ) as mock_flat:
            strategy.execute_aim(action, cue_ball, TABLE_Z, BALL_RADIUS)

        mock_bridge.assert_called_once_with(wrist, direction, -0.635)
        mock_flat.assert_not_called()
        robot_arm.reposition.assert_called_once_with(base_position)
        articulation_api.move_to_joint_position.assert_called_once_with(joint_targets, list(wrist))


class TestUr3eSwingStrategyExecuteStrike:
    def test_strike_calls_move_swing_elbow_pivot_not_move_swing(
        self,
        strategy: Ur3eSwingStrategy,
        articulation_api: MagicMock,
    ):
        action = _action()
        action.cue_ball_speed = 1.995
        cue_ball = (0.0, 0.0)
        wrist = (0.0, -1.35, 0.028575)
        contact_joint_targets = [1.0, -0.4, -0.8, -0.7, 0.25, 0.0]

        with patch(
            "core.services.ur3e_swing_strategy.cue_pose_calculator.compute_tilted_wrist_pose",
            return_value=(wrist, None, 0.0, None),
        ), patch(
            "core.services.ur3e_swing_strategy.ur3e_placement_calculator.compute_flat_base_position_and_joint_targets",
            return_value=((0.0, 0.0, 0.0), contact_joint_targets),
        ), patch(
            "core.services.ur3e_swing_strategy.ur3e_placement_calculator.compute_flat_target_elbow_velocity",
            return_value=2.5,
        ) as mock_velocity:
            strategy.execute_strike(action, cue_ball, TABLE_Z, BALL_RADIUS)

        mock_velocity.assert_called_once_with(1.995)
        articulation_api.move_swing.assert_not_called()
        articulation_api.move_swing_elbow_pivot.assert_called_once()
        call = articulation_api.move_swing_elbow_pivot.call_args
        backswing_joint_targets, backswing_target_position, contact_targets, elbow_dof_index, target_velocity = call.args
        assert contact_targets == contact_joint_targets
        assert elbow_dof_index == 2
        assert target_velocity == pytest.approx(2.5)
        # 後擺姿態只有 elbow 分量往回轉，其餘分量跟接觸姿態一致（見
        # move_swing_elbow_pivot() 的前提假設）。
        assert backswing_joint_targets[2] == pytest.approx(contact_joint_targets[2] - 0.5236)
        for i in (0, 1, 3, 4, 5):
            assert backswing_joint_targets[i] == pytest.approx(contact_joint_targets[i])
