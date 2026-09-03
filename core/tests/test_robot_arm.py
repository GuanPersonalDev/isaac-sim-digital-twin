from unittest.mock import MagicMock

import pytest

from core.models.robot_arm import RobotArm


class IncompleteRobotArm(RobotArm):
    pass


class ConcreteRobotArm(RobotArm):
    def __init__(self):
        self.reset_called = False

    @staticmethod
    def get_prim_path(base_path: str) -> str:
        return base_path + "/Robot"

    @staticmethod
    def get_end_effector_prim_path(base_path: str) -> str:
        return base_path + "/Robot/tip"

    def reset(self) -> None:
        self.reset_called = True

    def is_reset_complete(self) -> bool:
        return True

    def reposition(self, position: tuple[float, float, float]) -> None:
        self.reposition_called_with = position


class TestRobotArm:
    def test_incomplete_subclass_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            IncompleteRobotArm()

    def test_concrete_subclass_can_be_instantiated(self):
        robot = ConcreteRobotArm()

        assert isinstance(robot, RobotArm)

    def test_ur5_robot_is_a_robot_arm(self):
        from core.models.ur5_robot import UR5Robot

        robot = UR5Robot(
            base_path="/World/BilliardTable",
            stage_api=MagicMock(),
            articulation_api=MagicMock(),
            position=(1.5, 0.0, 0.0),
        )

        assert isinstance(robot, RobotArm)

    def test_barrett_wam_robot_is_a_robot_arm(self):
        from core.models.barrett_wam_robot import BarrettWamRobot

        robot = BarrettWamRobot(
            base_path="/World/BilliardTable",
            stage_api=MagicMock(),
            articulation_api=MagicMock(),
            position=(1.5, 0.0, 0.0),
        )

        assert isinstance(robot, RobotArm)

    def test_ur3e_robot_is_a_robot_arm(self):
        from core.models.ur3e_robot import UR3eRobot

        robot = UR3eRobot(
            base_path="/World/BilliardTable",
            stage_api=MagicMock(),
            articulation_api=MagicMock(),
            position=(1.5, 0.0, 0.0),
        )

        assert isinstance(robot, RobotArm)
