from dataclasses import dataclass


@dataclass
class Observation:
    """
    控制器的輸入
    """

    joint_positions: list[float]
    end_effector_position: list[float]
    gripper_open: bool
    target_position: list[float]
