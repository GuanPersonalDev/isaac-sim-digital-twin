from dataclasses import dataclass


@dataclass
class Action:
    """
    控制器的輸出
    """

    target_end_effector_position: list[float]
    gripper_open: bool
