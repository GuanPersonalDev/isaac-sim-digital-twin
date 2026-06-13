from dataclasses import dataclass


@dataclass
class Action:
    """
    output from controller
    """

    target_end_effector_position: list[float]
    gripper_open: bool
