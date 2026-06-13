from dataclasses import dataclass


@dataclass
class Observation:
    """
    input for controller
    """

    joint_positions: list[float]
    end_effector_position: list[float]
    gripper_open: bool
    target_position: list[float]
