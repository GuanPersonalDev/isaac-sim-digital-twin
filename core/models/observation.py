from dataclasses import dataclass


@dataclass
class Observation:
    """
    控制器的輸入
    """

    ball_positions: list[list[float]]
    cue_ball_position: list[float]
    joint_angles: list[float]
    shot_params: list[float]
