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
    is_init_state: bool
    is_ball_moving: bool
    is_motion_complete: bool
    has_error: bool
