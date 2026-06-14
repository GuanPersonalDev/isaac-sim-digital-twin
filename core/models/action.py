from dataclasses import dataclass


@dataclass
class Action:
    """
    控制器的輸出
    """

    cue_speed: float
    shot_angle: float
    position_offset: list[float]
