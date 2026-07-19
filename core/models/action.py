from dataclasses import dataclass


@dataclass
class Action:
    """
    控制器的輸出
    """

    cue_speed: float
    shot_angle: float
    position_offset: list[float] # 2維, 對應白球表面的加塞位置
    cue_ball_placement: list[float] # 2 維, 白球的 X, Y 位置
    should_control_articulation: bool
