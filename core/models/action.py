from dataclasses import dataclass


@dataclass
class Action:
    """
    控制器的輸出。

    RL 使用的物理域 6 維順序為：
    cue_ball_placement[0:2]、shot_angle、cue_ball_speed、
    position_offset[0:2]。

    cue_ball_placement 是桌台相對 XY（m）；shot_angle 是以桌台 +Y 為
    0 度、朝 -X 增加的水平角度（degree）；cue_ball_speed 是母球目標
    初速（m/s）；position_offset 依序是上下、左右擊球偏移，數值為球
    半徑比例。

    should_execute_action 是執行期控制旗標，不屬於 RL 6 維向量。執行期
    no-op Action 可使用 0.0 母球初速；RL action space 的範圍與裁切由
    BilliardEnv 負責。
    """

    cue_ball_placement: list[float]
    shot_angle: float
    cue_ball_speed: float
    position_offset: list[float]
    should_execute_action: bool
