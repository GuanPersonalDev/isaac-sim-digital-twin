from abc import ABC, abstractmethod

class PolicyPort(ABC):
    """
    RL policy 的介面
    """

    @abstractmethod
    def infer(self, observation: list[float]) -> list[float]:
        """
        Args:
            observation: 觀測值
        Returns:
            action: 動作
                - cue_ball_placement_x: 白球放置位置的 x 座標
                - cue_ball_placement_y: 白球放置位置的 y 座標
                - shot_angle: 球桿角度
                - cue_ball_speed: 白球速度
                - position_offset_vertical: 白球位置偏移的垂直方向
                - position_offset_horizontal: 白球位置偏移的水平方向

        回傳值需要再送進 core.services.rl_action_decoder.decode_rl_action()
        """
        ...