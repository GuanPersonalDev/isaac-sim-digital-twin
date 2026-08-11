from dataclasses import dataclass


@dataclass
class ShotResult:
    final_ball_positions: list[list[float]]
    cue_ball_pocketed: bool
    nine_ball_pocketed: bool
    spread_score: float
    # 母球球面與 1 號球球面在**首次接觸之前**的最小距離（m），dense shaping 用。
    # 0.0 = 碰到了。
    #
    # 預設 `math.inf`（= 沒有量到，塑形給 0）而不是 0.0：0.0 的語意是「碰到了」，
    # 也就是塑形滿分。舊呼叫端沒帶這個欄位時給滿分，是**完全不報錯**地把
    # reward 灌水；給 inf 則退回改動前的行為，安全的預設要往這個方向倒。
    closest_approach: float = float("inf")
