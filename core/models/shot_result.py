from dataclasses import dataclass


@dataclass
class ShotResult:
    final_ball_positions: list[list[float]]
    cue_ball_pocketed: bool
    nine_ball_pocketed: bool
    spread_score: float
