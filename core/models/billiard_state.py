from dataclasses import dataclass
from enum import Enum


class BilliardStatus(Enum):
    IDLE = "idle"
    AIMING = "aiming"
    STRIKING = "striking"
    WAITING = "waiting"
    RESET = "reset"
    ERROR = "error"


@dataclass
class BilliardState:
    status: BilliardStatus
    ball_positions: list[list[float]]
    cue_ball_position: list[float]
    joint_angles: list[float]
