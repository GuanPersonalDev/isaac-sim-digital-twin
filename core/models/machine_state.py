from dataclasses import dataclass
from enum import Enum


class MachineStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class MachineState:
    status: MachineStatus
    joint_positions: list[float]
    end_effector_position: list[float]
    gripper_open: bool
