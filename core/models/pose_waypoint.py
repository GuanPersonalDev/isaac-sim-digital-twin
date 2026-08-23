from dataclasses import dataclass, field


@dataclass(frozen=True)
class PoseWaypoint:
    """單一 Cartesian pose 目標，供 `ArticulationAPI.move_through_poses()` 依序播放。

    position: [x, y, z]
    orientation: [qw, qx, qy, qz]
    linear_velocity/angular_velocity: 抵達這個 waypoint 時要逼近的末端速度，
    預設 0（後擺姿態用）；擊球接觸姿態會帶非零值（見 swing_trajectory_calculator）。
    """

    position: list[float]
    orientation: list[float]
    linear_velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angular_velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
