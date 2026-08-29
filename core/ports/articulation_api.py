from abc import ABC, abstractmethod

from ..models.pose_waypoint import PoseWaypoint


class ArticulationAPI(ABC):
    """
    使用omniverse 或 isaac api 的抽象依賴設計，讓core不直接觸碰引擎 api ，當需要升級或改版時直接替換 implement
    """

    @abstractmethod
    def initialize(self) -> None:
        """
        初始化，場景載入後須呼叫一次
        """
        ...

    @abstractmethod
    def move_to_pose(self, position: list[float], orientation: list[float], linear_velocity: list[float] = [0.0, 0.0, 0.0], angular_velocity: list[float] = [0.0, 0.0, 0.0]) -> None:
        """
        移動末端到目標位姿, 抵達時逼近末端速度
        position: [x, y, z]
        orientation: [qw, qx, qy, qz]
        linear_velocity: [vx, vy, vz]
        angular_velocity: [wx, wy, wz]
        """
        ...

    @abstractmethod
    def move_through_poses(
        self,
        waypoints: list[PoseWaypoint],
        preceding_joint_targets: tuple[list[float], list[float]] | None = None,
    ) -> None:
        """
        依序移動末端通過一串 Cartesian pose 目標，內部自我驅動、自我轉換
        階段，呼叫端只需要呼叫一次。只有走到最後一個 waypoint 才視為
        「動作完成」，is_motion_complete() 在整段序列播放期間持續回傳
        False，語意跟 move_to_pose() 一致。

        waypoints: 至少 1 個 PoseWaypoint，依序播放。
        preceding_joint_targets: 若不為 None，格式為
        (joint_positions, target_end_effector_position)，會先用
        joint-space 動作收斂到這組姿態（避開差動 IK 在奇異點附近的失穩
        問題），收斂後才開始播放 waypoints。
        """
        ...

    @abstractmethod
    def move_swing(
        self,
        backswing_position: list[float],
        orientation: list[float],
        swing_end_position: list[float],
        orientation_gain: float = 1.0,
        max_angular_speed: float = 0.5,
    ) -> None:
        """
        揮桿專用速度最優控制：先移動到 backswing_position（姿態鎖死，跟
        move_to_pose 一樣）收斂後，自動切換成揮桿模式，沿直線移動到
        swing_end_position，姿態修正角速度限制在 max_angular_speed 內
        （orientation_gain 控制修正力道），不強制全程鎖死姿態，換取更高
        的可達平移速度。呼叫端只需呼叫一次，is_motion_complete() 在後擺+
        揮桿全程持續回傳 False。

        見 docs/issue-180-reachability-analysis.md 第十六節：一般
        move_to_pose/move_through_poses 的姿態鎖死策略，對某些案例會讓
        可達平移速度大幅低於運動學理論上限。
        """
        ...

    @abstractmethod
    def move_to_home(self) -> None:
        """
        回到待機姿態
        """
        ...

    @abstractmethod
    def get_end_effector_position(self) -> list[float]:
        """
        取得末端當前位置[x, y, z]
        """
        ...

    @abstractmethod
    def get_end_effector_orientation(self) -> list[float]:
        """
        取得末端當前朝向 [qw, qx, qy, qz]
        """
        ...

    @abstractmethod
    def is_motion_complete(self) -> bool:
        """
        是否已經達到目標
        """
        ...

    @abstractmethod
    def did_last_motion_timeout(self) -> bool:
        """
        最近一次動作是否是因為超過步數上限被強制視為完成（而不是真的收斂），
        呼叫端應視為「可能帶著誤差」，不能直接當成正常完成處理。
        """
        ...

    @abstractmethod
    def shutdown(self) -> None:
        """
        release process
        """
        ...

    @abstractmethod
    def cancel_pending_home_capture(self) -> None:
        """
        取消尚未觸發的一次性 home-capture callback（若存在）。
        initialize() 從未被呼叫、或 callback 已經觸發過，皆為 no-op。
        """
        ...

    @abstractmethod
    def move_to_joint_position(self, joint_positions: list[float], target_end_effector_position: list[float]) -> None:
        """
        joint_positions: 各關節角度[
        [
            base_yaw,
            shoulder_pitch,
            shoulder_yaw, 
            elbow_pitch,
            wrist_yaw,
            wrist_pitch,
            palm_yaw
        ]
        target_end_effector_position: 末端目標位置[x, y, z]

        """
        ...