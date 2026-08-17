from abc import ABC, abstractmethod


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
    def move_to_pose(self, position: list[float], orientation: list[float]) -> None:
        """
        移動末端到目標位姿
        position: [x, y, z]
        orientation: [qw, qx, qy, qz]
        """
        ...

    @abstractmethod
    def execute_strike(
        self, direction: list[float], distance: float, speed: float
    ) -> None:
        """
        沿指定方向擊球
        direction: [x, y, z]
        distance: 擊球距離
        speed: 擊球速度
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
    def is_motion_complete(self) -> bool:
        """
        是否已經達到目標
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