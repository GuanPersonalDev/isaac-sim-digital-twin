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
