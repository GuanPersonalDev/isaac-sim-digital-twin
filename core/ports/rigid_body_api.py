from abc import ABC, abstractmethod


class RigidBodyAPI(ABC):
    """
    Rigid body 狀態查詢的介面, core 與引擎的中間層(根據不同版本可以有不同的實作內容)
    """

    @abstractmethod
    def get_position(self, prim_path: str) -> list[float]:
        """
        回傳世界座標 (x, y, z) (m)
        """
        ...

    @abstractmethod
    def get_linear_velocity(self, prim_path: str) -> list[float]:
        """
        回傳速度 (vx, vy, vz) (m/s)
        """
        ...

    @abstractmethod
    def get_angular_velocity(self, prim_path: str) -> list[float]:
        """
        回傳角速度 (wx, wy, wz) (rad/s)
        """
        ...
