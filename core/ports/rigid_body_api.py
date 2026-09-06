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
    def set_position(self, prim_path: str, x: float, y: float, z: float) -> None:
        """
        Teleport 設定世界座標 (x, y, z) (m)。

        必須透過 rigid body 的物理視圖（跟 get_position/get_linear_velocity/
        set_velocities 同一條路徑）設定位置，不能用場景圖層級的 raw transform
        寫入（例如 StageAPI 的 xform op）代替——同一個 prim 若曾經被
        RigidBodyAPI 讀取過（例如 ObservationBuilder 每個 tick 都會呼叫
        get_position），rigid body 的物理視圖跟場景圖 transform 兩條路徑一旦
        分別寫入就會不同步，之後同一個 prim 呼叫 set_velocities() 會靜默失效
        （球看起來完全沒有被賦予速度）。
        """
        ...

    @abstractmethod
    def get_positions(self, prim_paths: list[str]) -> list[list[float]]:
        """
        一次回傳多個 prim 的世界座標 (x, y, z) (m)，順序與 prim_paths 相同。

        ⚠️ 讀多顆球時務必用這個方法，不要迴圈呼叫 get_position()——每次呼叫
        都是一次獨立的 GPU→CPU 同步。效能數據見 docs/CHANGELOG.md
        「GUI FPS 調校」一節。
        """
        ...

    @abstractmethod
    def get_linear_velocity(self, prim_path: str) -> list[float]:
        """
        回傳速度 (vx, vy, vz) (m/s)
        """
        ...

    @abstractmethod
    def get_velocities(
        self, prim_paths: list[str]
    ) -> tuple[list[list[float]], list[list[float]]]:
        """
        一次回傳多個 prim 的 (線速度清單, 角速度清單)，順序與 prim_paths 相同。

        ⚠️ 效能理由同 get_positions()：讀多顆球務必用這個方法，不要迴圈呼叫
        get_linear_velocity()／get_angular_velocity()。
        """
        ...

    @abstractmethod
    def get_angular_velocity(self, prim_path: str) -> list[float]:
        """
        回傳角速度 (wx, wy, wz) (rad/s)
        """
        ...

    @abstractmethod
    def set_velocities(self, prim_path: str, linear_velocity: list[float], angular_velocity: list[float]) -> None:
        """
        設定 Rigidbody 速度
        (vx, vy, vz) (m/s)
        (wx, wy, wz) (rad/s)
        """
        ...