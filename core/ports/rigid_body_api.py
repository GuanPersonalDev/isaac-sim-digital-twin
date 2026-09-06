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

        ⚠️ 每個 tick 要讀整桌 10 顆球時**必須用這個方法**，不要用迴圈逐顆呼叫
        get_position()。實測（scripts/benchmark_gui_frametime.py，GUI 場景、
        只開 Demo 桌）：單顆讀取一次固定成本 0.38ms，那是 tensor API 一次
        GPU→CPU 同步的代價，跟一次讀 1 顆還是 10 顆幾乎無關，所以逐顆讀 10 次
        就是 3.8ms/frame。三個呼叫端（ObservationBuilder、BallMotionMonitor、
        RollingResistanceService）合計每 frame 約 40 次讀取＝11.7ms，佔整個
        tick 的 83%，是 GUI 掉到 12 FPS 的主因。見 docs/CHANGELOG.md
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

        效能理由同 get_positions()，而且逐顆版本在這裡更糟：
        get_linear_velocity() 與 get_angular_velocity() 底下各自呼叫一次
        RigidPrim.get_velocities()，同一顆球要同時拿線速度與角速度就是兩次
        獨立同步。
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