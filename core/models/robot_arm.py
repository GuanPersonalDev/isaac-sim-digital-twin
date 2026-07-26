from abc import ABC, abstractmethod


class RobotArm(ABC):
    """
    機械手臂資產／模型層的抽象介面。上游程式碼（TableRobotManager、
    ObservationBuilder、TableOrchestrator、Extension 主類別）只依賴這個
    介面，不直接依賴 UR5Robot／BarrettWamRobot 等具體實作，未來要換手臂
    時只需要新增一個實作類別、改變實例化那一行，不需要動到其他呼叫端。

    具體實作類別的建構子須符合以下簽章（ABC 不強制檢查建構子簽章，
    僅在此記錄約定）：
        __init__(self, base_path: str, stage_api: StageAPI,
                  articulation_api: ArticulationAPI,
                  position: tuple[float, float, float]) -> None
    """

    @staticmethod
    @abstractmethod
    def get_prim_path(base_path: str) -> str:
        """回傳這隻手臂本體掛載的 prim 路徑（相對 base_path）"""
        ...

    @staticmethod
    @abstractmethod
    def get_end_effector_prim_path(base_path: str) -> str:
        """回傳末端執行器（用來對齊／固定球桿）的 prim 路徑"""
        ...

    @abstractmethod
    def reset(self) -> None:
        """回到待機姿態"""
        ...

    @abstractmethod
    def is_reset_complete(self) -> bool:
        """是否已到達待機姿態"""
        ...
