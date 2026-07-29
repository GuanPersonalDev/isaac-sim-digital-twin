from abc import ABC, abstractmethod

from core.models import Action, BilliardStatus, Observation


class ControllerBase(ABC):
    """
    控制器的抽象介面，AI Model 控制或是程式控制都要實作這個介面
    """

    @abstractmethod
    def get_action(self, observation: Observation) -> Action:
        """透過目前的 Observation 決定本 tick 的 Action。"""
        ...

    @abstractmethod
    def get_current_state(self) -> BilliardStatus:
        """
        回傳 get_action() 處理後的領域狀態，供 Orchestrator 分派該 Action。
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """將控制器生命週期重設為 BilliardStatus.RESET。"""
        ...
