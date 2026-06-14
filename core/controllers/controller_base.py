from abc import ABC, abstractmethod
from core.models import Observation, Action


class ControllerBase(ABC):
    """
    控制器的抽象介面，AI Model 控制或是程式控制都要實作這個介面
    """

    @abstractmethod
    def get_action(self, observation: Observation) -> Action: ...

    """
    透過現在的狀態 observation 決定控制結果 action
    """

    @abstractmethod
    def reset(self) -> None: ...
