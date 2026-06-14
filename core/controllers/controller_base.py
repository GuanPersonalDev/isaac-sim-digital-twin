from abc import ABC, abstractclassmethod, abstractmethod
from core.models import Observation, Action


class ControllerBase(ABC):
    @abstractmethod
    def get_action(self, observation: Observation) -> Action: ...

    @abstractmethod
    def reset(self) -> None: ...
