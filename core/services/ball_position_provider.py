from abc import ABC, abstractmethod


class BallPositionProvider(ABC):
    @abstractmethod
    def get_positions(self) -> dict:
        """
        取得球的位置
        """
        ...
