from abc import ABC, abstractmethod
from typing import Callable

from ...core.models.contact_event import ContactEvent


class PhysicsAPI(ABC):
    """
    物理API的抽象層
    """

    @abstractmethod
    def enable_contact_reporting(self, prim_path: str) -> None:
        """
        啟用碰撞事件報告
        """
        ...

    @abstractmethod
    def subscribe_contact_events(
        self, callback: Callable[[ContactEvent], None]
    ) -> None:
        """
        訂閱碰撞事件
        """
        ...

    @abstractmethod
    def unsubscribe_contact_events(self) -> None:
        """
        碰撞事件解除訂閱
        """
        ...
