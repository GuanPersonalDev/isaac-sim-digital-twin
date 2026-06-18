from abc import ABC, abstractmethod


class StageAPI(ABC):
    """
    USD Stage 的查詢介面
    """

    @abstractmethod
    def prim_exists(self, prim_path: str) -> bool:
        """
        指定路徑的 prim 是否存在
        """
        ...

    @abstractmethod
    def get_child_prim_paths(self, parent_prim_path: str) -> list[str]:
        """
        回傳指定路徑的 prim 底下的所有子 prim 的路徑
        """
        ...
