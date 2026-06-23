from abc import ABC, abstractmethod


class MaterialAPI(ABC):

    @abstractmethod
    def apply_preview_surface(self, prim_path: str, r: float, g: float, b: float) -> None:
        """
        在指定 prim_path 套用 UsdPreviewSurface 材質，diffuseColor 為 (r, g, b)。
        """
        ...

    @abstractmethod
    def apply_mdl_shader(
        self,
        prim_path: str,
        mdl_path: str,
        identifier: str,
        stripe_color: tuple[float, float, float],
        base_color: tuple[float, float, float],
    ) -> None:
        """
        在指定 prim_path 套用 MDL Shader。
        失敗時拋出 Exception，由呼叫端決定 fallback 行為。
        """
        ...
