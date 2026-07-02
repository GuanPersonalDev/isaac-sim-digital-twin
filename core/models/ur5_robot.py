from ..ports.stage_api import StageAPI
from ..services.asset_utility import UR5_PATH


class UR5Robot:

    def __init__(
        self,
        base_path: str,
        stage_api: StageAPI,
        position: tuple[float, float, float],
    ):
        self._prim_path = base_path + "/Robot"
        stage_api.create_reference_prim(self._prim_path, UR5_PATH)
        x, y, z = position
        stage_api.set_prim_translate(self._prim_path, x, y, z)

    def get_prim_path(self) -> str:
        return self._prim_path
