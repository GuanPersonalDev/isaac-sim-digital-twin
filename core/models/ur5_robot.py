from ..ports.stage_api import StageAPI
from ..ports.articulation_api import ArticulationAPI
from ..services.asset_utility import UR5_PATH


class UR5Robot:
    _END_EFFECTOR_LINK_NAME = "wrist_3_link"

    def __init__(
        self,
        base_path: str,
        stage_api: StageAPI,
        articulation_api: ArticulationAPI,
        position: tuple[float, float, float],
    ):
        prim_path = UR5Robot.get_prim_path(base_path)
        self._articulation_api = articulation_api
        stage_api.create_reference_prim(prim_path, UR5_PATH)
        x, y, z = position
        stage_api.set_prim_translate(prim_path, x, y, z)

    @staticmethod
    def get_prim_path(base_path: str) -> str:
        return base_path + "/Robot"

    @staticmethod
    def get_end_effector_prim_path(base_path: str) -> str:
        return UR5Robot.get_prim_path(base_path) + "/" + UR5Robot._END_EFFECTOR_LINK_NAME

    def reset(self) -> None:
        self._articulation_api.move_to_home()
        
    def is_reset_complete(self) -> bool:
        return self._articulation_api.is_motion_complete()