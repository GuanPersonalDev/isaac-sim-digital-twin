from ..ports.stage_api import StageAPI
from ..ports.articulation_api import ArticulationAPI
from ..services.asset_utility import UR5_PATH
from .robot_arm import RobotArm


class UR5Robot(RobotArm):
    _END_EFFECTOR_LINK_NAME = "wrist_3_link"

    def __init__(
        self,
        base_path: str,
        stage_api: StageAPI,
        articulation_api: ArticulationAPI,
        position: tuple[float, float, float],
    ):
        self._prim_path = UR5Robot.get_prim_path(base_path)
        self._articulation_api = articulation_api
        self._stage_api = stage_api
        self._stage_api.create_reference_prim(self._prim_path, UR5_PATH)
        x, y, z = position
        self._stage_api.set_prim_translate(self._prim_path, x, y, z)

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

    def reposition(self, position: tuple[float, float, float]) -> None:
        self._stage_api.set_prim_translate(self._prim_path, position[0], position[1], position[2])