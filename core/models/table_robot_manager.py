from ..ports.stage_api import StageAPI
from ..ports.articulation_api import ArticulationAPI
from ..models.ur5_robot import UR5Robot
from ..services.asset_utility import CUE_STICK_PATH


class TableRobotManager:
    _ROBOT_OFFSET_FROM_TABLE_CENTER = (1.5, 0.0, 0.0)

    def __init__(
        self,
        table_center: tuple[float, float, float],
        base_path: str,
        stage_api: StageAPI,
        articulation_api: ArticulationAPI,
    ) -> None:
        world_position = (
            table_center[0] + self._ROBOT_OFFSET_FROM_TABLE_CENTER[0],
            table_center[1] + self._ROBOT_OFFSET_FROM_TABLE_CENTER[1],
            table_center[2] + self._ROBOT_OFFSET_FROM_TABLE_CENTER[2],
        )
        self._robot_base_path = base_path
        self._robot = UR5Robot(base_path, stage_api, articulation_api, world_position)
        self._cue_stick_prim_path = base_path + "/CueStick"
        stage_api.create_reference_prim(self._cue_stick_prim_path, CUE_STICK_PATH)
        end_effector_path = UR5Robot.get_end_effector_prim_path(base_path)
        
        stage_api.align_prim_to_target(self._cue_stick_prim_path, end_effector_path)
        stage_api.filter_collision_pair(self._cue_stick_prim_path, end_effector_path)

        joint_path = self._cue_stick_prim_path + "/FixedJointToRobot"
        stage_api.create_fixed_joint(
            joint_path, self._cue_stick_prim_path, end_effector_path
        )

    def get_cue_stick_prim_path(self) -> str:
        return self._cue_stick_prim_path

    def get_robot_prim_path(self) -> str:
        return UR5Robot.get_prim_path(self._robot_base_path)

    def destroy(self) -> None:
        self._robot = None