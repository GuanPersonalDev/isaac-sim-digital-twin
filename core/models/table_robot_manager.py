from ..ports.stage_api import StageAPI
from ..ports.articulation_api import ArticulationAPI
from .robot_arm import RobotArm
from ..services.asset_utility import CUE_STICK_PATH


class TableRobotManager:
    """
    掛載手臂＋球桿的通用流程，實際掛哪一款手臂由呼叫端傳入的
    robot_arm_class 決定（見 RobotArm 抽象介面），本類別不依賴任何
    特定手臂的具體實作。
    """

    _ROBOT_OFFSET_FROM_TABLE_CENTER = (1.5, 0.0, 0.0)

    def __init__(
        self,
        table_center: tuple[float, float, float],
        base_path: str,
        stage_api: StageAPI,
        articulation_api: ArticulationAPI,
        robot_arm_class: type[RobotArm],
    ) -> None:
        world_position = (
            table_center[0] + self._ROBOT_OFFSET_FROM_TABLE_CENTER[0],
            table_center[1] + self._ROBOT_OFFSET_FROM_TABLE_CENTER[1],
            table_center[2] + self._ROBOT_OFFSET_FROM_TABLE_CENTER[2],
        )
        self._robot_base_path = base_path
        self._robot_arm_class = robot_arm_class
        self._stage_api = stage_api
        self._robot = robot_arm_class(base_path, stage_api, articulation_api, world_position)
        self._cue_stick_prim_path = base_path + "/CueStick"
        stage_api.create_reference_prim(self._cue_stick_prim_path, CUE_STICK_PATH)
        end_effector_path = robot_arm_class.get_end_effector_prim_path(base_path)

        stage_api.align_prim_to_target(self._cue_stick_prim_path, end_effector_path)
        stage_api.filter_collision_pair(self._cue_stick_prim_path, end_effector_path)

        joint_path = self._cue_stick_prim_path + "/FixedJointToRobot"
        stage_api.create_fixed_joint(
            joint_path, self._cue_stick_prim_path, end_effector_path
        )

    def get_cue_stick_prim_path(self) -> str:
        return self._cue_stick_prim_path

    def get_robot_prim_path(self) -> str:
        return self._robot_arm_class.get_prim_path(self._robot_base_path)

    def get_robot(self) -> RobotArm | None:
        return self._robot

    def destroy(self) -> None:
        # 不移除 self._robot_base_path 本身——它跟 BilliardTable 共用同一個
        # base_path（例如 /World/Table_Demo），只有 BilliardTable.destroy()
        # 有資格移除整個 base_path，這裡只移除自己掛載的 Robot/CueStick 子路徑。
        self._stage_api.remove_prim(
            self._robot_arm_class.get_prim_path(self._robot_base_path)
        )
        self._stage_api.remove_prim(self._cue_stick_prim_path)
        self._robot = None
