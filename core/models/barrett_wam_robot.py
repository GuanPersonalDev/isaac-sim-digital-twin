from ..ports.stage_api import StageAPI
from ..ports.articulation_api import ArticulationAPI
from ..services.asset_utility import BARRETT_WAM_PATH
from .robot_arm import RobotArm


class BarrettWamRobot(RobotArm):
    """
    Barrett WAM 7-DOF 手臂（見 assets/barrett_wam/README.md 資產來源說明）。
    末端連桿 wam_wrist_palm_stump_link 是 URDF 裡透過 fixed joint 掛在
    wam_wrist_palm_link 之後的最末端 link，對應 UR5Robot 用 wrist_3_link
    當末端執行器掛載點的慣例。
    """

    _END_EFFECTOR_LINK_NAME = "wam_wrist_palm_stump_link"

    def __init__(
        self,
        base_path: str,
        stage_api: StageAPI,
        articulation_api: ArticulationAPI,
        position: tuple[float, float, float],
    ):
        prim_path = BarrettWamRobot.get_prim_path(base_path)
        self._articulation_api = articulation_api
        stage_api.create_reference_prim(prim_path, BARRETT_WAM_PATH)
        x, y, z = position
        stage_api.set_prim_translate(prim_path, x, y, z)

    @staticmethod
    def get_prim_path(base_path: str) -> str:
        return base_path + "/Robot"

    @staticmethod
    def get_end_effector_prim_path(base_path: str) -> str:
        return (
            BarrettWamRobot.get_prim_path(base_path)
            + "/Geometry/world/wam_base_link/wam_shoulder_yaw_link/"
            + "wam_shoulder_pitch_link/wam_upper_arm_link/wam_forearm_link/"
            + "wam_wrist_yaw_link/wam_wrist_pitch_link/wam_wrist_palm_link/"
            + BarrettWamRobot._END_EFFECTOR_LINK_NAME
        )

    def reset(self) -> None:
        self._articulation_api.move_to_home()

    def is_reset_complete(self) -> bool:
        return self._articulation_api.is_motion_complete()
