from ..ports.stage_api import StageAPI
from ..ports.articulation_api import ArticulationAPI
from ..services.asset_utility import UR10E_PATH
from .robot_arm import RobotArm


class UR10eRobot(RobotArm):
    """Universal Robots UR10e 6-DOF 手臂（Isaac Sim 官方內建 USD 資產）。

    取代 `UR3eRobot` 成為生產路徑用的手臂（見 UR10e+專用出力結構重新設計
    計畫決策 1）：負載 12.5kg、可達距離 1300mm，遠大於 UR3e（3kg／500mm），
    搭配末端的線性滑軌出力機構（`_END_EFFECTOR_LINK_NAME` 之後接
    `PrismaticJoint`，見 `TableRobotManager`）取代手臂關節本身角速度出力。

    `wrist_3_link` 沿用 UR 家族標準末端連桿命名（跟 `UR5Robot`／`UR3eRobot`
    一致）。這個類別只提供機械式的 `RobotArm` 介面（掛載 USD、`reset()`／
    `reposition()` delegate 給 `ArticulationAPI`）；瞄準/揮桿邏輯由
    `core/services/ur10e_rmpflow_controller.py`（RMPflow 手臂定位）與
    `ArticulationAPIImpl` 的滑軌關節推桿方法（STRIKE 用）另外實作，經
    `Ur10eSwingStrategy` 接上 `DemoTableOrchestrator`（見 decision 9）。
    """

    _END_EFFECTOR_LINK_NAME = "wrist_3_link"

    def __init__(
        self,
        base_path: str,
        stage_api: StageAPI,
        articulation_api: ArticulationAPI,
        position: tuple[float, float, float],
    ):
        self._prim_path = UR10eRobot.get_prim_path(base_path)
        self._articulation_api = articulation_api
        self._stage_api = stage_api
        self._stage_api.create_reference_prim(self._prim_path, UR10E_PATH)
        x, y, z = position
        self._stage_api.set_prim_translate(self._prim_path, x, y, z)

    @staticmethod
    def get_prim_path(base_path: str) -> str:
        return base_path + "/Robot"

    @staticmethod
    def get_end_effector_prim_path(base_path: str) -> str:
        return UR10eRobot.get_prim_path(base_path) + "/" + UR10eRobot._END_EFFECTOR_LINK_NAME

    def reset(self) -> None:
        self._articulation_api.move_to_home()

    def is_reset_complete(self) -> bool:
        return self._articulation_api.is_motion_complete()

    def reposition(self, position: tuple[float, float, float]) -> None:
        self._stage_api.set_prim_translate(self._prim_path, position[0], position[1], position[2])
