from ..ports.stage_api import StageAPI
from ..ports.articulation_api import ArticulationAPI
from ..services.asset_utility import UR3E_PATH
from .robot_arm import RobotArm


class UR3eRobot(RobotArm):
    """Universal Robots UR3e 6-DOF 手臂（Isaac Sim 官方內建 USD 資產）。

    `wrist_3_link` 是 UR 家族的標準末端連桿慣例，跟 `UR5Robot` 同一個
    命名（UR3e 的 USD 資產階層裡 `wrist_3_link` 直接掛在參照根目錄下，
    不像 `BarrettWamRobot` 需要一長串 `Geometry/world/...` 路徑）。

    ⚠️ 這個類別只提供機械式的 `RobotArm` 介面（掛載 USD、`reset()`／
    `reposition()` delegate 給 `ArticulationAPI`），讓 UR3e 能被
    `TableRobotManager` 掛載並執行通用的 joint-space/Cartesian 動作。
    `core/services/table_orchestrator.py` 的 `_execute_aim()`／
    `_execute_strike()` 目前仍然寫死呼叫 `base_placement_calculator.py`
    （`CANONICAL_REST_JOINTS`／`compute_base_pose()`／
    `compute_canonical_wrist_position()`／`CANONICAL_FLAT_ORIENTATION`）
    ——那一整套是針對 WAM7「固定 6 關節姿態＋單一 base_yaw 關節瞄準」設計
    的，跟 UR3e 已驗證的「肘關節主導、依接觸方向重新搜姿態」方法（見
    `scripts/design_human_like_ur3e_pose.py`／
    `scripts/test_elevated_bridge_ur3e_table.py`）不是同一套邏輯，這個
    類別不會讓瞄準/擊球自動可用，需要另外設計 UR3e 版的姿態/基座計算
    才能接上 `TableOrchestrator`。
    """

    _END_EFFECTOR_LINK_NAME = "wrist_3_link"

    def __init__(
        self,
        base_path: str,
        stage_api: StageAPI,
        articulation_api: ArticulationAPI,
        position: tuple[float, float, float],
    ):
        self._prim_path = UR3eRobot.get_prim_path(base_path)
        self._articulation_api = articulation_api
        self._stage_api = stage_api
        self._stage_api.create_reference_prim(self._prim_path, UR3E_PATH)
        x, y, z = position
        self._stage_api.set_prim_translate(self._prim_path, x, y, z)

    @staticmethod
    def get_prim_path(base_path: str) -> str:
        return base_path + "/Robot"

    @staticmethod
    def get_end_effector_prim_path(base_path: str) -> str:
        return UR3eRobot.get_prim_path(base_path) + "/" + UR3eRobot._END_EFFECTOR_LINK_NAME

    def reset(self) -> None:
        self._articulation_api.move_to_home()

    def is_reset_complete(self) -> bool:
        return self._articulation_api.is_motion_complete()

    def reposition(self, position: tuple[float, float, float]) -> None:
        self._stage_api.set_prim_translate(self._prim_path, position[0], position[1], position[2])
