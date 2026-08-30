from ..models.billiard_state import BilliardStatus
from ..models.billiard_table import BilliardTable
from ..models.observation import Observation
from ..models.table_robot_manager import TableRobotManager
from ..ports.articulation_api import ArticulationAPI
from ..ports.rigid_body_api import RigidBodyAPI
from .pocket_event_handler import PocketEventHandler
from .table_runtime import TableRuntime


class TableSession:
    """
    封裝一張桌子的完整生命週期資源（BilliardTable + TableRuntime +
    PocketEventHandler），以 table_id（沿用 prim path）當唯一識別，取代
    Extension 端原本三個平行 list 靠建立順序隱含對應的設計。
    """

    def __init__(
        self,
        table_id: str,
        table: BilliardTable,
        runtime: TableRuntime,
        pocket_handler: PocketEventHandler,
        rigid_body_api: RigidBodyAPI,
    ) -> None:
        self._table_id = table_id
        self._table = table
        self._runtime = runtime
        self._pocket_handler = pocket_handler
        self._rigid_body_api = rigid_body_api

    def get_table_id(self) -> str:
        return self._table_id

    def tick(self) -> None:
        self._runtime.tick()

    def request_full_reset(self) -> None:
        """
        Timeline PLAY 時由 Extension 呼叫：狀態機回到 BilliardStatus.RESET，
        場景（球位、手臂）在下一個 tick 回到開局，不沿用上一輪 Stop 前殘留
        的狀態。
        """
        self._runtime.request_full_reset()

    def get_current_state(self) -> BilliardStatus:
        return self._runtime.get_current_state()

    def get_last_observation(self) -> Observation | None:
        return self._runtime.get_last_observation()

    def get_ball_velocities(self) -> dict[int, tuple[list[float], list[float]]]:
        """
        ball_id -> (linear_velocity, angular_velocity)。僅供 Debug Menu
        「顯示各球速度」進階 toggle 勾選時逐幀呼叫，避免預設就對
        tensor-based RigidBodyAPI 做額外查詢、影響效能。
        """
        ball_prim_paths = self._table.get_table_ball_set().get_ball_prim_paths()
        velocities: dict[int, tuple[list[float], list[float]]] = {}
        for ball_id, prim_path in enumerate(ball_prim_paths):
            linear = self._rigid_body_api.get_linear_velocity(prim_path)
            angular = self._rigid_body_api.get_angular_velocity(prim_path)
            velocities[ball_id] = (linear, angular)
        return velocities

    def destroy(self) -> None:
        self._pocket_handler.stop()
        self._table.destroy()


class DemoTableSession(TableSession):
    """
    Demo 桌額外持有 TableRobotManager 與 ArticulationAPI，處理「Toggle
    完全解耦於 Timeline」與「ArticulationAPI.initialize() 必須等 Timeline
    Play 後才能呼叫」兩者之間的落差：Toggle ON 時只建 USD 場景（手臂 prim
    存在但不能動），等 Timeline PLAY 事件另外補呼叫 initialize_articulation()。
    """

    def __init__(
        self,
        table_id: str,
        table: BilliardTable,
        runtime: TableRuntime,
        pocket_handler: PocketEventHandler,
        rigid_body_api: RigidBodyAPI,
        robot_manager: TableRobotManager,
        articulation_api: ArticulationAPI,
    ) -> None:
        super().__init__(table_id, table, runtime, pocket_handler, rigid_body_api)
        self._robot_manager = robot_manager
        self._articulation_api = articulation_api
        self._articulation_initialized = False

    def initialize_articulation(self) -> None:
        """Timeline PLAY 事件觸發時呼叫（僅在尚未 initialize 時）"""
        self._articulation_api.initialize()
        self._articulation_initialized = True

    def is_articulation_initialized(self) -> bool:
        return self._articulation_initialized

    def destroy(self) -> None:
        if self._articulation_initialized:
            self._articulation_api.shutdown()
        # 無論是否已 initialize 都呼叫：若一次性 home-capture callback 尚未
        # 觸發就先取消，避免它之後對著已被 remove_prim() 移除的手臂 prim
        # 呼叫 get_end_effector_position() 而報錯。
        self._articulation_api.cancel_pending_home_capture()
        self._robot_manager.destroy()
        super().destroy()
