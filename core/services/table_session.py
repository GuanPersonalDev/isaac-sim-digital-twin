from ..models.billiard_state import BilliardStatus
from ..models.billiard_table import BilliardTable
from ..models.observation import Observation
from ..models.table_robot_manager import TableRobotManager
from ..ports.articulation_api import ArticulationAPI
from ..ports.rigid_body_api import RigidBodyAPI
from .pocket_event_handler import PocketEventHandler
from .spread_score_calculator import TABLE_LENGTH, TABLE_WIDTH
from .table_runtime import TableRuntime

_TABLE_OBSTACLE_HEIGHT_M = 0.15
"""註冊給 RMPflow 的球檯障礙物厚度（見 `DemoTableSession.
_register_rmpflow_obstacles()`）。"""

_ROBOT_BASE_ORIENTATION = [1.0, 0.0, 0.0, 0.0]
"""手臂底座的世界朝向固定是單位四元數——`RobotArm.reposition()` 只設
translate，從來不旋轉底座。"""


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
        self._sync_initial_robot_base_pose()
        self._register_rmpflow_obstacles()
        self._articulation_initialized = True

    def _sync_initial_robot_base_pose(self) -> None:
        """讓 RMPflow 知道手臂底座目前在世界座標的哪裡。

        第一個動作是 RESET（`RobotArm.reset()` → `move_to_home()`），而
        `move_to_home()` 會用 RMPflow 的運動學模型把 HOME 關節角換算成世界
        座標的末端目標，再從**實際量到的**末端世界位姿內插出 waypoint。若
        沒有先同步底座位姿，RMPflow 內部會當底座在原點，起點與目標分屬兩個
        座標系。`Ur10eSwingStrategy.execute_aim()` 每次瞄準都會重新同步，但
        那是第一次 RESET **之後**的事，補不上這一段。

        WAM7／UR3e 走差動 IK，沒有這個概念，`ArticulationAPI` 對它們是
        no-op（見該介面的 docstring）。
        """
        self._articulation_api.set_robot_base_pose(
            list(self._robot_manager.get_initial_robot_base_position()),
            list(_ROBOT_BASE_ORIENTATION),
        )

    def _register_rmpflow_obstacles(self) -> None:
        """把球檯與母球註冊成 RMPflow 的避障物（UR10e 重新設計計畫決策 6
        的第一層防護）。必須在 `initialize()` 之後——UR10e 的控制器是在那裡
        才建立的；對 WAM7／UR3e 這兩款手臂，`ArticulationAPI` 的這兩個方法
        本來就是 no-op（它們走差動 IK，沒有 RMPflow）。

        ⚠️ 球檯方塊刻意放在桌面**之下**（`table_z` 往下延伸），代表桌面
        以下的實體結構（石板／桌腳／桌框），讓桌面正上方保持淨空。放在桌面
        之上會把桿尖擊球高度（約 `table_z + ball_radius`）本身涵蓋進障礙物
        範圍，最終逼近等於在跟自己要抵達的位置打架——實測踩過，見
        docs/CHANGELOG.md。

        母球用會持續追蹤最新世界座標的動態球體，不是註冊當下的固定快照
        （球會被打去別的地方）。
        """
        table_ball_set = self._table.get_table_ball_set()
        if table_ball_set is None:
            return

        table_center = self._table.get_table_center()
        table_z = table_ball_set.get_table_z()
        self._articulation_api.register_static_box_obstacle(
            [table_center[0], table_center[1], table_z - _TABLE_OBSTACLE_HEIGHT_M / 2.0],
            [TABLE_WIDTH, TABLE_LENGTH, _TABLE_OBSTACLE_HEIGHT_M],
        )
        self._articulation_api.register_dynamic_sphere_obstacle(
            table_ball_set.get_ball_prim_paths()[0], table_ball_set.get_ball_radius()
        )

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
