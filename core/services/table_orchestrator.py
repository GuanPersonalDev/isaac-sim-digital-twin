import logging
from abc import ABC, abstractmethod

from ..controllers.controller_base import ControllerBase
from ..models.action import Action
from ..models.billiard_state import BilliardStatus
from ..models.observation import Observation
from ..models.table_ball_set import TableBallSet
from ..ports.articulation_api import ArticulationAPI
from ..models.robot_arm import RobotArm
from .ball_motion_monitor import BallMotionMonitor
from .ball_position_provider import BallPositionProvider
from .impulse_striking_service import ImpulseStrikingService
from .error_state import ErrorState
from .robot_swing_strategy import RobotSwingStrategy
from .rolling_resistance_service import RollingResistanceService

logger = logging.getLogger(__name__)


class TableOrchestrator(ABC):
    """
    共用執行骨架：取得 Action → 依 ControllerBase.get_current_state() 分派下游動作
    → 查詢球是否還在移動 → 查詢下游動作是否完成 → 組裝下一個 tick 的 Observation。
    差異部分（RESET 的手臂處理、AIMING/STRIKING 的實際動作、動作完成判定）交由子類別實作。
    """

    def __init__(
        self,
        script_controller: ControllerBase,
        table_ball_set: TableBallSet,
        ball_position_provider: BallPositionProvider,
        error_state: ErrorState,
        rolling_resistance_service: RollingResistanceService
    ) -> None:
        self._script_controller = script_controller
        self._table_ball_set = table_ball_set
        self._ball_position_provider = ball_position_provider
        self._error_state = error_state
        self._rolling_resistance_service = rolling_resistance_service

    def step(self, observation: Observation) -> None:
        """
        每個 tick 呼叫一次的共用骨架，見 docs/tech-design-5-9-table-orchestrator.md 第 3 節、
        docs/tech-design/rolling-resistance-correction-tech-design.md 第 4 節。
        """
        self._rolling_resistance_service.apply(self._table_ball_set.get_ball_prim_paths())
        self._check_downstream_failure()

        action = self._script_controller.get_action(observation)
        current_state = self._script_controller.get_current_state()
        
        if action.should_execute_action:
            try:
                match current_state:
                    case BilliardStatus.RESET:
                        self._reset_balls()
                        self._reset_downstream()
                    case BilliardStatus.AIMING:
                        self._execute_aim(action)
                    case BilliardStatus.STRIKING:
                        self._execute_strike(action)
            except Exception as e:
                self._error_state.mark_error(e)


    def _reset_balls(self) -> None:
        """
        共用：呼叫 table_ball_set.reset(positions)，positions 來自 ball_position_provider。
        Teleport 語意，呼叫完當下即完成。
        """
        positions = self._ball_position_provider.get_positions()
        self._table_ball_set.reset(positions)
        
    def reset(self) -> None:
        """
        外部重新初始化入口，必須同時清除 error_state 與重置狀態機：
        ControllerBase.get_action() 的具體實作可能優先處理 has_error，
        只清一邊會讓狀態機瞬間又跳回 ERROR。
        """
        self._error_state.clear()
        self._script_controller.reset()

    def full_reset(self) -> None:
        """
        Timeline PLAY 重播用：除了狀態機與 error_state，連場景也回到開局
        （重新擺球 + 手臂歸位）。

        單純把狀態設回 RESET 不會重擺球——RESET 狀態的兩個下游動作只在
        WAITING → RESET 的那一個 tick 由 should_execute_action 帶出來
        （見 step()），之後每個 RESET tick 的 Action 都是 no-op。

        必須在 physics step 內呼叫（見 TableRuntime.tick()），不能直接在
        Timeline 事件 callback 裡呼叫。
        """
        self.reset()
        self._reset_balls()
        self._reset_downstream()

    def get_current_state(self) -> BilliardStatus:
        return self._script_controller.get_current_state()

    def _check_downstream_failure(self) -> None:
        """
        下游動作失敗（逾時）時標記 error_state，預設無下游可檢查。

        動作逾時的善後（`_step_motion()`）會把目標清空，讓 `is_motion_complete()`
        從此恆為 True——那是為了不要把手臂卡死，但狀態機的 RESET → IDLE 與
        AIMING → STRIKING 兩個轉換條件都只看 `is_motion_complete`，於是一次
        逾時會讓狀態機一路直通到 STRIKING，中間的動作根本沒真的執行過（實測：
        RESET 逾時後手臂還停在預設姿態，球桿跟擊球線差 90°，卻已經在揮桿）。
        在這裡把逾時轉成明確的錯誤，讓它停在 ERROR 而不是繼續空轉。
        """
        ...

    @abstractmethod
    def _reset_downstream(self) -> None:
        """下游（手臂等）reset """
        ...

    @abstractmethod
    def _execute_aim(self, action: Action) -> None:
        """AIMING 狀態下游動作，內容留給 #96"""
        ...

    @abstractmethod
    def _execute_strike(self, action: Action) -> None:
        """STRIKING 狀態下游動作，內容留給 #97（Demo）／#177（Training）"""
        ...

class DemoTableOrchestrator(TableOrchestrator):
    def __init__(
        self,
        script_controller: ControllerBase,
        table_ball_set: TableBallSet,
        ball_position_provider: BallPositionProvider,
        robot_arm: RobotArm,
        articulation_api: ArticulationAPI,
        swing_strategy: RobotSwingStrategy,
        error_state: ErrorState,
        rolling_resistance_service: RollingResistanceService
    ) -> None:
        super().__init__(script_controller, table_ball_set, ball_position_provider, error_state, rolling_resistance_service)
        self._robot_arm = robot_arm
        self._articulation_api = articulation_api
        self._swing_strategy = swing_strategy

    def _reset_downstream(self) -> None:
        self._robot_arm.reset()

    def _check_downstream_failure(self) -> None:
        if self._articulation_api.did_last_motion_timeout():
            self._error_state.mark_error(RuntimeError("手臂動作逾時未收斂"))

    def _execute_aim(self, action: Action) -> None:
        table_z = self._table_ball_set.get_table_z()
        ball_radius = self._table_ball_set.DEFAULT_BALL_RADIUS
        cue_ball = (action.cue_ball_placement[0], action.cue_ball_placement[1])
        # ModelController 的 policy 每一局自己決定母球要擺在 Kitchen 範圍內
        # 的哪裡（cue_ball_placement），必須先把母球實際 teleport 過去，
        # 下面 swing_strategy.execute_aim() 才不會對著一個沒有球的地方
        # 瞄準（跟 Training 端 _apply_strike() 的 teleport 保持一致）。
        # AIMING 每局只會呼叫一次 _execute_aim()（見 TableOrchestrator.
        # step()：should_execute_action 只在 IDLE→AIMING 轉換那一 tick 為
        # True），這裡瞬移不會每個 tick 重複拉回，不影響球被打出去後的
        # 自由運動。
        self._table_ball_set.place_ball(0, cue_ball[0], cue_ball[1])
        self._swing_strategy.execute_aim(action, cue_ball, table_z, ball_radius)

    def _execute_strike(self, action: Action) -> None:
        # 上一個動作（瞄準）若逾時未收斂——跟手臂型號無關的通用檢查，留在
        # orchestrator 本身，不放進策略。
        if self._articulation_api.did_last_motion_timeout():
            raise RuntimeError("瞄準動作逾時未收斂")

        cue_ball = (action.cue_ball_placement[0], action.cue_ball_placement[1])
        table_z = self._table_ball_set.get_table_z()
        ball_radius = self._table_ball_set.DEFAULT_BALL_RADIUS
        self._swing_strategy.execute_strike(action, cue_ball, table_z, ball_radius)


class TrainingTableOrchestrator(TableOrchestrator):
    def __init__(
        self,
        script_controller: ControllerBase,
        table_ball_set: TableBallSet,
        ball_position_provider: BallPositionProvider,
        impulse_striking_service: ImpulseStrikingService,
        error_state: ErrorState,
        rolling_resistance_service: RollingResistanceService
    ) -> None:
        super().__init__(script_controller, table_ball_set, ball_position_provider, error_state, rolling_resistance_service)
        self._impulse_striking_service = impulse_striking_service

    def _reset_downstream(self) -> None:
        """
        目前沒有其他需要處理的 reset 元件
        """
        pass

    def _execute_aim(self, action: Action) -> None:
        """
        Training Table 沒有手臂，隨時可以準備好擊球
        """
        pass


    def _execute_strike(self, action: Action) -> None:
        table_x, table_y = self._table_ball_set.get_table_x_y()
        table_z = self._table_ball_set.get_table_z()
        self._impulse_striking_service.strike(action, table_x, table_y, table_z=table_z)
