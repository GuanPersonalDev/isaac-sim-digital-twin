from abc import ABC, abstractmethod

from ..services.base_placement_calculator import CANONICAL_REST_JOINTS, compute_base_pose, required_grip_position
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
from .rolling_resistance_service import RollingResistanceService


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

    def get_current_state(self) -> BilliardStatus:
        return self._script_controller.get_current_state()

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
        error_state: ErrorState,
        rolling_resistance_service: RollingResistanceService
    ) -> None:
        super().__init__(script_controller, table_ball_set, ball_position_provider, error_state, rolling_resistance_service)
        self._robot_arm = robot_arm
        self._articulation_api = articulation_api

    def _reset_downstream(self) -> None:
        self._robot_arm.reset()

    def _execute_aim(self, action: Action) -> None:
        #TODO: 把 action 轉譯成 robot_arm 需要的操作
        table_z = self._table_ball_set.get_table_z()
        grip_position = required_grip_position(action.cue_ball_placement[0],
            action.cue_ball_placement[1], action.shot_angle)
        base_position, base_yaw_rad = compute_base_pose(action.cue_ball_placement[0],
            action.cue_ball_placement[1], action.shot_angle, table_z)
        joint_targets = [base_yaw_rad, *CANONICAL_REST_JOINTS]
        self._robot_arm.reposition(base_position)
        self._articulation_api.move_to_joint_position(joint_targets, [grip_position[0], grip_position[1], table_z + self._table_ball_set.DEFAULT_BALL_RADIUS])

    def _execute_strike(self, action: Action) -> None:
        #TODO: 把 action 轉譯成 robot_arm 需要的操作
        ...

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
