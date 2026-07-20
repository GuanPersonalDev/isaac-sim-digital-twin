from abc import ABC, abstractmethod

from ..controllers.script_controller import ScriptController
from ..models.action import Action
from ..models.billiard_state import BilliardStatus
from ..models.observation import Observation
from ..models.table_ball_set import TableBallSet
from ..ports.articulation_api import ArticulationAPI
from ..models.ur5_robot import UR5Robot
from .ball_motion_monitor import BallMotionMonitor
from .ball_position_provider import BallPositionProvider
from .impulse_striking_service import ImpulseStrikingService


class TableOrchestrator(ABC):
    """
    共用執行骨架：取得 Action → 依 ScriptController.get_current_state() 分派下游動作
    → 查詢球是否還在移動 → 查詢下游動作是否完成 → 組裝下一個 tick 的 Observation。
    差異部分（RESET 的手臂處理、AIMING/STRIKING 的實際動作、動作完成判定）交由子類別實作。
    """

    def __init__(
        self,
        script_controller: ScriptController,
        table_ball_set: TableBallSet,
        ball_position_provider: BallPositionProvider,
    ) -> None:
        self._script_controller = script_controller
        self._table_ball_set = table_ball_set
        self._ball_position_provider = ball_position_provider

    def step(self, observation: Observation) -> None:
        """
        每個 tick 呼叫一次的共用骨架，見 docs/tech-design-5-9-table-orchestrator.md 第 3 節。
        """
        action = self._script_controller.get_action(observation)
        current_state = self._script_controller.get_current_state()
        if action.should_execute_action:
            match current_state:
                case BilliardStatus.RESET:
                    self._reset_balls()
                    self._reset_downstream()
                case BilliardStatus.AIMING:
                    self._execute_aim(action)
                case BilliardStatus.STRIKING:
                    self._execute_strike(action)
        

    def _reset_balls(self) -> None:
        """
        共用：呼叫 table_ball_set.reset(positions)，positions 來自 ball_position_provider。
        Teleport 語意，呼叫完當下即完成。
        """
        positions = self._ball_position_provider.get_positions()
        self._table_ball_set.reset(positions)

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
        script_controller: ScriptController,
        table_ball_set: TableBallSet,
        ball_position_provider: BallPositionProvider,
        ur5_robot: UR5Robot,
        articulation_api: ArticulationAPI,
    ) -> None:
        super().__init__(script_controller, table_ball_set, ball_position_provider)
        self._ur5_robot = ur5_robot
        self._articulation_api = articulation_api

    def _reset_downstream(self) -> None:
        self._ur5_robot.reset()

    def _execute_aim(self, action: Action) -> None:
        #TODO: 把 action 轉譯成 ur5_robot 需要的操作
        ...

    def _execute_strike(self, action: Action) -> None:
        #TODO: 把 action 轉譯成 ur5_robot 需要的操作
        ...

class TrainingTableOrchestrator(TableOrchestrator):
    def __init__(
        self,
        script_controller: ScriptController,
        table_ball_set: TableBallSet,
        ball_position_provider: BallPositionProvider,
        impulse_striking_service: ImpulseStrikingService,
    ) -> None:
        super().__init__(script_controller, table_ball_set, ball_position_provider)
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
        self._impulse_striking_service.strike(action, table_z=self._table_ball_set.get_table_z())
