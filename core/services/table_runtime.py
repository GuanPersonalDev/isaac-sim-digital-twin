from ..controllers.controller_base import ControllerBase
from ..models.billiard_state import BilliardStatus
from ..models.observation import Observation
from .observation_builder import ObservationBuilder
from .table_orchestrator import TableOrchestrator

class TableRuntime:
    def __init__(self, observation_builder: ObservationBuilder, orchestrator: TableOrchestrator) -> None:
        self._observation_builder = observation_builder
        self._orchestrator = orchestrator
        self._last_observation: Observation | None = None
        self._pending_full_reset = False
        self._pending_controller: ControllerBase | None = None

    def tick(self) -> None:
        if self._pending_controller is not None:
            self._orchestrator.set_controller(self._pending_controller)
            self._pending_controller = None
            # 換了操作策略之後一定要重新開局：舊策略可能留在 AIMING/STRIKING
            # 一半，新策略的狀態機是從 RESET 起跳，兩者的假設對不上。
            self._pending_full_reset = True

        if self._pending_full_reset:
            self._pending_full_reset = False
            self._orchestrator.full_reset()

        observation = self._observation_builder.build()
        self._last_observation = observation
        self._orchestrator.step(observation)

    def request_full_reset(self) -> None:
        """
        外部重新初始化入口（Timeline PLAY）。

        狀態機與 Observation 立刻清掉，Debug Menu 在第一個 tick 之前就會顯示
        RESET，不會殘留上一輪 Stop 瞬間的舊值；重擺球與手臂歸位這兩個「寫場景」
        的動作則排到下一個 tick 才做——PLAY 事件當下 physics 一步都還沒跑，
        場景寫入應該跟其他寫入一樣發生在 PHYSICS_POST_STEP 內。
        """
        self._orchestrator.reset()
        self._last_observation = None
        self._pending_full_reset = True

    def request_controller_swap(self, controller: ControllerBase) -> None:
        """外部（Debug UI）呼叫入口：換手臂的操作策略。跟 request_full_reset()
        同樣的理由，不在呼叫當下立刻套用——這通常是從 UI callback 觸發，不在
        physics step 內，實際套用要等下一個 tick。"""
        self._pending_controller = controller

    def get_last_observation(self) -> Observation | None:
        return self._last_observation

    def get_current_state(self) -> BilliardStatus:
        return self._orchestrator.get_current_state()