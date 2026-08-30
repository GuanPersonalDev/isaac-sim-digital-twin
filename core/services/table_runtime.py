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

    def tick(self) -> None:
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

    def get_last_observation(self) -> Observation | None:
        return self._last_observation

    def get_current_state(self) -> BilliardStatus:
        return self._orchestrator.get_current_state()