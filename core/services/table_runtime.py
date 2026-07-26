from ..models.billiard_state import BilliardStatus
from ..models.observation import Observation
from .observation_builder import ObservationBuilder
from .table_orchestrator import TableOrchestrator

class TableRuntime:
    def __init__(self, observation_builder: ObservationBuilder, orchestrator: TableOrchestrator) -> None:
        self._observation_builder = observation_builder
        self._orchestrator = orchestrator
        self._last_observation: Observation | None = None

    def tick(self) -> None:
        observation = self._observation_builder.build()
        self._last_observation = observation
        self._orchestrator.step(observation)

    def get_last_observation(self) -> Observation | None:
        return self._last_observation

    def get_current_state(self) -> BilliardStatus:
        return self._orchestrator.get_current_state()