from .observation_builder import ObservationBuilder
from .table_orchestrator import TableOrchestrator

class TableRuntime:
    def __init__(self, observation_builder: ObservationBuilder, orchestrator: TableOrchestrator) -> None:
        self._observation_builder = observation_builder
        self._orchestrator = orchestrator
        
    def tick(self) -> None:
        observation = self._observation_builder.build()
        self._orchestrator.step(observation)