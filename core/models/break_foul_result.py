from dataclasses import dataclass


@dataclass(frozen=True)
class BreakFoulResult:
    penalty: float
    should_reset: bool
