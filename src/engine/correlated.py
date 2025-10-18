from __future__ import annotations

from typing import Sequence

from src.scenarios.models import EventShock
from .base import PathEngine
from .types import Artifact, StateVector


class CorrelatedPathEngine(PathEngine):
    """Wrapper that decorates another engine with correlation metadata."""

    def __init__(self, base_engine: PathEngine, correlation: float = 0.0):
        self.base_engine = base_engine
        self.correlation = correlation

    def simulate(
        self,
        state: StateVector,
        scenarios: Sequence[EventShock],
        horizon_days: int,
        n_paths: int,
        dt: str = "1d",
    ) -> Artifact:
        artifact = self.base_engine.simulate(state, scenarios, horizon_days, n_paths, dt)
        artifact.metadata.setdefault("correlations", {})[state.symbol] = self.correlation
        artifact.metadata.setdefault("engines", {})[state.symbol] = artifact.metadata.get("engine")
        artifact.metadata.setdefault("symbols", []).append(state.symbol)
        return artifact
