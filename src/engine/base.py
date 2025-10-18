"""Base classes for simulation engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .state import StateVector
from ..scenarios.types import EventShock


@dataclass(slots=True)
class SimulationArtifact:
    """Container for simulated paths and metadata."""

    time_grid: np.ndarray
    paths: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "n_paths": int(self.paths.shape[0]),
            "n_steps": int(self.paths.shape[1]),
            "horizon_days": float(self.time_grid[-1]),
        }


class PathEngine(ABC):
    """Abstract interface for Monte Carlo path engines."""

    @abstractmethod
    def simulate(
        self,
        state: StateVector,
        scenarios: Sequence[EventShock],
        horizon_days: int,
        n_paths: int,
        dt: str = "1d",
    ) -> SimulationArtifact:
        """Simulate price paths under the given state and scenarios."""

    @staticmethod
    def _parse_dt(dt: str) -> tuple[int, float]:
        if not dt.endswith("d"):
            raise ValueError("Only day-based dt strings are supported (e.g. '1d').")
        days = int(dt[:-1] or 1)
        dt_years = days / 252.0
        return days, dt_years

    @staticmethod
    def _build_time_grid(horizon_days: int, step_days: int) -> np.ndarray:
        n_steps = int(np.ceil(horizon_days / step_days))
        grid = np.arange(0, (n_steps + 1) * step_days, step_days, dtype=float)
        if grid[-1] > horizon_days:
            grid[-1] = float(horizon_days)
        return grid
