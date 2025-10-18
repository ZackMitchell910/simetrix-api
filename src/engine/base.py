"""Base primitives shared by all path engines."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence

import numpy as np

from src.scenarios.models import EventShock


@dataclass
class StateVector:
    """Container holding the inputs required to simulate price paths."""

    symbol: str
    spot: float
    mu: float
    sigma: float
    asof: datetime
    risk_free_rate: float = 0.0
    dividend_yield: float = 0.0
    iv_surface: Mapping[int, float] | None = None
    correlation: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def annualised_drift(self) -> float:
        """Return the net drift after cost-of-carry adjustments."""

        return self.mu + self.risk_free_rate - self.dividend_yield


@dataclass
class Artifact:
    """Simulation output that bundles paths and diagnostics."""

    paths: np.ndarray
    time_grid: Sequence[datetime]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.paths.ndim != 2:
            raise ValueError("paths must be a 2-D array [n_paths, n_steps]")
        if len(self.time_grid) != self.paths.shape[1]:
            raise ValueError("time_grid length must match the second axis of paths")

    @property
    def terminal_distribution(self) -> np.ndarray:
        return self.paths[:, -1]

    def summary(self) -> dict[str, Any]:
        terminal = self.terminal_distribution
        return {
            "n_paths": int(self.paths.shape[0]),
            "n_steps": int(self.paths.shape[1] - 1),
            "terminal_mean": float(np.mean(terminal)),
            "terminal_std": float(np.std(terminal, ddof=1)),
            "metadata": self.metadata,
        }


class PathEngine(abc.ABC):
    """Abstract base class for pluggable simulation engines."""

    def __init__(self, *, scheduler: "ShockScheduler | None" = None) -> None:
        self.scheduler = scheduler

    @abc.abstractmethod
    def simulate(
        self,
        state: StateVector,
        scenarios: Sequence[EventShock],
        horizon_days: int,
        n_paths: int,
        dt: str = "1d",
        *,
        random_state: np.random.Generator | None = None,
    ) -> Artifact:
        """Simulate price paths conditioned on the provided scenarios."""

    def _ensure_scheduler(self) -> "ShockScheduler":
        from .shocks import ShockScheduler

        if self.scheduler is None:
            self.scheduler = ShockScheduler()
        return self.scheduler

    def _time_grid(self, asof: datetime, horizon_days: int, step_days: int) -> list[datetime]:
        return [asof + timedelta(days=step_days * idx) for idx in range(0, horizon_days // step_days + 1)]

    def _parse_dt(self, dt: str) -> int:
        if not dt.endswith("d"):
            raise ValueError("Only day-based step sizes are supported (e.g. '1d').")
        return max(int(dt[:-1]), 1)


__all__ = ["StateVector", "Artifact", "PathEngine"]
