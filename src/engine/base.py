from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..scenarios.models import EventShock
from .iv_anchor import ImpliedVolAnchor
from .shocks import ShockOverrides, ShockScheduler
from .types import Artifact, StateVector


def _parse_dt(dt: str) -> float:
    if dt.endswith("d"):
        return float(dt[:-1]) if dt[:-1] else 1.0
    if dt.endswith("h"):
        return float(dt[:-1]) / 24.0
    raise ValueError(f"Unsupported dt granularity: {dt}")


@dataclass(slots=True)
class SimulationContext:
    time_grid: np.ndarray
    step_days: np.ndarray
    overrides: ShockOverrides


class PathEngine(ABC):
    """Abstract base class for pluggable path engines."""

    def __init__(self, scheduler: ShockScheduler | None = None, iv_anchor: ImpliedVolAnchor | None = None):
        self.scheduler = scheduler or ShockScheduler()
        self.iv_anchor = iv_anchor or ImpliedVolAnchor()

    def simulate(
        self,
        state: StateVector,
        scenarios: Sequence[EventShock],
        horizon_days: int,
        n_paths: int,
        dt: str = "1d",
    ) -> Artifact:
        if horizon_days <= 0:
            raise ValueError("horizon_days must be positive")
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")

        time_grid, step_days = self._build_time_grid(horizon_days, dt)
        overrides = self.scheduler.build(time_grid=time_grid, shocks=scenarios)
        context = SimulationContext(time_grid=time_grid, step_days=step_days, overrides=overrides)
        paths = self._simulate_paths(state=state, context=context, n_paths=n_paths)
        anchored = self.iv_anchor.anchor(paths=paths, state=state, time_grid=time_grid, overrides=overrides)
        metadata = {
            "engine": self.__class__.__name__,
            "dt": dt,
            "horizon_days": horizon_days,
        }
        return Artifact(paths=anchored, time_grid=time_grid, metadata=metadata)

    def _build_time_grid(self, horizon_days: int, dt: str) -> tuple[np.ndarray, np.ndarray]:
        step_length_days = _parse_dt(dt)
        n_steps = int(np.ceil(horizon_days / step_length_days))
        time_grid = np.linspace(0.0, n_steps * step_length_days, num=n_steps + 1)
        step_days = np.diff(time_grid, prepend=time_grid[0])
        step_days = step_days[1:]
        return time_grid, step_days

    @abstractmethod
    def _simulate_paths(self, state: StateVector, context: SimulationContext, n_paths: int) -> np.ndarray:
        """Return simulated paths as a ``(n_paths, n_steps + 1)`` array."""


class GBMPathEngine(PathEngine):
    """Baseline geometric Brownian motion engine."""

    def _simulate_paths(self, state: StateVector, context: SimulationContext, n_paths: int) -> np.ndarray:
        spot = float(state.as_array()[0])
        n_steps = len(context.step_days)
        paths = np.empty((n_paths, n_steps + 1), dtype=float)
        paths[:, 0] = spot
        log_paths = np.full(n_paths, np.log(spot), dtype=float)

        for idx, step in enumerate(context.step_days):
            dt_years = step / 252.0
            drift = state.drift + context.overrides.drift_bump[idx]
            vol = state.vol * context.overrides.vol_multiplier[idx]
            diffusion = (drift - 0.5 * vol ** 2) * dt_years + vol * np.sqrt(dt_years) * np.random.standard_normal(n_paths)
            log_paths = log_paths + diffusion
            paths[:, idx + 1] = np.exp(log_paths)
        return paths


__all__ = ["PathEngine", "GBMPathEngine", "SimulationContext"]
