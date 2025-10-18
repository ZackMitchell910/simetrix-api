from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import GBMPathEngine, SimulationContext
from .types import Artifact, StateVector


class CorrelatedPathEngine(GBMPathEngine):
    """Generates correlated asset paths using Cholesky factorisation."""
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
        scenarios,
        scenarios: Sequence[EventShock],
        horizon_days: int,
        n_paths: int,
        dt: str = "1d",
    ) -> Artifact:
        correlation = state.correlation
        if correlation is None:
            raise ValueError("Correlation matrix required for correlated simulations")
        if correlation.shape[0] != correlation.shape[1]:
            raise ValueError("Correlation matrix must be square")
        self._cholesky = np.linalg.cholesky(correlation)
        artifact = super().simulate(state=state, scenarios=scenarios, horizon_days=horizon_days, n_paths=n_paths, dt=dt)
        del self._cholesky
        return artifact

    def _simulate_paths(self, state: StateVector, context: SimulationContext, n_paths: int) -> np.ndarray:
        spots = state.as_array()
        n_assets = spots.size
        n_steps = len(context.step_days)
        paths = np.empty((n_paths, n_steps + 1, n_assets), dtype=float)
        paths[:, 0, :] = spots
        log_paths = np.log(spots)[None, :].repeat(n_paths, axis=0)

        chol = getattr(self, "_cholesky", np.eye(n_assets))

        for idx, step in enumerate(context.step_days):
            dt_years = step / 252.0
            drift = state.drift + context.overrides.drift_bump[idx]
            vol = state.vol * context.overrides.vol_multiplier[idx]
            normals = np.random.standard_normal((n_paths, n_assets)) @ chol.T
            diffusion = (drift - 0.5 * vol ** 2) * dt_years + vol * np.sqrt(dt_years) * normals
            log_paths = log_paths + diffusion
            paths[:, idx + 1, :] = np.exp(log_paths)
        if n_assets == 1:
            return paths[:, :, 0]
        return paths


__all__ = ["CorrelatedPathEngine"]
        artifact = self.base_engine.simulate(state, scenarios, horizon_days, n_paths, dt)
        artifact.metadata.setdefault("correlations", {})[state.symbol] = self.correlation
        artifact.metadata.setdefault("engines", {})[state.symbol] = artifact.metadata.get("engine")
        artifact.metadata.setdefault("symbols", []).append(state.symbol)
        return artifact
