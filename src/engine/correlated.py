from __future__ import annotations

from typing import Sequence

import numpy as np

from ..scenarios.types import EventShock
from .base import PathEngine, SimulationArtifact
from .shocks import ShockScheduler
from .state import StateVector
from src.scenarios.models import EventShock
from .base import PathEngine
from .types import Artifact, StateVector

class CorrelatedPathEngine(PathEngine):
    """Generate correlated jump-diffusion paths for multiple assets."""

    def __init__(self, random_state: np.random.Generator | None = None) -> None:
        self.random_state = random_state or np.random.default_rng()

    def simulate(
        self,
        state: StateVector,
        scenarios: Sequence[EventShock],
        horizon_days: int,
        n_paths: int,
        dt: str = "1d",
    ) -> SimulationArtifact:
        if state.corr is None:
            raise ValueError("Correlation matrix must be provided for correlated engine")

        corr = np.asarray(state.corr, dtype=float)
        if corr.shape[0] != corr.shape[1]:
            raise ValueError("Correlation matrix must be square")
        n_assets = corr.shape[0]

        chol = np.linalg.cholesky(corr)

        spot = np.broadcast_to(np.asarray(state.spot, dtype=float), (n_assets,))
        drift = np.broadcast_to(np.asarray(state.drift, dtype=float), (n_assets,))
        vol = np.broadcast_to(np.asarray(state.vol, dtype=float), (n_assets,))

        step_days, dt_years = self._parse_dt(dt)
        time_grid = self._build_time_grid(horizon_days, step_days)
        scheduler = ShockScheduler(state, scenarios)
        overrides = scheduler.build(time_grid, step_days)

        n_steps = len(time_grid) - 1
        paths = np.empty((n_paths, n_steps + 1, n_assets), dtype=float)
        paths[:, 0, :] = spot

        lam_path = np.maximum(overrides.jump_intensity, 0.0)
        jump_mean_path = overrides.jump_mean
        jump_std_path = overrides.jump_std

        for step in range(1, n_steps + 1):
            mu = drift + overrides.drift_bump[step - 1]
            sigma = vol * overrides.vol_multiplier[step - 1]

            normals = self.random_state.standard_normal((n_paths, n_assets)) @ chol.T
            diffusion = sigma * np.sqrt(dt_years) * normals

            lam = lam_path[step - 1]
            jump_mean = np.broadcast_to(jump_mean_path[step - 1], (n_assets,))
            jump_std = np.broadcast_to(jump_std_path[step - 1], (n_assets,))

            jump_counts = self.random_state.poisson(lam * dt_years, size=(n_paths, n_assets))
            jump_term = np.zeros((n_paths, n_assets))
            jump_mask = jump_counts > 0
            if np.any(jump_mask):
                jump_mean_expanded = np.broadcast_to(jump_mean, (n_paths, n_assets))
                jump_std_expanded = np.broadcast_to(jump_std, (n_paths, n_assets))
                jump_term[jump_mask] = self.random_state.normal(
                    loc=jump_mean_expanded[jump_mask] * jump_counts[jump_mask],
                    scale=jump_std_expanded[jump_mask]
                    * np.sqrt(jump_counts[jump_mask]),
                )

            expectation_adjustment = (
                mu
                - 0.5 * sigma**2
                - lam
                * (np.exp(jump_mean + 0.5 * jump_std**2) - 1.0)
            )
            increment = expectation_adjustment * dt_years + diffusion + jump_term
            paths[:, step, :] = paths[:, step - 1, :] * np.exp(increment)

        metadata = {"correlation": corr}
        return SimulationArtifact(time_grid=time_grid, paths=paths, metadata=metadata)
    ) -> Artifact:
        artifact = self.base_engine.simulate(state, scenarios, horizon_days, n_paths, dt)
        artifact.metadata.setdefault("correlations", {})[state.symbol] = self.correlation
        artifact.metadata.setdefault("engines", {})[state.symbol] = artifact.metadata.get("engine")
        artifact.metadata.setdefault("symbols", []).append(state.symbol)
        return artifact
