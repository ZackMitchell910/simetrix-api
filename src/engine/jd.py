"""Merton jump-diffusion Monte Carlo engine."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..scenarios.types import EventShock
from .base import PathEngine, SimulationArtifact
from .iv_anchor import ImpliedVolAnchor
from .shocks import ShockScheduler
from .state import StateVector


class JumpDiffusionEngine(PathEngine):
    """Simulate price paths under a jump-diffusion with scenario overrides."""

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
        step_days, dt_years = self._parse_dt(dt)
        time_grid = self._build_time_grid(horizon_days, step_days)
        scheduler = ShockScheduler(state, scenarios)
        overrides = scheduler.build(time_grid, step_days)

        sigma_path = state.vol * overrides.vol_multiplier
        if state.iv_curve:
            anchor = ImpliedVolAnchor(state.iv_curve)
            anchor_result = anchor.match(horizon_days, sigma_path, dt_years)
            sigma_path = anchor_result.sigma_path
            metadata = {"iv_anchor": anchor_result.metadata}
        else:
            metadata = {}

        n_steps = len(time_grid) - 1
        paths = np.empty((n_paths, n_steps + 1), dtype=float)
        paths[:, 0] = state.spot

        for step in range(1, n_steps + 1):
            sigma = float(max(sigma_path[step - 1], 0.0))
            mu = state.drift + overrides.drift_bump[step - 1]
            lam = max(overrides.jump_intensity[step - 1], 0.0)
            jump_mean = overrides.jump_mean[step - 1]
            jump_std = max(overrides.jump_std[step - 1], 0.0)

            expectation_adjustment = (
                mu
                - 0.5 * sigma**2
                - lam * (np.exp(jump_mean + 0.5 * jump_std**2) - 1.0)
            )
            normal = self.random_state.standard_normal(n_paths)
            diffusion = sigma * np.sqrt(dt_years) * normal
            jump_counts = self.random_state.poisson(lam * dt_years, size=n_paths)
            jump_term = np.zeros(n_paths)
            mask = jump_counts > 0
            if np.any(mask):
                jump_term[mask] = self.random_state.normal(
                    loc=jump_mean * jump_counts[mask],
                    scale=jump_std * np.sqrt(jump_counts[mask]),
                )
            increment = expectation_adjustment * dt_years + diffusion + jump_term
            paths[:, step] = paths[:, step - 1] * np.exp(increment)

        return SimulationArtifact(time_grid=time_grid, paths=paths, metadata=metadata)
