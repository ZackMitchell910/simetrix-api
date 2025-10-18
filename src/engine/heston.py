"""Optional Heston stochastic volatility engine."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..scenarios.types import EventShock
from .base import PathEngine, SimulationArtifact
from .jd import JumpDiffusionEngine
from .shocks import ShockScheduler
from .state import StateVector


class HestonEngine(PathEngine):
    """Simple Euler discretization of the Heston model with shocks."""

    def __init__(
        self,
        kappa: float = 2.0,
        theta: float = 0.04,
        vol_of_vol: float = 0.5,
        rho: float = -0.5,
        random_state: np.random.Generator | None = None,
    ) -> None:
        self.kappa = kappa
        self.theta = theta
        self.vol_of_vol = vol_of_vol
        self.rho = rho
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

        n_steps = len(time_grid) - 1
        paths = np.empty((n_paths, n_steps + 1), dtype=float)
        variance = np.empty((n_paths, n_steps + 1), dtype=float)
        paths[:, 0] = state.spot
        variance[:, 0] = max(state.vol, 1e-6) ** 2

        for step in range(1, n_steps + 1):
            prev_var = variance[:, step - 1]
            sigma = np.sqrt(np.maximum(prev_var, 1e-12))
            mu = state.drift + overrides.drift_bump[step - 1]
            lam = max(overrides.jump_intensity[step - 1], 0.0)
            jump_mean = overrides.jump_mean[step - 1]
            jump_std = max(overrides.jump_std[step - 1], 0.0)

            z1 = self.random_state.standard_normal(n_paths)
            z2 = self.random_state.standard_normal(n_paths)
            w1 = z1
            w2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2

            variance[:, step] = np.maximum(
                prev_var
                + self.kappa * (self.theta - prev_var) * dt_years
                + self.vol_of_vol * sigma * np.sqrt(dt_years) * w2,
                0.0,
            )
            sigma_inst = np.sqrt(variance[:, step]) * overrides.vol_multiplier[step - 1]

            expectation_adjustment = (
                mu
                - 0.5 * sigma_inst**2
                - lam * (np.exp(jump_mean + 0.5 * jump_std**2) - 1.0)
            )
            diffusion = sigma_inst * np.sqrt(dt_years) * w1
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

        metadata = {"variance_path": variance}
        return SimulationArtifact(time_grid=time_grid, paths=paths, metadata=metadata)

    def to_jump_diffusion(self) -> JumpDiffusionEngine:
        """Return a jump diffusion engine seeded with the same RNG."""

        return JumpDiffusionEngine(random_state=self.random_state)
