from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from ..scenarios.types import EventShock
from .base import PathEngine, SimulationArtifact
from .jd import JumpDiffusionEngine
from .shocks import ShockScheduler
from .state import StateVector

def _parse_dt(dt: str) -> float:
    if dt.endswith("d"):
        return float(dt[:-1] or 1.0)
    if dt.endswith("h"):
        return float(dt[:-1]) / 24.0
    raise ValueError(f"Unsupported dt format: {dt}")
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
    ) -> Artifact:
        step_days = _parse_dt(dt)
        n_steps = max(1, math.ceil(horizon_days / step_days))
        times = np.linspace(0.0, n_steps * step_days, n_steps + 1)
        scheduler = ShockScheduler.from_scenarios(scenarios, state, times)
        overrides = scheduler.overrides()

        rng = state.rng or np.random.default_rng()
        paths = np.empty((n_paths, n_steps + 1), dtype=float)
        variances = np.empty_like(paths)
        paths[:, 0] = state.spot
        variances[:, 0] = max(state.vol ** 2, 1e-8)

        dt_years = step_days / self.year_basis
        sqrt_dt = math.sqrt(dt_years)

        for idx in range(1, n_steps + 1):
            v_prev = variances[:, idx - 1]
            drift_adj = overrides.drift[idx - 1]
            vol_mult = overrides.vol_multiplier[idx - 1]
            vol = np.sqrt(np.maximum(v_prev, 1e-8)) * vol_mult

            z1 = rng.standard_normal(n_paths)
            z2 = rng.standard_normal(n_paths)
            w1 = z1
            w2 = self.rho * z1 + math.sqrt(1.0 - self.rho ** 2) * z2

            v_next = (
                np.maximum(
                    v_prev
                    + self.kappa * (self.theta - v_prev) * dt_years
                    + self.xi * np.sqrt(np.maximum(v_prev, 0.0)) * sqrt_dt * w2,
                    1e-8,
                )
            )
            sigma_t = np.sqrt(v_next)
            drift = (state.drift + drift_adj - 0.5 * sigma_t ** 2) * dt_years
            diffusion = sigma_t * sqrt_dt * w1
            paths[:, idx] = paths[:, idx - 1] * np.exp(drift + diffusion)
            variances[:, idx] = v_next

        artifact = Artifact(times=times, paths=paths)
        artifact.metadata.update(
            {
                "engine": "heston",
                "dt_days": step_days,
                "year_basis": self.year_basis,
                "params": {
                    "kappa": self.kappa,
                    "theta": self.theta,
                    "xi": self.xi,
                    "rho": self.rho,
                },
            }
        )
        return artifact
