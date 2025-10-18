from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from src.scenarios.models import EventShock
from .base import PathEngine
from .iv_anchor import anchor_to_implied_vol
from .shocks import ShockScheduler
from .types import Artifact, StateVector


def _parse_dt(dt: str) -> float:
    if dt.endswith("d"):
        return float(dt[:-1] or 1.0)
    if dt.endswith("h"):
        return float(dt[:-1]) / 24.0
    raise ValueError(f"Unsupported dt format: {dt}")


class JumpDiffusionEngine(PathEngine):
    """Merton-style jump diffusion with scenario hooks."""

    def __init__(self, year_basis: float = 252.0):
        self.year_basis = year_basis

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
    ) -> Artifact:
        step_days = _parse_dt(dt)
        n_steps = max(1, math.ceil(horizon_days / step_days))
        times = np.linspace(0.0, step_days * n_steps, n_steps + 1)
        scheduler = ShockScheduler.from_scenarios(scenarios, state, times)
        overrides = scheduler.overrides()

        rng = state.rng or np.random.default_rng()
        paths = np.empty((n_paths, n_steps + 1), dtype=float)
        paths[:, 0] = state.spot

        base_mu = state.drift
        base_sigma = state.vol
        base_lambda = max(0.0, state.jump_intensity)
        base_jump_mean = state.jump_mean
        base_jump_std = max(0.0, state.jump_std)

        dt_years = step_days / self.year_basis
        sqrt_dt = math.sqrt(dt_years)

        for idx in range(1, n_steps + 1):
            drift = base_mu + overrides.drift[idx - 1]
            sigma = max(1e-8, base_sigma * overrides.vol_multiplier[idx - 1])
            lam = max(0.0, base_lambda + overrides.jump_intensity[idx - 1])
            jump_mean = base_jump_mean + overrides.jump_mean[idx - 1]
            jump_std = max(base_jump_std, overrides.jump_std[idx - 1])

            z = rng.standard_normal(n_paths)
            diffusion = (drift - 0.5 * sigma ** 2) * dt_years + sigma * sqrt_dt * z

            if lam > 0.0:
                n_jumps = rng.poisson(lam * dt_years, size=n_paths)
                if np.any(n_jumps):
                    jump_draws = rng.normal(loc=jump_mean, scale=max(1e-8, jump_std), size=n_paths)
                    jump_component = jump_draws * n_jumps
                else:
                    jump_component = 0.0
            else:
                jump_component = 0.0

            paths[:, idx] = paths[:, idx - 1] * np.exp(diffusion + jump_component)

        artifact = Artifact(times=times, paths=paths)
        anchor_to_implied_vol(artifact, state)
        artifact.metadata.update(
            {
                "engine": "jump_diffusion",
                "dt_days": step_days,
                "year_basis": self.year_basis,
                "scenarios": [shock.variant for shock in scenarios],
            }
        )
        return artifact
