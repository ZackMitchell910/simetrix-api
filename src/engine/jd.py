"""Merton-style jump diffusion path engine."""

from __future__ import annotations

from math import exp
from typing import Sequence

import numpy as np

from src.scenarios.models import EventShock

from .base import Artifact, PathEngine, StateVector
from .iv_anchor import IVAnchor, TRADING_DAYS


class JumpDiffusionEngine(PathEngine):
    """Simulate price paths under a jump-diffusion process."""

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
        step_days = self._parse_dt(dt)
        time_grid = self._time_grid(state.asof, horizon_days, step_days)
        scheduler = self._ensure_scheduler()
        adjustments = scheduler.compile(scenarios, time_grid)

        adjusted_sigma = adjustments.effective_sigma(state.sigma)
        anchor = IVAnchor(state.iv_surface)
        scale = anchor.variance_scale(adjusted_sigma, step_days, horizon_days)
        sigma_path = adjusted_sigma * scale

        rng = random_state or np.random.default_rng()
        paths = np.empty((n_paths, len(time_grid)), dtype=float)
        paths[:, 0] = state.spot

        annual_drift = state.annualised_drift()
        dt_years = step_days / TRADING_DAYS

        for step in range(len(time_grid) - 1):
            sigma = sigma_path[step]
            drift = annual_drift + adjustments.drift[step]
            jump_lambda = max(adjustments.jump_intensity[step], 0.0)
            jump_mu = adjustments.jump_mean[step]
            jump_sigma = adjustments.jump_std[step]
            if jump_lambda > 0:
                kappa = exp(jump_mu + 0.5 * jump_sigma ** 2) - 1.0
            else:
                kappa = 0.0

            diffusion = sigma * np.sqrt(dt_years) * rng.standard_normal(n_paths)
            if jump_lambda > 0:
                poisson = rng.poisson(jump_lambda * dt_years, size=n_paths)
                jump_component = np.zeros(n_paths, dtype=float)
                mask = poisson > 0
                if np.any(mask):
                    mean = jump_mu * poisson[mask]
                    std = jump_sigma * np.sqrt(poisson[mask])
                    if np.any(std > 0):
                        jump_component[mask] = rng.normal(mean, std)
                    else:
                        jump_component[mask] = mean
            else:
                poisson = None
                jump_component = 0.0

            drift_term = (drift - 0.5 * sigma ** 2 - jump_lambda * kappa) * dt_years
            paths[:, step + 1] = paths[:, step] * np.exp(drift_term + diffusion + jump_component)

        metadata = {
            "scheduler": adjustments.metadata,
            "dt": dt,
            "anchor_scale": scale,
        }
        return Artifact(paths=paths, time_grid=time_grid, metadata=metadata)


__all__ = ["JumpDiffusionEngine"]
