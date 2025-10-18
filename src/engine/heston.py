"""Simplified Heston stochastic volatility engine."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.scenarios.models import EventShock

from .base import Artifact, PathEngine, StateVector
from .iv_anchor import IVAnchor, TRADING_DAYS


class HestonEngine(PathEngine):
    """Euler-Maruyama discretisation of the Heston model."""

    def __init__(
        self,
        *,
        kappa: float = 1.5,
        theta: float | None = None,
        xi: float = 0.4,
        rho: float = -0.5,
        scheduler=None,
    ) -> None:
        super().__init__(scheduler=scheduler)
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho

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

        anchor = IVAnchor(state.iv_surface)
        sigma_template = adjustments.effective_sigma(state.sigma)
        anchor_scale = anchor.variance_scale(sigma_template, step_days, horizon_days)
        theta = (self.theta if self.theta is not None else state.sigma ** 2) * anchor_scale ** 2

        rng = random_state or np.random.default_rng()
        n_steps = len(time_grid) - 1
        paths = np.empty((n_paths, len(time_grid)), dtype=float)
        variance = np.empty_like(paths)
        variance[:, 0] = (state.sigma * anchor_scale) ** 2
        paths[:, 0] = state.spot

        annual_drift = state.annualised_drift()
        dt_years = step_days / TRADING_DAYS
        sqrt_dt = np.sqrt(dt_years)
        rho = np.clip(self.rho, -0.999, 0.999)
        rho_perp = np.sqrt(1.0 - rho ** 2)

        for step in range(n_steps):
            v_prev = np.maximum(variance[:, step], 1e-12)
            z1 = rng.standard_normal(n_paths)
            z2 = rng.standard_normal(n_paths)
            w1 = z1
            w2 = rho * z1 + rho_perp * z2

            sigma = np.sqrt(v_prev)
            sigma *= adjustments.volatility[step]
            drift = annual_drift + adjustments.drift[step]

            diffusion = sigma * sqrt_dt * w1
            jump_component = np.zeros(n_paths, dtype=float)
            jump_lambda = max(adjustments.jump_intensity[step], 0.0)
            jump_mu = adjustments.jump_mean[step]
            jump_sigma = adjustments.jump_std[step]
            if jump_lambda > 0:
                poisson = rng.poisson(jump_lambda * dt_years, size=n_paths)
                mask = poisson > 0
                if np.any(mask):
                    mean = jump_mu * poisson[mask]
                    std = jump_sigma * np.sqrt(poisson[mask])
                    if np.any(std > 0):
                        jump_component[mask] = rng.normal(mean, std)
                    else:
                        jump_component[mask] = mean

            drift_term = (drift - 0.5 * sigma ** 2) * dt_years
            paths[:, step + 1] = paths[:, step] * np.exp(drift_term + diffusion + jump_component)

            variance[:, step + 1] = np.maximum(
                v_prev
                + self.kappa * (theta - v_prev) * dt_years
                + self.xi * np.sqrt(v_prev) * sqrt_dt * w2,
                1e-12,
            )

        metadata = {
            "scheduler": adjustments.metadata,
            "dt": dt,
            "anchor_scale": anchor_scale,
            "kappa": self.kappa,
            "xi": self.xi,
            "rho": rho,
        }
        return Artifact(paths=paths, time_grid=time_grid, metadata=metadata)


__all__ = ["HestonEngine"]
