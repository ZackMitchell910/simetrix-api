from __future__ import annotations

import numpy as np

from .base import PathEngine, SimulationContext
from .types import StateVector


class HestonPathEngine(PathEngine):
    """Simple Euler discretisation of the Heston stochastic volatility model."""

    def __init__(self, kappa: float = 2.0, theta: float = 0.04, xi: float = 0.5, rho: float = -0.5, **kwargs):
        super().__init__(**kwargs)
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho

    def _simulate_paths(self, state: StateVector, context: SimulationContext, n_paths: int) -> np.ndarray:
        spot = float(state.as_array()[0])
        n_steps = len(context.step_days)
        paths = np.empty((n_paths, n_steps + 1), dtype=float)
        paths[:, 0] = spot
        log_paths = np.full(n_paths, np.log(spot), dtype=float)

        var = np.full(n_paths, max(state.vol, 1e-6) ** 2, dtype=float)
        for idx, step in enumerate(context.step_days):
            dt_years = step / 252.0
            drift = state.drift + context.overrides.drift_bump[idx]

            z1 = np.random.standard_normal(n_paths)
            z2 = np.random.standard_normal(n_paths)
            w1 = z1
            w2 = self.rho * z1 + np.sqrt(1 - self.rho ** 2) * z2

            var = np.maximum(
                var
                + self.kappa * (self.theta - var) * dt_years
                + self.xi * np.sqrt(np.maximum(var, 1e-8)) * np.sqrt(dt_years) * w2,
                1e-10,
            )
            vol = np.sqrt(var) * context.overrides.vol_multiplier[idx]
            diffusion = (drift - 0.5 * vol ** 2) * dt_years + vol * np.sqrt(dt_years) * w1
            log_paths += diffusion
            paths[:, idx + 1] = np.exp(log_paths)
        return paths


__all__ = ["HestonPathEngine"]
