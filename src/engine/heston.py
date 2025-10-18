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
        self.year_basis = year_basis

    def simulate(
        self,
        state: StateVector,
        scenarios: Sequence[EventShock],
        horizon_days: int,
        n_paths: int,
        dt: str = "1d",
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
