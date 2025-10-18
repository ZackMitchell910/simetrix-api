from __future__ import annotations

import numpy as np

from ..scenarios.models import EventShock
from .base import GBMPathEngine, SimulationContext
from .types import StateVector


class MertonJumpDiffusionEngine(GBMPathEngine):
    """Merton-style jump diffusion with scenario-driven overrides."""

    def _simulate_paths(self, state: StateVector, context: SimulationContext, n_paths: int) -> np.ndarray:
        spot = float(state.as_array()[0])
        n_steps = len(context.step_days)
        paths = np.empty((n_paths, n_steps + 1), dtype=float)
        paths[:, 0] = spot
        log_paths = np.full(n_paths, np.log(spot), dtype=float)

        for idx, step in enumerate(context.step_days):
            dt_years = step / 252.0
            drift = state.drift + context.overrides.drift_bump[idx]
            vol = state.vol * context.overrides.vol_multiplier[idx]
            normals = np.random.standard_normal(n_paths)
            diffusion = (drift - 0.5 * vol ** 2) * dt_years + vol * np.sqrt(dt_years) * normals
            log_paths += diffusion

            lam = max(context.overrides.jump_intensity[idx], 0.0) * dt_years
            if lam > 0:
                n_jumps = np.random.poisson(lam, size=n_paths)
                if np.any(n_jumps):
                    mean = context.overrides.jump_mean[idx]
                    std = max(context.overrides.jump_std[idx], 1e-6)
                    jump_component = np.random.normal(mean, std, size=n_paths) * n_jumps
                    log_paths += jump_component

            paths[:, idx + 1] = np.exp(log_paths)
        return paths


__all__ = ["MertonJumpDiffusionEngine"]
