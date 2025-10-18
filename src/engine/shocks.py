from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..scenarios.models import EventShock


@dataclass(slots=True)
class ShockOverrides:
    drift_bump: np.ndarray
    vol_multiplier: np.ndarray
    jump_intensity: np.ndarray
    jump_mean: np.ndarray
    jump_std: np.ndarray


class ShockScheduler:
    """Translate ``EventShock`` definitions into per-step overrides."""

    def build(self, time_grid: np.ndarray, shocks: Sequence[EventShock]) -> ShockOverrides:
        n_steps = len(time_grid) - 1
        drift_bump = np.zeros(n_steps, dtype=float)
        vol_multiplier = np.ones(n_steps, dtype=float)
        jump_intensity = np.zeros(n_steps, dtype=float)
        jump_mean = np.zeros(n_steps, dtype=float)
        jump_std = np.zeros(n_steps, dtype=float)

        if not shocks:
            return ShockOverrides(drift_bump, vol_multiplier, jump_intensity, jump_mean, jump_std)

        step_midpoints = time_grid[:-1]
        for shock in shocks:
            start = float(shock.window_start)
            end = float(shock.window_end)
            mask = (step_midpoints >= start) & (step_midpoints <= end)
            indices = np.where(mask)[0]
            for idx in indices:
                drift_bump[idx] += shock.drift
                vol_multiplier[idx] *= max(shock.vol_multiplier, 0.0)
                base_intensity = jump_intensity[idx]
                new_intensity = base_intensity + max(shock.jump_intensity, 0.0)
                if new_intensity > 0.0:
                    jump_mean[idx] = self._combine_means(
                        base_intensity,
                        jump_mean[idx],
                        shock.jump_intensity,
                        shock.jump_mean,
                    )
                    jump_std[idx] = self._combine_std(
                        base_intensity,
                        jump_std[idx],
                        shock.jump_intensity,
                        shock.jump_std,
                    )
                    jump_intensity[idx] = new_intensity
        return ShockOverrides(drift_bump, vol_multiplier, jump_intensity, jump_mean, jump_std)

    @staticmethod
    def _combine_means(w1: float, m1: float, w2: float, m2: float) -> float:
        total = w1 + w2
        if total == 0:
            return 0.0
        return (w1 * m1 + w2 * m2) / total

    @staticmethod
    def _combine_std(w1: float, s1: float, w2: float, s2: float) -> float:
        total = w1 + w2
        if total == 0:
            return 0.0
        variance = 0.0
        if w1 > 0:
            variance += w1 * (s1 ** 2)
        if w2 > 0:
            variance += w2 * (s2 ** 2)
        return np.sqrt(variance / total)


__all__ = ["ShockScheduler", "ShockOverrides"]
