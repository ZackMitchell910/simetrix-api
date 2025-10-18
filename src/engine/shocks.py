"""Shock scheduling utilities used by simulation engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

import numpy as np

from src.scenarios.models import EventShock


@dataclass
class ShockApplication:
    """Compiled set of adjustments aligned with the simulation grid."""

    drift: np.ndarray
    volatility: np.ndarray
    jump_intensity: np.ndarray
    jump_mean: np.ndarray
    jump_std: np.ndarray
    metadata: list[dict[str, Any]] = field(default_factory=list)

    def effective_sigma(self, base_sigma: float) -> np.ndarray:
        return base_sigma * self.volatility


class ShockScheduler:
    """Project discrete EventShock objects onto the simulation grid."""

    def compile(
        self,
        scenarios: Sequence[EventShock],
        time_grid: Sequence[datetime],
    ) -> ShockApplication:
        if len(time_grid) < 2:
            raise ValueError("time_grid must contain at least two points")

        n_steps = len(time_grid) - 1
        drift = np.zeros(n_steps, dtype=float)
        volatility = np.ones(n_steps, dtype=float)
        jump_intensity = np.zeros(n_steps, dtype=float)
        jump_mean = np.zeros(n_steps, dtype=float)
        jump_std = np.zeros(n_steps, dtype=float)
        metadata: list[dict[str, Any]] = []

        for shock in scenarios:
            if shock.window_end < time_grid[0] or shock.window_start > time_grid[-1]:
                continue
            indices = self._indices_for(shock, time_grid)
            if not indices.size:
                continue
            drift[indices] += shock.drift * shock.prior
            volatility[indices] *= max(shock.volatility_scale, 0.0)
            jump_intensity[indices] += max(shock.jump_intensity, 0.0) * shock.prior
            jump_mean[indices] += shock.jump_mean * shock.prior
            jump_std[indices] = np.maximum(jump_std[indices], shock.jump_std)
            metadata.append(
                {
                    "label": shock.label,
                    "variant": shock.variant,
                    "prior": shock.prior,
                    "indices": indices.tolist(),
                }
            )

        return ShockApplication(
            drift=drift,
            volatility=volatility,
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_std=jump_std,
            metadata=metadata,
        )

    def _indices_for(self, shock: EventShock, time_grid: Sequence[datetime]) -> np.ndarray:
        start = shock.window_start
        end = shock.window_end
        mask = []
        for idx in range(len(time_grid) - 1):
            left = time_grid[idx]
            right = time_grid[idx + 1]
            if right <= start or left >= end:
                mask.append(False)
            else:
                mask.append(True)
        return np.nonzero(mask)[0]


__all__ = ["ShockScheduler", "ShockApplication"]
