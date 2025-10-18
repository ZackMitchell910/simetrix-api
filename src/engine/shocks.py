from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np

from src.scenarios.models import EventShock
from .types import StateVector


@dataclass(slots=True)
class ShockSchedule:
    times: np.ndarray
    drift: np.ndarray
    vol_multiplier: np.ndarray
    jump_intensity: np.ndarray
    jump_mean: np.ndarray
    jump_std: np.ndarray


class ShockScheduler:
    """Translate high level scenarios into per-step overrides."""

    def __init__(self, state: StateVector, scenarios: Sequence[EventShock]):
        self.state = state
        self.scenarios = list(scenarios)

    def build(self, time_grid: np.ndarray, step_days: int) -> ScheduledShockPath:
        n_steps = len(time_grid) - 1
        drift = np.zeros(n_steps)
        vol = np.ones(n_steps)
        jump_intensity = np.full(n_steps, max(self.state.jump_intensity, 0.0))
        jump_mean_num = np.full(
            n_steps,
            self.state.jump_intensity * self.state.jump_mean,
        )
        jump_second_moment = np.full(
            n_steps,
            self.state.jump_intensity
            * (self.state.jump_std**2 + self.state.jump_mean**2),
        )

        if not self.scenarios:
            jump_mean = np.divide(
                jump_mean_num,
                np.maximum(jump_intensity, 1e-12),
                out=np.zeros_like(jump_mean_num),
                where=jump_intensity > 0,
            )
            jump_var = np.maximum(
                jump_second_moment / np.maximum(jump_intensity, 1e-12)
                - jump_mean**2,
                0.0,
            )
            jump_std = np.sqrt(jump_var)
            return ScheduledShockPath(drift, vol, jump_intensity, jump_mean, jump_std)

        as_of = self.state.as_of
        time_points = [as_of + timedelta(days=float(day)) for day in time_grid]

        for shock in self.scenarios:
            start_idx = self._locate_index(time_points, shock.window_start)
            end_idx = self._locate_index(time_points, shock.window_end)
            if end_idx <= start_idx:
                end_idx = min(start_idx + 1, n_steps)
            for idx in range(start_idx, min(end_idx, n_steps)):
                drift[idx] += shock.drift_bump
                vol[idx] *= max(shock.vol_multiplier, 0.01)
                jump_intensity[idx] += max(shock.jump_intensity, 0.0)
                weight = max(shock.jump_intensity, 0.0)
                jump_mean_num[idx] += weight * shock.jump_mean
                jump_second_moment[idx] += weight * (
                    shock.jump_std**2 + shock.jump_mean**2
                )

        jump_mean = np.zeros(n_steps)
        jump_std = np.zeros(n_steps)
        for idx in range(n_steps):
            lam = jump_intensity[idx]
            if lam <= 0:
                continue
            mean = jump_mean_num[idx] / lam
            second_moment = jump_second_moment[idx] / lam
            var = max(second_moment - mean**2, 0.0)
            jump_mean[idx] = mean
            jump_std[idx] = np.sqrt(var)

        return ScheduledShockPath(drift, vol, jump_intensity, jump_mean, jump_std)

    @staticmethod
    def _locate_index(time_points: Sequence[datetime], timestamp: datetime) -> int:
        for idx, point in enumerate(time_points):
            if point >= timestamp:
                return max(idx - 1, 0)
        return max(len(time_points) - 2, 0)
