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
    """Translate scenario windows into per-step overrides."""

    def __init__(self, schedule: ShockSchedule):
        self.schedule = schedule

    @classmethod
    def from_scenarios(
        cls,
        scenarios: Sequence[EventShock],
        state: StateVector,
        times: np.ndarray,
    ) -> "ShockScheduler":
        n_steps = len(times) - 1
        drift = np.zeros(n_steps, dtype=float)
        vol = np.ones(n_steps, dtype=float)
        jump_intensity = np.zeros(n_steps, dtype=float)
        jump_mean = np.zeros(n_steps, dtype=float)
        jump_std = np.zeros(n_steps, dtype=float)

        for shock in scenarios:
            start_idx, end_idx = _window_to_index(shock, state.asof, times)
            if start_idx >= end_idx:
                continue
            override = shock.override
            drift[start_idx:end_idx] += override.drift_bump
            vol[start_idx:end_idx] *= override.vol_multiplier
            jump_intensity[start_idx:end_idx] += override.jump_intensity
            jump_mean[start_idx:end_idx] += override.jump_mean
            jump_std[start_idx:end_idx] = np.maximum(
                jump_std[start_idx:end_idx], override.jump_std
            )

        schedule = ShockSchedule(
            times=np.asarray(times, dtype=float),
            drift=drift,
            vol_multiplier=vol,
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_std=jump_std,
        )
        return cls(schedule)

    def overrides(self) -> ShockSchedule:
        return self.schedule


def _window_to_index(shock: EventShock, asof: datetime, times: np.ndarray) -> tuple[int, int]:
    start = max(0.0, _days_between(asof, shock.window_start))
    end = max(0.0, _days_between(asof, shock.window_end))
    idx_start = int(np.searchsorted(times, start, side="left"))
    idx_end = int(np.searchsorted(times, end, side="right"))
    idx_start = min(idx_start, len(times) - 1)
    idx_end = min(max(idx_end, idx_start + 1), len(times) - 1)
    return idx_start, idx_end


def _days_between(asof: datetime, ts: datetime) -> float:
    delta = ts - asof
    return delta.total_seconds() / 86400.0
