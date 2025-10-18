from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .types import StateVector


@dataclass(slots=True)
class ImpliedVolAnchor:
    """Scales simulated paths so the terminal variance matches market IV."""

    tolerance: float = 0.1

    def anchor(self, paths: np.ndarray, state: StateVector, time_grid: np.ndarray, overrides) -> np.ndarray:
        if state.iv_curve is None or paths.size == 0:
            return paths
        horizon_days = time_grid[-1]
        target_iv = state.get_iv(horizon_days)
        if target_iv is None:
            return paths
        dt_years = horizon_days / 252.0
        if dt_years <= 0:
            return paths

        log_paths = np.log(paths / paths[:, [0]])
        terminal = log_paths[:, -1]
        actual_var = np.var(terminal)
        target_var = (target_iv ** 2) * dt_years
        if actual_var <= 0:
            return paths
        scale = np.sqrt(target_var / actual_var)
        if not np.isfinite(scale) or scale <= 0:
            return paths
        adjusted_log_paths = log_paths * scale
        adjusted_log_paths[:, 0] = 0.0
        anchored_paths = np.exp(adjusted_log_paths) * paths[:, [0]]
        return anchored_paths


__all__ = ["ImpliedVolAnchor"]
