"""Helpers for matching terminal variance with market implied volatility."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


TRADING_DAYS = 252


@dataclass
class IVAnchor:
    """Anchor simulated variance to an implied volatility term structure."""

    iv_surface: Mapping[int, float] | None

    def interpolated_iv(self, horizon_days: int) -> float | None:
        if not self.iv_surface:
            return None
        tenors = sorted(int(k) for k in self.iv_surface.keys())
        if not tenors:
            return None
        if horizon_days <= tenors[0]:
            return float(self.iv_surface[tenors[0]])
        if horizon_days >= tenors[-1]:
            return float(self.iv_surface[tenors[-1]])
        for left, right in zip(tenors[:-1], tenors[1:]):
            if left <= horizon_days <= right:
                left_iv = float(self.iv_surface[left])
                right_iv = float(self.iv_surface[right])
                weight = (horizon_days - left) / (right - left)
                return left_iv + weight * (right_iv - left_iv)
        return float(self.iv_surface[tenors[-1]])

    def variance_scale(
        self,
        sigma_path: Sequence[float],
        step_days: int,
        horizon_days: int,
    ) -> float:
        implied_iv = self.interpolated_iv(horizon_days)
        if implied_iv is None:
            return 1.0
        realised_variance = 0.0
        dt = step_days / TRADING_DAYS
        for sigma in sigma_path:
            realised_variance += (sigma ** 2) * dt
        target_variance = (implied_iv ** 2) * (horizon_days / TRADING_DAYS)
        if realised_variance <= 0 or target_variance <= 0:
            return 1.0
        return math.sqrt(target_variance / realised_variance)

    def anchor_sigma_path(
        self,
        sigma_path: Sequence[float],
        step_days: int,
        horizon_days: int,
    ) -> np.ndarray:
        scale = self.variance_scale(sigma_path, step_days, horizon_days)
        return np.asarray(sigma_path, dtype=float) * scale


__all__ = ["IVAnchor"]
