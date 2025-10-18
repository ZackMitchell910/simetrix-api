"""Utilities to align simulated variance with market implied volatility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(slots=True)
class AnchorResult:
    sigma_path: np.ndarray
    metadata: dict[str, float]


class ImpliedVolAnchor:
    """Rescale the instantaneous volatility path to match terminal IV."""

    def __init__(self, iv_curve: Mapping[int, float]):
        self.iv_curve = dict(sorted(iv_curve.items()))

    def match(self, horizon_days: int, sigma_path: np.ndarray, dt_years: float) -> AnchorResult:
        target_iv = self._interpolate_iv(horizon_days)
        if target_iv is None:
            return AnchorResult(sigma_path=sigma_path, metadata={"scale": 1.0})

        horizon_years = horizon_days / 252.0
        target_variance = (target_iv ** 2) * horizon_years
        model_variance = float(np.sum((sigma_path ** 2) * dt_years))
        if model_variance <= 0:
            scale = 1.0
        else:
            scale = float(np.sqrt(target_variance / model_variance))
        adjusted = sigma_path * scale
        metadata = {
            "scale": scale,
            "target_iv": target_iv,
            "target_variance": target_variance,
            "model_variance": model_variance,
        }
        return AnchorResult(sigma_path=adjusted, metadata=metadata)

    def _interpolate_iv(self, horizon_days: int) -> float | None:
        if not self.iv_curve:
            return None
        tenors = np.array(list(self.iv_curve.keys()), dtype=float)
        ivs = np.array(list(self.iv_curve.values()), dtype=float)
        if horizon_days <= tenors[0]:
            return float(ivs[0])
        if horizon_days >= tenors[-1]:
            return float(ivs[-1])
        return float(np.interp(horizon_days, tenors, ivs))
