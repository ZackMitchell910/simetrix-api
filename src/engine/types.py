from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np


@dataclass
class StateVector:
    """Compact representation of the market state driving simulations."""

    spot: float | Sequence[float]
    drift: float
    vol: float
    as_of: Optional[object] = None
    correlation: Optional[np.ndarray] = None
    iv_curve: Optional[Mapping[int, float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_array(self) -> np.ndarray:
        return np.atleast_1d(np.asarray(self.spot, dtype=float))

    def get_iv(self, horizon_days: int | float) -> Optional[float]:
        if not self.iv_curve:
            return None
        pairs = sorted((float(k), float(v)) for k, v in self.iv_curve.items())
        if not pairs:
            return None
        horizon = float(horizon_days)
        if horizon <= pairs[0][0]:
            return pairs[0][1]
        if horizon >= pairs[-1][0]:
            return pairs[-1][1]
        lower_pair = max((pair for pair in pairs if pair[0] <= horizon), key=lambda item: item[0])
        upper_pair = min((pair for pair in pairs if pair[0] >= horizon), key=lambda item: item[0])
        if lower_pair[0] == upper_pair[0]:
            return lower_pair[1]
        weight = (horizon - lower_pair[0]) / (upper_pair[0] - lower_pair[0])
        return lower_pair[1] + weight * (upper_pair[1] - lower_pair[1])


@dataclass
class Artifact:
    """Simulation artefact containing price paths and diagnostics."""

    paths: np.ndarray
    time_grid: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "n_paths": int(self.paths.shape[0]),
            "n_steps": int(self.paths.shape[1] - 1),
            "horizon_days": float(self.time_grid[-1] - self.time_grid[0]) if len(self.time_grid) else 0.0,
            **self.metadata,
        }
