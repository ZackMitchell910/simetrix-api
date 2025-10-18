from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, MutableMapping

import numpy as np


@dataclass(slots=True)
class StateVector:
    """Minimal snapshot of the market state needed by path engines."""

    symbol: str
    asof: datetime
    spot: float
    drift: float
    vol: float
    risk_free_rate: float = 0.0
    jump_intensity: float = 0.0
    jump_mean: float = 0.0
    jump_std: float = 0.0
    iv_by_expiry: Mapping[datetime, float] = field(default_factory=dict)
    rng: np.random.Generator | None = None


@dataclass(slots=True)
class Artifact:
    """Container for Monte Carlo simulation outputs."""

    times: np.ndarray
    paths: np.ndarray
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def with_metadata(self, **metadata: Any) -> "Artifact":
        self.metadata.update(metadata)
        return self
