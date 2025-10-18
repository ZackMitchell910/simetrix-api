"""State vector describing the underlying asset dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class StateVector:
    """Minimal state required by the path engines."""

    spot: float
    drift: float
    vol: float
    as_of: datetime
    jump_intensity: float = 0.0
    jump_mean: float = 0.0
    jump_std: float = 0.0
    corr: Any | None = None
    iv_curve: dict[int, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
