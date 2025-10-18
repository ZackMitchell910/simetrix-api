from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ShockOverride:
    """Quantitative overrides applied by an :class:`EventShock`.

    Attributes
    ----------
    drift_bump : float
        Annualised drift adjustment applied additively.
    vol_multiplier : float
        Multiplicative factor applied to the instantaneous volatility.
    jump_intensity : float
        Additional jump arrival intensity (lambda) expressed in annualised units.
    jump_mean : float
        Mean jump size in log space.
    jump_std : float
        Standard deviation of the jump size in log space.
    """

    drift_bump: float = 0.0
    vol_multiplier: float = 1.0
    jump_intensity: float = 0.0
    jump_mean: float = 0.0
    jump_std: float = 0.0

    def merge(self, other: "ShockOverride") -> "ShockOverride":
        """Combine overrides sequentially."""
        return ShockOverride(
            drift_bump=self.drift_bump + other.drift_bump,
            vol_multiplier=self.vol_multiplier * other.vol_multiplier,
            jump_intensity=self.jump_intensity + other.jump_intensity,
            jump_mean=self.jump_mean + other.jump_mean,
            jump_std=max(self.jump_std, other.jump_std),
        )


@dataclass(frozen=True, slots=True)
class EventShock:
    """Structured representation of a hypothetical event outcome."""

    symbol: str
    name: str
    variant: str
    prior: float
    window_start: datetime
    window_end: datetime
    override: ShockOverride = field(default_factory=ShockOverride)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def with_prior(self, prior: float) -> "EventShock":
        return replace(self, prior=max(0.0, float(prior)))

    def with_metadata(self, **metadata: Any) -> "EventShock":
        merged = dict(self.metadata)
        merged.update(metadata)
        return replace(self, metadata=merged)
