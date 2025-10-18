"""Data structures representing scenario shocks."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Dict, Mapping, Optional


@dataclass
class EventShock:
    """Represents a discrete scenario applied during the simulation horizon.

    The fields are intentionally generic so that both scenario generation and
    pricing engines can attach additional metadata without breaking downstream
    consumers.  The `mutually_exclusive_group` flag allows scenario generators to
    emit several variants (for example, *beat*, *inline*, *miss*) that should be
    normalised so that their priors sum to at most one.
    """

    symbol: str
    label: str
    window_start: datetime
    window_end: datetime
    prior: float
    variant: str = "base"
    description: str | None = None
    drift: float = 0.0
    volatility_scale: float = 1.0
    jump_intensity: float = 0.0
    jump_mean: float = 0.0
    jump_std: float = 0.0
    mutually_exclusive_group: str | None = None
    evidence: list[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def copy(self, **updates: Any) -> "EventShock":
        """Return a copy with updated fields."""

        return replace(self, **updates)

    def with_prior(self, prior: float) -> "EventShock":
        """Return a copy with an updated prior while keeping other fields."""

        return self.copy(prior=prior)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the shock to a dictionary for logging/diagnostics."""

        payload: Dict[str, Any] = {
            "symbol": self.symbol,
            "label": self.label,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "prior": self.prior,
            "variant": self.variant,
            "description": self.description,
            "drift": self.drift,
            "volatility_scale": self.volatility_scale,
            "jump_intensity": self.jump_intensity,
            "jump_mean": self.jump_mean,
            "jump_std": self.jump_std,
            "mutually_exclusive_group": self.mutually_exclusive_group,
            "evidence": list(self.evidence),
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


def normalise_priors(shocks: Mapping[str | None, list[EventShock]]) -> list[EventShock]:
    """Normalise priors within each mutually exclusive group.

    Parameters
    ----------
    shocks:
        Mapping from mutually exclusive group name to the list of shocks that
        belong to the group.  The ``None`` key represents shocks with no
        exclusivity constraints.
    """

    normalised: list[EventShock] = []
    for group, variants in shocks.items():
        if not variants:
            continue
        total = sum(max(variant.prior, 0.0) for variant in variants)
        if group is None or total <= 1.0 or total == 0.0:
            normalised.extend(variant if variant.prior >= 0 else variant.with_prior(0.0) for variant in variants)
            continue
        scale = min(1.0 / total, 1.0)
        for variant in variants:
            clipped = max(variant.prior, 0.0) * scale
            normalised.append(variant.with_prior(clipped))
    return normalised
