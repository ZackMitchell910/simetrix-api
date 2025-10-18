from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class EventShock:
    """Represents a discrete scenario shock applied during simulation.

    Attributes
    ----------
    shock_id:
        Stable identifier for the scenario (e.g. ``"earnings-beat"``).
    label:
        Human readable description for diagnostics and UI surfaces.
    prior:
        Prior probability weight assigned to the scenario. The generator and
        scenario book guarantee that mutually exclusive variants sum to less
        than or equal to one.
    window_start:
        Start of the event window expressed in trading days offset from the
        pricing ``as_of`` date used in the simulation.
    window_end:
        End of the event window expressed in trading days offset from the
        pricing ``as_of`` date.
    drift:
        Annualised drift override added to the baseline process while the shock
        is active.
    vol_multiplier:
        Multiplicative factor applied to the baseline volatility during the
        event window.
    jump_intensity:
        Annualised Poisson intensity governing the arrival of jumps.
    jump_mean:
        Average log jump size applied when a jump arrives.
    jump_std:
        Standard deviation of the log jump size distribution.
    mutually_exclusive_group:
        Identifier used to ensure prior weights are normalised across
        alternatives that cannot co-occur.
    metadata:
        Auxiliary diagnostics and rationale for analytics consumers.
    """

    shock_id: str
    label: str
    prior: float
    window_start: float
    window_end: float
    drift: float = 0.0
    vol_multiplier: float = 1.0
    jump_intensity: float = 0.0
    jump_mean: float = 0.0
    jump_std: float = 0.0
    mutually_exclusive_group: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_active(self, offset_days: float) -> bool:
        """Return ``True`` if the supplied offset lies within the event window."""

        return self.window_start <= offset_days <= self.window_end

    def copy_with_prior(self, prior: float) -> "EventShock":
        """Clone the shock with an updated prior while preserving metadata."""

        return EventShock(
            shock_id=self.shock_id,
            label=self.label,
            prior=prior,
            window_start=self.window_start,
            window_end=self.window_end,
            drift=self.drift,
            vol_multiplier=self.vol_multiplier,
            jump_intensity=self.jump_intensity,
            jump_mean=self.jump_mean,
            jump_std=self.jump_std,
            mutually_exclusive_group=self.mutually_exclusive_group,
            metadata=dict(self.metadata),
        )


@dataclass
class ScenarioDiagnostics:
    """Container for quality metrics used to assess scenario calibration."""

    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    sample_size: int = 0

    def add(self, predicted: str, actual: str) -> None:
        group = self.confusion_matrix.setdefault(predicted, {})
        group[actual] = group.get(actual, 0) + 1
        self.sample_size += 1

    def as_dict(self) -> Dict[str, Any]:
        return {"confusion_matrix": self.confusion_matrix, "sample_size": self.sample_size}
