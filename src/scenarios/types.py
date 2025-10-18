"""Data structures used across scenario generation and simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass(slots=True)
class Headline:
    """Simple representation of a news headline."""

    published: datetime
    text: str
    source: str | None = None

    def key_terms(self) -> set[str]:
        """Return a set of lowercase tokens for quick keyword scans."""

        return {token.lower() for token in self.text.split()}


@dataclass(slots=True)
class CalendarItem:
    """Entry describing a dated corporate event (e.g. earnings)."""

    symbol: str
    event_date: datetime
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EventShock:
    """Scenario variant applied to price paths within a time window."""

    event_id: str
    variant: str
    window_start: datetime
    window_end: datetime
    prior: float
    drift_bump: float = 0.0
    vol_multiplier: float = 1.0
    jump_intensity: float = 0.0
    jump_mean: float = 0.0
    jump_std: float = 0.0
    description: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def copy_with(self, **kwargs: Any) -> "EventShock":
        """Return a shallow copy with a subset of fields replaced."""

        data = self.__dict__.copy()
        data.update(kwargs)
        return EventShock(**data)


@dataclass(slots=True)
class HistoricalOutcome:
    """Historical realization of a scenario prediction."""

    timestamp: datetime
    event_id: str
    predicted_variant: str
    actual_variant: str
    weight: float = 1.0

    @property
    def hit(self) -> bool:
        return self.predicted_variant == self.actual_variant
