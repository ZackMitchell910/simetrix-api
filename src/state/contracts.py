from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence

from src.scenarios.schema import EventShock

__all__ = ["StateVector"]


@dataclass(slots=True)
class StateVector:
    t: datetime
    spot: float
    drift_annual: float
    vol_annual: float
    jump_intensity: float
    jump_mean: float
    jump_vol: float
    regime: str
    macro: Mapping[str, Any] = field(default_factory=dict)
    sentiment: Mapping[str, Any] = field(default_factory=dict)
    events: Sequence[EventShock] = field(default_factory=list)
    cross: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_context(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly diagnostics structure."""

        payload = asdict(self)
        payload["t"] = self.t.isoformat()
        payload["events"] = [event.to_dict() for event in self.events]
        return payload
