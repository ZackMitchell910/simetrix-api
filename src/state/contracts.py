from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List

from src.scenarios.schema import EventShock


@dataclass(slots=True)
class StateVector:
    """Canonical representation of the simulation state at a given time."""

    t: datetime
    spot: float
    drift_annual: float
    vol_annual: float
    jump_intensity: float
    jump_mean: float
    jump_vol: float
    regime: str
    macro: Dict[str, Any] = field(default_factory=dict)
    sentiment: Dict[str, Any] = field(default_factory=dict)
    events: List[EventShock] = field(default_factory=list)
    cross: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        events_payload = [event.as_dict() if hasattr(event, "as_dict") else asdict(event) for event in self.events]
        return {
            "t": self.t.isoformat(),
            "spot": float(self.spot),
            "drift_annual": float(self.drift_annual),
            "vol_annual": float(self.vol_annual),
            "jump_intensity": float(self.jump_intensity),
            "jump_mean": float(self.jump_mean),
            "jump_vol": float(self.jump_vol),
            "regime": self.regime,
            "macro": dict(self.macro),
            "sentiment": dict(self.sentiment),
            "events": events_payload,
            "cross": dict(self.cross),
            "provenance": dict(self.provenance),
        }


__all__ = ["StateVector"]
