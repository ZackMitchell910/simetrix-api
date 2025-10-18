from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - only for static type checking
    from src.scenarios.schema import EventShock


@dataclass
class StateVector:
    """Structured snapshot of latent market state driving simulations."""

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
    events: List["EventShock"] = field(default_factory=list)
    cross: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the state vector into JSON-friendly primitives."""

        payload = {
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
            "cross": dict(self.cross),
            "provenance": dict(self.provenance),
        }
        events_serialized: list[Any] = []
        for event in self.events:
            if hasattr(event, "to_dict"):
                events_serialized.append(event.to_dict())
            else:  # pragma: no cover - graceful fallback for simple mappings
                try:
                    events_serialized.append(asdict(event))
                except Exception:
                    events_serialized.append(dict(event))  # type: ignore[arg-type]
        payload["events"] = events_serialized
        return payload


__all__ = ["StateVector"]
