"""Contracts for the time-varying simulation state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping

from src.scenarios.schema import EventShock


@dataclass(slots=True)
class StateVector:
    """Structured state used by the simulation and diagnostics layers."""

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
    events: list[EventShock] = field(default_factory=list)
    cross: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize into a JSON-friendly dictionary."""

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
            "events": [event.to_dict() for event in self.events],
            "cross": dict(self.cross),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StateVector":
        """Hydrate from a dictionary payload."""

        dt_value = payload.get("t")
        if isinstance(dt_value, str):
            t = datetime.fromisoformat(dt_value)
        elif isinstance(dt_value, datetime):
            t = dt_value
        else:  # pragma: no cover - fallback
            t = datetime.utcnow()

        events_payload: Iterable[Mapping[str, Any]] = payload.get("events", [])  # type: ignore[assignment]
        events = [EventShock.from_dict(event) for event in events_payload]

        return cls(
            t=t,
            spot=float(payload.get("spot", 0.0)),
            drift_annual=float(payload.get("drift_annual", 0.0)),
            vol_annual=float(payload.get("vol_annual", 0.0)),
            jump_intensity=float(payload.get("jump_intensity", 0.0)),
            jump_mean=float(payload.get("jump_mean", 0.0)),
            jump_vol=float(payload.get("jump_vol", 0.0)),
            regime=str(payload.get("regime", "neutral")),
            macro=dict(payload.get("macro", {})),
            sentiment=dict(payload.get("sentiment", {})),
            events=events,
            cross=dict(payload.get("cross", {})),
            provenance=dict(payload.get("provenance", {})),
        )
