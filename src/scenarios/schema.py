"""Scenario dataclasses shared with the LLM scenario generator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Mapping


@dataclass(slots=True)
class EventShock:
    """Structured override representing a discrete scenario."""

    id: str
    title: str
    window_start: datetime
    window_end: datetime
    p: float
    overrides: Dict[str, Any]
    trigger: str
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "p": float(self.p),
            "overrides": dict(self.overrides),
            "trigger": self.trigger,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EventShock":
        def _to_dt(value: Any) -> datetime:
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                return datetime.fromisoformat(value)
            raise TypeError(f"Unsupported datetime payload: {value!r}")

        return cls(
            id=str(payload.get("id", "")),
            title=str(payload.get("title", "")),
            window_start=_to_dt(payload.get("window_start", datetime.utcnow())),
            window_end=_to_dt(payload.get("window_end", datetime.utcnow())),
            p=float(payload.get("p", 0.0)),
            overrides=dict(payload.get("overrides", {})),
            trigger=str(payload.get("trigger", "unknown")),
            rationale=str(payload.get("rationale", "")),
        )
