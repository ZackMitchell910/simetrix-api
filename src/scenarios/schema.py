from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

__all__ = ["EventShock"]


@dataclass(slots=True)
class EventShock:
    id: str
    title: str
    window_start: datetime
    window_end: datetime
    p: float
    overrides: Mapping[str, Any] = field(default_factory=dict)
    trigger: str = ""
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
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
