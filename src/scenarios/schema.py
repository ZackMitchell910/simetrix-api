from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass
class EventShock:
    """Structured shock injected by the scenario generator."""

    id: str
    title: str
    window_start: datetime
    window_end: datetime
    p: float
    overrides: Dict[str, Any] = field(default_factory=dict)
    trigger: str = "macro"
    rationale: str = ""

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


__all__ = ["EventShock"]
