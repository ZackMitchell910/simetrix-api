from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict


@dataclass(slots=True)
class AdapterRow:
    symbol: str
    asof: datetime
    source: str
    confidence: float
    payload: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        data = {
            "symbol": self.symbol,
            "asof": self.asof.isoformat(),
            "source": self.source,
            "confidence": float(self.confidence),
        }
        data.update(self.payload)
        return data


def normalize_asof(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


__all__ = ["AdapterRow", "normalize_asof"]
