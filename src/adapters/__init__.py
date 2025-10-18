from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, List


@dataclass(frozen=True)
class AdapterRecord:
    symbol: str
    asof: datetime
    source: str
    confidence: float
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "asof": self.asof.isoformat(),
            "source": self.source,
            "confidence": float(self.confidence),
            "payload": dict(self.payload),
        }


AdapterFrame = List[AdapterRecord]


def empty_frame(symbol: str, source: str) -> AdapterFrame:
    return [
        AdapterRecord(
            symbol=symbol.upper(),
            asof=datetime.now(timezone.utc),
            source=source,
            confidence=0.0,
            payload={"status": "empty"},
        )
    ]


def ensure_frame(records: Iterable[AdapterRecord | dict[str, Any]]) -> AdapterFrame:
    frame: AdapterFrame = []
    for record in records:
        if isinstance(record, AdapterRecord):
            frame.append(record)
        else:
            frame.append(AdapterRecord(**record))
    return frame


__all__ = ["AdapterRecord", "AdapterFrame", "empty_frame", "ensure_frame"]
