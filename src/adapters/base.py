from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence

__all__ = ["AdapterFrame", "coerce_window"]


@dataclass
class AdapterFrame:
    """Typed response returned by feed adapters.

    ``payload`` is intentionally free-form: adapters may return a sequence of
    records or a mapping containing pre-aggregated signals.  Downstream callers
    should rely on ``confidence`` to determine how aggressively to use the
    frame.
    """

    symbol: str
    asof: datetime
    source: str
    confidence: float
    payload: Sequence[Mapping[str, Any]] | Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.upper()
        self.source = str(self.source or "").strip() or "unknown"
        self.confidence = float(max(0.0, min(1.0, self.confidence)))
        if not isinstance(self.payload, Mapping) and not isinstance(self.payload, Sequence):
            raise TypeError("payload must be a mapping or sequence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "asof": self.asof.isoformat(),
            "source": self.source,
            "confidence": self.confidence,
            "payload": self.payload,
            "metadata": dict(self.metadata),
        }


def coerce_window(window: Any, *, default_days: int = 3) -> timedelta:
    if isinstance(window, timedelta):
        return window
    if isinstance(window, (int, float)):
        return timedelta(days=float(window))
    if window is None:
        return timedelta(days=default_days)
    raise TypeError(f"Unsupported window type: {type(window)!r}")
