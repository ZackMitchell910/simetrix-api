from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from . import AdapterFrame, AdapterRecord


async def fetch(symbol: str, asof: datetime | None = None, window: int = 1) -> AdapterFrame:
    """Fetch social sentiment features."""

    now = asof or datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "window_days": int(max(1, window)),
        "signals": {
            "buzz": None,
            "polarity": None,
        },
    }
    record = AdapterRecord(
        symbol=symbol.upper(),
        asof=now,
        source="social",
        confidence=0.0,
        payload=payload,
    )
    return [record]


__all__ = ["fetch"]
