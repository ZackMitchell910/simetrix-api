from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from . import AdapterFrame, AdapterRecord


async def fetch(symbol: str, asof: datetime | None = None, window: int = 90) -> AdapterFrame:
    """Fetch core fundamental metrics."""

    now = asof or datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "window_days": int(max(1, window)),
        "metrics": {
            "pe": None,
            "ps": None,
            "earnings_yield": None,
        },
    }
    record = AdapterRecord(
        symbol=symbol.upper(),
        asof=now,
        source="fundamentals",
        confidence=0.0,
        payload=payload,
    )
    return [record]


__all__ = ["fetch"]
