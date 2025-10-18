from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from . import AdapterFrame, AdapterRecord


async def fetch(symbol: str, asof: datetime | None = None, window: int = 2) -> AdapterFrame:
    """Fetch recent news sentiment for the symbol."""

    now = asof or datetime.now(timezone.utc)
    payload: Mapping[str, Any] = {
        "window_days": int(max(1, window)),
        "asof": now.isoformat(),
        "articles": [],
    }
    record = AdapterRecord(
        symbol=symbol.upper(),
        asof=now,
        source="news",
        confidence=0.0,
        payload=dict(payload),
    )
    return [record]


__all__ = ["fetch"]
