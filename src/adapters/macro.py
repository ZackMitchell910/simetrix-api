from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from . import AdapterFrame, AdapterRecord


async def fetch(symbol: str, asof: datetime | None = None, window: int = 30) -> AdapterFrame:
    """Fetch macro signals relevant for the symbol universe."""

    now = asof or datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "window_days": int(max(1, window)),
        "macro": {
            "rff": None,
            "cpi_yoy": None,
            "u_rate": None,
        },
    }
    record = AdapterRecord(
        symbol=symbol.upper(),
        asof=now,
        source="macro",
        confidence=0.0,
        payload=payload,
    )
    return [record]


__all__ = ["fetch"]
