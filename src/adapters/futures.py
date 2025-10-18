from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from . import AdapterFrame, AdapterRecord


async def fetch(symbol: str, asof: datetime | None = None, window: int = 5) -> AdapterFrame:
    """Fetch futures term structure to inform drift priors."""

    now = asof or datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "window_days": int(max(1, window)),
        "curve": [],
        "basis": None,
    }
    record = AdapterRecord(
        symbol=symbol.upper(),
        asof=now,
        source="futures",
        confidence=0.0,
        payload=payload,
    )
    return [record]


__all__ = ["fetch"]
