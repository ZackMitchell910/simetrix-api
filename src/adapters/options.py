"""Options surface adapter."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping

from src.adapters.base import FeedFrame, FeedRecord
from src.services.feature_store import FeatureStore


def _normalize_rows(rows: Iterable[Mapping[str, Any]]) -> FeedFrame:
    parsed = []
    for row in rows:
        asof_raw = row.get("asof")
        if isinstance(asof_raw, str):
            asof = datetime.fromisoformat(asof_raw)
        elif isinstance(asof_raw, datetime):
            asof = asof_raw
        else:  # pragma: no cover
            continue
        parsed.append(
            FeedRecord(
                symbol=str(row.get("symbol", "")),
                asof=asof,
                source=str(row.get("source", "options")),
                confidence=float(row.get("confidence", 0.0)),
                payload=dict(row.get("payload", {})),
                tags=tuple(row.get("tags", ())),
            )
        )
    return FeedFrame(parsed)


async def fetch(
    symbol: str,
    asof: datetime,
    window: int,
    *,
    store: FeatureStore | None = None,
) -> FeedFrame:
    """Fetch aggregated options greeks/IV time series."""

    symbol_upper = symbol.upper().strip()
    rows: list[Mapping[str, Any]] = []
    if store is not None:
        key = f"options:{symbol_upper}:{asof.date().isoformat()}:{window}"
        cached, _ = await store.get(key)
        if isinstance(cached, list):
            rows.extend(cached)
    if not rows:
        rows.append(
            {
                "symbol": symbol_upper,
                "asof": (asof - timedelta(hours=2)).isoformat(),
                "source": "options",
                "confidence": 0.05,
                "payload": {"implied_vol": 0.4, "skew": 0.0},
                "tags": ("synthetic",),
            }
        )
    return _normalize_rows(rows)


__all__ = ["fetch"]
