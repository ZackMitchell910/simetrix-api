"""Fundamental data adapter."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping

from src.adapters.base import FeedFrame, FeedRecord
from src.services.feature_store import FeatureStore



async def fetch(symbol: str, asof: datetime, window: timedelta | int | None = None) -> AdapterFrame:
    window_td = coerce_window(window, default_days=90)
    cache_key = f"{symbol.upper()}:{int(window_td.total_seconds())}"

    async def _loader() -> Mapping[str, object]:
        return {
            "source": "fundamentals",
            "data": [],
            "confidence": 0.0,
            "metadata": {"window_seconds": int(window_td.total_seconds())},
        }

    payload = await FEATURE_STORE.get_or_load(
        "fundamentals",
        cache_key,
        _loader,
        ttl=_TTL_SECONDS,
        diagnostics={"symbol": symbol.upper(), "window_seconds": int(window_td.total_seconds())},
    )

    return AdapterFrame(
        symbol=symbol,
        asof=asof,
        source=str(payload.get("source", "fundamentals")),
        confidence=float(payload.get("confidence", 0.0)),
        payload=payload.get("data") or [],
        metadata={**(payload.get("metadata") or {}), "window_seconds": int(window_td.total_seconds())},
    )
