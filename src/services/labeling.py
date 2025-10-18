from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

import httpx

from src.core import settings
from src.db.duck import matured_predictions_now
from src.labeler import label_mature_predictions
from src.observability import log_json

logger = logging.getLogger("simetrix.services.labeling")


def _polygon_key() -> str:
    env_key = (os.getenv("PT_POLYGON_KEY") or os.getenv("POLYGON_KEY") or "").strip()
    if env_key:
        return env_key
    try:
        return (settings.polygon_key or "").strip()
    except Exception:
        return ""


def to_polygon_ticker(raw: str) -> str:
    s = (raw or "").strip().upper()
    if not s:
        return ""
    if s.endswith("-USD") and not s.startswith("X:"):
        base = s[:-4]
        return f"X:{base}USD"
    if s.startswith("X:"):
        return s
    return s


async def _polygon_close_for_day(
    client: httpx.AsyncClient,
    symbol: str,
    day_iso: str,
    api_key: str,
) -> Optional[float]:
    params = {"apiKey": api_key}
    url = f"https://api.polygon.io/v1/open-close/{symbol}/{day_iso}"
    resp = await client.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        return None
    close = data.get("close")
    try:
        if isinstance(close, (int, float)):
            return float(close)
    except Exception:
        return None
    return None


async def _fetch_realized_price(
    symbol: str,
    target_ts: datetime,
    *,
    max_back_days: int = 4,
) -> Optional[float]:
    key = _polygon_key()
    if not key:
        logger.debug("Polygon key missing; realized price fetch skipped for %s", symbol)
        return None

    poly_symbol = to_polygon_ticker(symbol)
    if not poly_symbol:
        return None

    dt_utc = target_ts.astimezone(timezone.utc)
    base_day = dt_utc.date()

    timeout = httpx.Timeout(10.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            for offset in range(max_back_days + 1):
                day_iso = (base_day - timedelta(days=offset)).isoformat()
                try:
                    price = await _polygon_close_for_day(client, poly_symbol, day_iso, key)
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code if exc.response else "?"
                    if status == 429:
                        logger.warning("Polygon rate limited fetching close for %s %s", poly_symbol, day_iso)
                        break
                    logger.warning(
                        "Polygon HTTP error %s for %s %s: %s",
                        status,
                        poly_symbol,
                        day_iso,
                        exc,
                    )
                    continue
                except Exception as exc:
                    logger.warning("Realized price lookup failed for %s %s: %s", poly_symbol, day_iso, exc)
                    continue
                if price is not None:
                    return float(price)
    except Exception as exc:
        logger.warning("Realized price fetch failed for %s: %s", symbol, exc)
    return None


def labeler_pass(limit: int) -> dict[str, int]:
    """
    Synchronous labeling pass used by scripts and background jobs.
    """
    if label_mature_predictions is None:
        return {"processed": 0, "labeled": 0}

    rows = matured_predictions_now(limit=limit)
    processed = len(rows)
    if processed == 0:
        return {"processed": 0, "labeled": 0}

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)

        def _fetch(symbol: str, target: datetime) -> Optional[float]:
            try:
                return loop.run_until_complete(_fetch_realized_price(symbol, target))
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("labeler fetch failed for %s: %s", symbol, exc)
                return None

        labeled = label_mature_predictions(_fetch, limit=limit, rows=rows)
        return {"processed": processed, "labeled": int(labeled)}
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        asyncio.set_event_loop(None)
        loop.close()


async def run_labeling_pass(limit: int) -> dict[str, int]:
    """Async wrapper for labeler_pass."""
    return await asyncio.to_thread(labeler_pass, limit)


async def label_outcomes(limit: int) -> dict[str, Any]:
    """
    Public entry point used by admin/API cron to label matured predictions.
    """
    if label_mature_predictions is None:
        raise RuntimeError("labeler_unavailable")

    clamp = max(100, min(int(limit), int(settings.labeling_batch_limit)))
    stats = await run_labeling_pass(clamp)
    processed = int(stats.get("processed", 0))
    labeled = int(stats.get("labeled", 0))
    log_json("info", msg="label_outcomes", processed=processed, labeled=labeled, limit=clamp)
    return {"status": "ok", "processed": processed, "labeled": labeled, "limit": clamp}


__all__ = ["labeler_pass", "run_labeling_pass", "label_outcomes", "to_polygon_ticker"]
