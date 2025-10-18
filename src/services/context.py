from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict

import numpy as np
from fastapi import HTTPException

from src.core import REDIS
from src.services import ingestion as ingestion_service

try:
    from src.feature_store import connect as feature_store_connect
except Exception:  # pragma: no cover - optional when FS unavailable
    feature_store_connect = None  # type: ignore

logger = logging.getLogger("simetrix.services.context")



def _as_utc_datetime(value: datetime | date | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)
    return None


async def get_sentiment_features(symbol: str, days: int = 7) -> dict[str, Any]:
    base = {"avg_sent_7d": 0.0, "last24h": 0.0, "n_news": 0}
    fs_connect = feature_store_connect
    if not callable(fs_connect):
        return base
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return base
    window_days = int(max(1, days))
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    try:
        con = fs_connect()
    except Exception as exc:
        logger.debug("get_sentiment_features: fs_connect failed: %s", exc)
        return base
    try:
        rows = con.execute(
            """
            SELECT ts, sentiment
            FROM news_articles
            WHERE symbol = ? AND ts >= ?
            ORDER BY ts DESC
            """,
            [symbol, cutoff],
        ).fetchall()
    except Exception as exc:
        logger.debug("get_sentiment_features: query failed for %s: %s", symbol, exc)
        rows = []
    finally:
        try:
            con.close()
        except Exception:
            pass
    if not rows:
        return base

    now = datetime.now(timezone.utc)
    vals = []
    last24_vals = []
    for ts, sentiment in rows:
        val = float(sentiment or 0.0)
        vals.append(val)
        ts_dt = _as_utc_datetime(ts)
        if ts_dt is None:
            continue
        if (now - ts_dt).total_seconds() <= 86400:
            last24_vals.append(val)

    base["avg_sent_7d"] = float(np.mean(vals)) if vals else 0.0
    base["last24h"] = float(np.mean(last24_vals)) if last24_vals else 0.0
    base["n_news"] = len(vals)
    return base


async def get_earnings_features(symbol: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "surprise_last": 0.0,
        "guidance_delta": 0.0,
        "days_since_earn": None,
        "days_to_next": None,
    }
    fs_connect = feature_store_connect
    if not callable(fs_connect):
        return out
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return out
    try:
        con = fs_connect()
    except Exception as exc:
        logger.debug("get_earnings_features: fs_connect failed: %s", exc)
        return out
    try:
        last = con.execute(
            """
            SELECT report_date, surprise, guidance_delta
            FROM earnings
            WHERE symbol = ?
            ORDER BY report_date DESC
            LIMIT 1
            """,
            [symbol],
        ).fetchone()
    except Exception as exc:
        logger.debug("get_earnings_features: query failed for %s: %s", symbol, exc)
        last = None
    finally:
        try:
            con.close()
        except Exception:
            pass

    if last:
        report_date, surprise, guidance_delta = last
        if surprise is not None:
            out["surprise_last"] = float(surprise)
        if guidance_delta is not None:
            out["guidance_delta"] = float(guidance_delta)
        rd = report_date
        if isinstance(rd, datetime):
            rd = rd.date()
        if isinstance(rd, date):
            out["days_since_earn"] = (datetime.now(timezone.utc).date() - rd).days
    return out


async def get_macro_features() -> dict[str, Any]:
    base = {"rff": None, "cpi_yoy": None, "u_rate": None}
    fs_connect = feature_store_connect
    if not callable(fs_connect):
        return base
    try:
        con = fs_connect()
    except Exception as exc:
        logger.debug("get_macro_features: fs_connect failed: %s", exc)
        return base
    try:
        row = con.execute(
            """
            SELECT rff, cpi_yoy, u_rate
            FROM macro_daily
            ORDER BY as_of DESC
            LIMIT 1
            """
        ).fetchone()
    except Exception as exc:
        logger.debug("get_macro_features: query failed: %s", exc)
        row = None
    finally:
        try:
            con.close()
        except Exception:
            pass
    if not row:
        return base
    return {
        "rff": float(row[0]) if row[0] is not None else None,
        "cpi_yoy": float(row[1]) if row[1] is not None else None,
        "u_rate": float(row[2]) if row[2] is not None else None,
    }


def detect_regime(px: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(px, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size < 40:
        return {"name": "neutral", "score": 0.0}

    rets = np.diff(np.log(arr))
    if rets.size == 0:
        return {"name": "neutral", "score": 0.0}

    rv20 = float(np.std(rets[-20:]) * math.sqrt(252)) if rets.size >= 20 else float(np.std(rets) * math.sqrt(252))
    mom20 = float((arr[-1] / arr[max(0, arr.size - 21)]) - 1.0)

    v = float(np.clip((rv20 - 0.30) / 0.30, -1.5, 1.5))
    m = float(np.clip(mom20 / 0.10, -1.5, 1.5))

    if v > 0.6 and m < -0.3:
        name, score = "vol-shock", -0.7
    elif m > 0.4 and v < 0.3:
        name, score = "bull-trend", 0.6
    elif m < -0.4 and v < 0.3:
        name, score = "bear-trend", -0.5
    else:
        name, score = "neutral", float(np.clip(m - 0.3 * v, -0.4, 0.4))
    return {"name": name, "score": score}


async def build_context(symbol: str) -> dict[str, Any]:
    sym = (symbol or "").strip().upper()
    if not sym:
        raise HTTPException(status_code=400, detail="Symbol required")

    redis = REDIS
    try:
        px = await ingestion_service.fetch_cached_hist_prices(sym, 220, redis)
        regime = detect_regime(np.asarray(px, dtype=float))
    except Exception as exc:
        logger.debug("context_regime_failed for %s: %s", sym, exc)
        regime = {"name": "neutral", "score": 0.0}

    sentiment = await get_sentiment_features(sym)
    earnings = await get_earnings_features(sym)
    macro = await get_macro_features()

    return {
        "symbol": sym,
        "regime": regime,
        "sentiment": sentiment,
        "earnings": earnings,
        "macro": macro,
    }
