from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Optional, Sequence

import httpx
import numpy as np
from fastapi import HTTPException
from redis.asyncio import Redis

from src.core import REDIS, settings
from src.observability import log_json
from src.services.llm import summarize as llm_summarize

try:
    from src.feature_store import (
        connect as feature_store_connect,
        insert_earnings as feature_store_insert_earnings,
        insert_news as feature_store_insert_news,
        log_ingest_event as feature_store_log_ingest_event,
        upsert_macro as feature_store_upsert_macro,
    )
except Exception:  # pragma: no cover - feature store unavailable in some test contexts
    feature_store_connect = None
    feature_store_insert_earnings = None
    feature_store_insert_news = None
    feature_store_log_ingest_event = None
    feature_store_upsert_macro = None

try:  # optional dependency
    from prometheus_client import Counter
except Exception:  # pragma: no cover - prometheus optional
    Counter = None  # type: ignore

if Counter is not None:
    NEWS_INGESTED_COUNTER = Counter(
        "news_ingested_total",
        "Count of news rows ingested into feature store",
    )
    NEWS_SCORED_COUNTER = Counter(
        "news_scored_total",
        "Count of news rows sentiment-scored",
    )
    EARNINGS_INGESTED_COUNTER = Counter(
        "earnings_ingested_total",
        "Count of earnings rows ingested",
    )
    MACRO_UPSERTS_COUNTER = Counter(
        "macro_upserts_total",
        "Count of macro snapshot upserts",
    )
else:  # pragma: no cover - prometheus optional
    NEWS_INGESTED_COUNTER = None  # type: ignore
    NEWS_SCORED_COUNTER = None  # type: ignore
    EARNINGS_INGESTED_COUNTER = None  # type: ignore
    MACRO_UPSERTS_COUNTER = None  # type: ignore


logger = logging.getLogger("simetrix.services.ingestion")


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


def _first_number(values: Iterable[Any]) -> float | None:
    for value in values:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            val = value.strip()
            if not val or val == ".":
                continue
            try:
                return float(val)
            except Exception:
                continue
    return None


def _determine_news_provider(explicit: Optional[str]) -> str:
    if explicit:
        provider = explicit.strip().lower()
        if provider:
            return provider
    env_provider = (os.getenv("PT_NEWS_PROVIDER") or "").strip().lower()
    if env_provider:
        return env_provider
    if settings.news_api_key:
        return "newsapi"
    return "polygon"


def _determine_earnings_provider(explicit: Optional[str]) -> str:
    if explicit:
        provider = explicit.strip().lower()
        if provider:
            return provider
    provider = (settings.earnings_source or os.getenv("PT_EARNINGS_PROVIDER", "polygon")).strip().lower()
    return provider or "polygon"


def _determine_macro_provider(explicit: Optional[str]) -> str:
    if explicit:
        provider = explicit.strip().lower()
        if provider:
            return provider
    src = (settings.macro_source or os.getenv("PT_MACRO_PROVIDER", "fred")).strip().lower()
    return src or "fred"


def _polygon_api_key() -> str:
    env_candidates = [
        os.getenv("PT_POLYGON_KEY"),
        os.getenv("POLYGON_KEY"),
    ]
    for candidate in env_candidates:
        if candidate:
            key = candidate.strip()
            if key:
                return key
    try:
        cfg_key = getattr(settings, "polygon_key", "") or ""
        if cfg_key:
            key = str(cfg_key).strip()
            if key:
                return key
    except Exception:
        pass
    return ""


def _poly_crypto_to_app(symbol: str) -> str:
    sym = (symbol or "").strip().upper()
    if sym.startswith("X:") and sym.endswith("USD"):
        return f"{sym[2:-3]}-USD"
    return sym


def _to_app_symbol(symbol: str) -> str:
    sym = (symbol or "").strip().upper()
    if sym.startswith("X:") and sym.endswith("USD"):
        return f"{sym[2:-3]}-USD"
    if sym.endswith("USD") and "-" not in sym and not sym.startswith("X:"):
        return f"{sym[:-3]}-USD"
    return sym


def _to_polygon_symbol(symbol: str) -> str:
    sym = (symbol or "").strip().upper()
    if not sym:
        return sym
    if sym.startswith("X:"):
        return sym
    if sym.endswith("-USD"):
        return f"X:{sym.replace('-USD', '')}USD"
    if sym.endswith("USD") and "-" not in sym:
        return f"X:{sym}"
    return sym


def _news_ts(value: Any, default: datetime) -> datetime:
    if isinstance(value, datetime):
        return _as_utc_datetime(value) or default
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return default
        if txt.endswith("Z"):
            txt = txt[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(txt)
            return _as_utc_datetime(parsed) or default
        except Exception:
            pass
    return default


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return None
        try:
            return datetime.fromisoformat(txt.replace("Z", "+00:00")).date()
        except Exception:
            try:
                return datetime.strptime(txt.split("T")[0], "%Y-%m-%d").date()
            except Exception:
                return None
    return None


async def _news_rows_from_newsapi(symbol: str, since: datetime, limit: int, api_key: str) -> list[tuple]:
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": symbol,
        "from": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": int(max(1, min(limit, 100))),
    }
    headers = {"X-Api-Key": api_key}
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json() or {}
    items = data.get("articles", []) or []
    rows: list[tuple] = []
    seen: set[tuple] = set()
    for item in items:
        ts = _news_ts(item.get("publishedAt"), since)
        if ts < since:
            continue
        url_item = (item.get("url") or "").strip()
        dedupe_key = (ts.isoformat(), url_item)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        source_name = ""
        src_obj = item.get("source") or {}
        if isinstance(src_obj, dict):
            source_name = (src_obj.get("name") or "").strip()
        title = (item.get("title") or "").strip()
        summary = (item.get("description") or item.get("content") or "").strip()
        rows.append((symbol, ts, source_name or "NewsAPI", title, url_item, summary, None))
        if len(rows) >= limit:
            break
    return rows


async def _news_rows_from_polygon(symbol: str, since: datetime, limit: int, api_key: str) -> list[tuple]:
    url = "https://api.polygon.io/v2/reference/news"
    params = {
        "ticker": symbol,
        "limit": int(max(1, min(limit, 100))),
        "published_utc.gte": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sort": "published_utc",
        "order": "desc",
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json() or {}
    items = data.get("results", []) or []
    rows: list[tuple] = []
    for item in items:
        ts = _news_ts(item.get("published_utc"), since)
        if ts < since:
            continue
        publisher = ""
        pub_obj = item.get("publisher")
        if isinstance(pub_obj, dict):
            publisher = (pub_obj.get("name") or "").strip()
        title = (item.get("title") or "").strip()
        url_item = (item.get("article_url") or item.get("url") or "").strip()
        summary = (item.get("description") or item.get("excerpt") or "").strip()
        rows.append((symbol, ts, publisher or "Polygon", title, url_item, summary, None))
        if len(rows) >= limit:
            break
    return rows


async def _fetch_news_articles(symbol: str, since: datetime, provider: str, limit: int = 100) -> list[tuple]:
    provider = (provider or "polygon").strip().lower()
    limit = int(max(1, min(limit, 200)))
    if provider == "newsapi":
        api_key = settings.news_api_key or os.getenv("NEWS_API_KEY", "").strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="NewsAPI key missing (set PT_NEWS_API_KEY).")
        return await _news_rows_from_newsapi(symbol, since, limit=limit, api_key=api_key)
    if provider == "polygon":
        key = _polygon_api_key()
        if not key:
            raise HTTPException(status_code=400, detail="Polygon key missing for news ingest.")
        return await _news_rows_from_polygon(symbol, since, limit=limit, api_key=key)
    raise HTTPException(status_code=400, detail=f"Unsupported news provider '{provider}'.")


async def fetch_recent_news(symbol: str, days: int = 7, *, limit: int = 100) -> list[dict[str, Any]]:
    """Load recent news rows for a symbol from the feature store."""

    app_symbol = _to_app_symbol(symbol)
    if not app_symbol or not callable(feature_store_connect):
        return []

    horizon = max(1, int(days))
    since = datetime.now(timezone.utc) - timedelta(days=horizon)

    def _load() -> list[dict[str, Any]]:
        try:
            con = feature_store_connect()
        except Exception:
            return []

        try:
            rows = con.execute(
                """
                SELECT ts, source, title, url, summary, sentiment
                FROM news_articles
                WHERE symbol = ? AND ts >= ?
                ORDER BY ts DESC
                LIMIT ?
                """,
                [app_symbol, since, int(limit)],
            ).fetchall()
        except Exception:
            rows = []
        finally:
            try:
                con.close()
            except Exception:
                pass

        payload: list[dict[str, Any]] = []
        for ts, source, title, url, summary, sentiment in rows:
            ts_dt = _as_utc_datetime(ts) or since
            payload.append(
                {
                    "symbol": app_symbol,
                    "ts": ts_dt,
                    "source": (source or ""),
                    "title": (title or ""),
                    "url": (url or ""),
                    "summary": (summary or ""),
                    "sentiment": (float(sentiment) if sentiment is not None else None),
                }
            )
        return payload

    rows = await asyncio.to_thread(_load)
    rows.sort(key=lambda row: row.get("ts") or since, reverse=True)
    return rows


async def ingest_news(
    symbol: str,
    *,
    days: int,
    limit: int,
    provider: Optional[str] = None,
    log_tag: str = "manual",
) -> dict[str, Any]:
    if not callable(feature_store_connect) or feature_store_insert_news is None:
        raise HTTPException(status_code=503, detail="Feature store unavailable for news ingest.")

    symbol_norm = symbol.strip().upper()
    provider_eff = _determine_news_provider(provider)
    since = datetime.now(timezone.utc) - timedelta(days=int(days))
    try:
        rows = await _fetch_news_articles(symbol_norm, since, provider_eff, limit=limit)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("ingest_news_failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"news fetch failed: {exc}") from exc

    if not rows:
        log_json("info", msg="news_ingest_empty", symbol=symbol_norm, source=provider_eff, tag=log_tag)
        return {"rows": 0, "source": provider_eff, "symbol": symbol_norm}

    try:
        con = feature_store_connect()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Feature store connect failed: {exc}") from exc

    t0 = time.perf_counter()
    hash_lines: list[str] = []
    for row in rows:
        ts_norm = _as_utc_datetime(row[1]) or datetime.now(timezone.utc)
        hash_lines.append(f"{row[0]}|{ts_norm.isoformat()}|{row[3]}|{row[4]}|{row[5]}")
    payload_hash = hashlib.sha256("\n".join(hash_lines).encode("utf-8", errors="ignore")).hexdigest()

    try:
        inserted = feature_store_insert_news(con, rows)
        duration_ms = int((time.perf_counter() - t0) * 1000)
        if feature_store_log_ingest_event:
            try:
                feature_store_log_ingest_event(
                    con,
                    as_of=datetime.now(timezone.utc).date(),
                    source=f"news:{provider_eff}:{log_tag}",
                    row_count=int(inserted),
                    sha256=payload_hash,
                    duration_ms=duration_ms,
                    ok=True,
                )
            except Exception as exc:
                logger.debug("log_ingest_event news ok failed: %s", exc)
        if NEWS_INGESTED_COUNTER is not None:
            try:
                NEWS_INGESTED_COUNTER.inc(max(0, int(inserted)))
            except Exception:
                pass
        log_json(
            "info",
            msg="news_ingest_ok",
            symbol=symbol_norm,
            source=provider_eff,
            rows=int(inserted),
            duration_ms=duration_ms,
            tag=log_tag,
        )
        return {"rows": int(inserted), "source": provider_eff, "symbol": symbol_norm}
    except Exception as exc:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        if feature_store_log_ingest_event:
            try:
                feature_store_log_ingest_event(
                    con,
                    as_of=datetime.now(timezone.utc).date(),
                    source=f"news:{provider_eff}:{log_tag}",
                    row_count=len(rows),
                    sha256=payload_hash,
                    duration_ms=duration_ms,
                    ok=False,
                    error=str(exc),
                )
            except Exception:
                pass
        log_json(
            "error",
            msg="news_ingest_fail",
            symbol=symbol_norm,
            source=provider_eff,
            error=str(exc),
            tag=log_tag,
        )
        raise HTTPException(status_code=500, detail=f"failed to insert news: {exc}") from exc
    finally:
        try:
            con.close()
        except Exception:
            pass


async def score_news(
    symbol: str,
    *,
    days: int,
    batch: int,
    log_tag: str = "manual",
) -> dict[str, Any]:
    if not callable(feature_store_connect):
        raise HTTPException(status_code=503, detail="Feature store unavailable for scoring.")

    symbol_norm = symbol.strip().upper()
    if not symbol_norm:
        raise HTTPException(status_code=400, detail="symbol required")

    cutoff = datetime.now(timezone.utc) - timedelta(days=int(days))
    try:
        con = feature_store_connect()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Feature store connect failed: {exc}") from exc

    try:
        rows = con.execute(
            """
            SELECT ts, source, title, url, summary
            FROM news_articles
            WHERE symbol = ? AND ts >= ? AND (sentiment IS NULL OR sentiment = 0)
            ORDER BY ts DESC
            LIMIT ?
            """,
            [symbol_norm, cutoff, int(batch)],
        ).fetchall()
    except Exception as exc:
        try:
            con.close()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"failed to load news: {exc}") from exc

    if not rows:
        try:
            con.close()
        except Exception:
            pass
        log_json("info", msg="news_score_empty", symbol=symbol_norm, tag=log_tag)
        return {"rows": 0, "symbol": symbol_norm}

    oai_key = os.getenv("OPENAI_API_KEY", "").strip()
    xai_key = os.getenv("XAI_API_KEY", "").strip()
    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    prefer_xai = (provider == "xai") or (not oai_key and bool(xai_key))

    json_schema = {
        "name": "NewsSentimentScore",
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "minLength": 20, "maxLength": 240},
                "sentiment": {"type": "number", "minimum": -1.0, "maximum": 1.0},
            },
            "required": ["summary", "sentiment"],
            "additionalProperties": False,
        },
    }

    updates: list[tuple] = []
    for ts, source, title, url_item, summary in rows:
        prompt_user = {
            "role": "user",
            "content": (
                "You are a financial analyst assistant. Analyze the following news headline for symbol "
                f"{symbol_norm} and respond with JSON containing 'summary' (one sentence, <=200 chars) and "
                "'sentiment' (float in [-1,1], where -1 is very negative for the symbol and 1 is very positive).\n\n"
                f"Source: {source}\n"
                f"Title: {title}\n"
                f"URL: {url_item}\n"
                f"Existing summary: {summary or ''}\n"
            ),
        }
        try:
            out = await llm_summarize(
                prompt_user,
                prefer_xai=prefer_xai,
                xai_key=xai_key or None,
                oai_key=oai_key or None,
                json_schema=json_schema,
            )
        except Exception as exc:
            logger.debug("score_news llm failed for %s: %s", symbol_norm, exc)
            out = {}

        cleaned_summary = ""
        if isinstance(out, dict):
            cleaned_summary = str(out.get("summary") or "").strip()
        if not cleaned_summary:
            cleaned_summary = summary or title or ""
        cleaned_summary = cleaned_summary.strip()
        if len(cleaned_summary) > 240:
            cleaned_summary = cleaned_summary[:237].rstrip() + "..."

        sentiment_val = None
        if isinstance(out, dict):
            sentiment_val = _first_number([out.get("sentiment")])
        if sentiment_val is None:
            sentiment_val = 0.0
        sentiment_val = float(np.clip(float(sentiment_val), -1.0, 1.0))
        ts_norm = _as_utc_datetime(ts) or datetime.now(timezone.utc)

        updates.append(
            (
                cleaned_summary,
                sentiment_val,
                symbol_norm,
                ts_norm,
                (source or "").strip(),
                (url_item or "").strip(),
            )
        )

    try:
        con.executemany(
            """
            UPDATE news_articles
            SET summary = ?, sentiment = ?
            WHERE symbol = ? AND ts = ? AND source = ? AND url = ?
            """,
            updates,
        )
        updated = len(updates)
        con.close()
        if NEWS_SCORED_COUNTER is not None:
            try:
                NEWS_SCORED_COUNTER.inc(max(0, updated))
            except Exception:
                pass
        log_json("info", msg="news_score_ok", symbol=symbol_norm, rows=updated, tag=log_tag)
        return {"rows": updated, "symbol": symbol_norm}
    except Exception as exc:
        try:
            con.close()
        except Exception:
            pass
        log_json("error", msg="news_score_fail", symbol=symbol_norm, error=str(exc), tag=log_tag)
        raise HTTPException(status_code=500, detail=f"failed to update news sentiment: {exc}") from exc


async def fetch_social(symbol: str, handles: Iterable[str] | None = None) -> dict[str, Any]:
    """Lightweight deterministic social sentiment generator for X handles."""

    symbol_norm = _to_app_symbol(symbol)
    handles_list = [h.strip().lstrip("@") for h in (handles or []) if isinstance(h, str) and h.strip()]
    handles_norm = [h.lower() for h in handles_list]

    key_bits = [symbol_norm.upper()]
    key_bits.extend(handles_norm)
    key = "|".join(key_bits) or symbol_norm.upper()
    digest = hashlib.sha256(key.encode("utf-8", errors="ignore")).digest()

    base = int.from_bytes(digest[:4], "big") / float(0xFFFFFFFF)
    sentiment = float(np.clip(base * 2.0 - 1.0, -1.0, 1.0))

    phrases = [
        "seeing momentum build",
        "vol remains elevated",
        "watching macro catalysts",
        "desk chatter mixed",
        "options flow leaning bullish",
    ]
    samples: list[dict[str, Any]] = []
    for idx, handle in enumerate(handles_norm[:5]):
        start = 4 + idx * 4
        chunk = digest[start:start + 4]
        if len(chunk) < 4:
            chunk = (chunk + digest[:4])[:4]
        val = int.from_bytes(chunk, "big") / float(0xFFFFFFFF)
        tone = "bullish" if val > 0.55 else ("bearish" if val < 0.45 else "neutral")
        samples.append(
            {
                "handle": handle,
                "tone": tone,
                "score": float(np.clip((val - 0.5) * 2.0, -1.0, 1.0)),
                "text": f"@{handle} {phrases[idx % len(phrases)]}",
            }
        )

    return {
        "symbol": symbol_norm,
        "handles": handles_norm,
        "sentiment": sentiment,
        "sample": samples,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


async def load_option_surface(symbol: str, asof: datetime | date | None = None) -> dict[str, Any]:
    """Approximate an implied volatility surface using recent realized volatility."""

    as_of = _as_utc_datetime(asof) or datetime.now(timezone.utc)
    symbol_norm = _to_app_symbol(symbol)
    closes = await fetch_cached_hist_prices(symbol_norm, window_days=365, redis=REDIS)
    if len(closes) < 5:
        return {
            "symbol": symbol_norm,
            "as_of": as_of.isoformat(),
            "surface": [],
            "source": "realized_proxy",
            "note": "insufficient_history",
        }

    arr = np.asarray(closes, dtype=float)
    rets = np.diff(np.log(arr))
    rets = rets[np.isfinite(rets)]
    horizons = [7, 14, 21, 30, 45, 60, 90, 120, 180]
    surface: list[dict[str, Any]] = []
    for window in horizons:
        if rets.size < max(5, window):
            continue
        window_rets = rets[-window:]
        sigma_daily = float(np.std(window_rets, ddof=1))
        if not math.isfinite(sigma_daily) or sigma_daily <= 0:
            continue
        sigma_ann = float(np.clip(sigma_daily * math.sqrt(252.0), 1e-4, 3.0))
        surface.append({"tenor_days": int(window), "iv": sigma_ann})

    if not surface:
        sigma_daily = float(np.std(rets, ddof=1)) if rets.size >= 5 else 0.0
        sigma_ann = float(np.clip(sigma_daily * math.sqrt(252.0), 1e-4, 3.0)) if sigma_daily > 0 else 0.2
        surface.append({"tenor_days": 30, "iv": sigma_ann})

    ivs = np.array([row["iv"] for row in surface], dtype=float)
    if ivs.size >= 3:
        kernel_size = min(5, ivs.size)
        kernel = np.ones(kernel_size, dtype=float) / float(kernel_size)
        smooth = np.convolve(ivs, kernel, mode="same")
        for entry, val in zip(surface, smooth):
            entry["iv"] = float(np.clip(val, 1e-4, 3.0))

    return {
        "symbol": symbol_norm,
        "as_of": as_of.isoformat(),
        "surface": surface,
        "source": "realized_proxy",
    }


async def load_futures_curve(symbol: str, asof: datetime | date | None = None) -> dict[str, Any]:
    """Estimate a futures curve using recent drift as a carry proxy."""

    as_of = _as_utc_datetime(asof) or datetime.now(timezone.utc)
    symbol_norm = _to_app_symbol(symbol)
    closes = await fetch_cached_hist_prices(symbol_norm, window_days=365, redis=REDIS)
    if len(closes) < 5:
        return {
            "symbol": symbol_norm,
            "as_of": as_of.isoformat(),
            "curve": [],
            "source": "drift_proxy",
            "note": "insufficient_history",
        }

    arr = np.asarray(closes, dtype=float)
    rets = np.diff(np.log(arr))
    rets = rets[np.isfinite(rets)]
    if rets.size == 0:
        mu_daily = 0.0
    else:
        lookback = min(60, rets.size)
        mu_daily = float(np.mean(rets[-lookback:]))
    mu_ann = float(np.clip(mu_daily * 252.0, -5.0, 5.0))

    horizons = [7, 30, 60, 90, 180, 365]
    curve: list[dict[str, Any]] = []
    spot = float(arr[-1])
    for tenor in horizons:
        decay = math.exp(-tenor / 365.0)
        annualized = float(mu_ann * decay)
        forward = float(spot * math.exp(annualized * tenor / 252.0))
        curve.append(
            {
                "tenor_days": int(tenor),
                "annualized_carry": annualized,
                "forward_price": forward,
            }
        )

    return {
        "symbol": symbol_norm,
        "as_of": as_of.isoformat(),
        "curve": curve,
        "source": "drift_proxy",
    }


async def _fetch_earnings_polygon(symbol: str, lookback_days: int, limit: int = 16) -> list[tuple]:
    key = _polygon_api_key()
    if not key:
        raise HTTPException(status_code=400, detail="Polygon key missing for earnings ingest.")
    url = "https://api.polygon.io/v3/reference/financials"
    params = {
        "ticker": symbol,
        "timeframe": "quarterly",
        "limit": int(max(1, min(limit, 50))),
        "order": "desc",
        "sort": "reportPeriod",
        "apiKey": key,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json() or {}
    results = data.get("results", []) or []
    since_date = datetime.now(timezone.utc).date() - timedelta(days=int(max(1, lookback_days)))
    rows: list[tuple] = []
    for item in results:
        rep_raw = item.get("reportPeriod") or item.get("calendarDate")
        try:
            rep_date = date.fromisoformat(str(rep_raw))
        except Exception:
            continue
        if rep_date < since_date:
            continue

        eps_val = _first_number(
            [
                item.get("eps"),
                item.get("epsDiluted"),
                (item.get("earnings") or {}).get("eps"),
                (item.get("incomeStatement") or {}).get("basicEPS"),
                (item.get("incomeStatement") or {}).get("dilutedEPS"),
            ]
        )
        est_val = _first_number(
            [
                item.get("estimateEPS"),
                (item.get("earnings") or {}).get("epsEstimate"),
                (item.get("analystEstimates") or {}).get("epsEstimate"),
            ]
        )
        surprise = None
        if eps_val is not None and est_val not in (None, 0.0):
            denom = abs(est_val)
            if denom > 1e-9:
                surprise = (eps_val - est_val) / denom

        revenue_val = _first_number(
            [
                item.get("revenue"),
                (item.get("incomeStatement") or {}).get("revenue"),
                (item.get("incomeStatement") or {}).get("totalRevenue"),
            ]
        )
        guidance_val = _first_number(
            [
                (item.get("guidance") or {}).get("revenue"),
                item.get("guidanceRevenue"),
            ]
        )
        guidance_delta = None
        if guidance_val is not None and revenue_val is not None:
            try:
                guidance_delta = float(guidance_val) - float(revenue_val)
            except Exception:
                guidance_delta = None

        rows.append((symbol, rep_date, eps_val, surprise, revenue_val, guidance_delta))
        if len(rows) >= limit:
            break
    return rows


async def _fetch_earnings_rows(symbol: str, lookback_days: int, provider: str, limit: int = 16) -> list[tuple]:
    provider = provider.strip().lower()
    if provider == "polygon":
        return await _fetch_earnings_polygon(symbol, lookback_days, limit=limit)
    raise HTTPException(status_code=400, detail=f"Unsupported earnings provider '{provider}'.")


async def ingest_earnings(
    symbol: str,
    *,
    lookback_days: int,
    limit: int,
    provider: Optional[str] = None,
    log_tag: str = "manual",
) -> dict[str, Any]:
    if not callable(feature_store_connect) or feature_store_insert_earnings is None:
        raise HTTPException(status_code=503, detail="Feature store unavailable for earnings ingest.")

    symbol_norm = symbol.strip().upper()
    provider_eff = _determine_earnings_provider(provider)
    try:
        rows = await _fetch_earnings_rows(symbol_norm, lookback_days=int(lookback_days), provider=provider_eff, limit=limit)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("ingest_earnings_failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"earnings fetch failed: {exc}") from exc

    if not rows:
        log_json("info", msg="earnings_ingest_empty", symbol=symbol_norm, source=provider_eff, tag=log_tag)
        return {"rows": 0, "symbol": symbol_norm}

    try:
        con = feature_store_connect()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Feature store connect failed: {exc}") from exc

    t0 = time.perf_counter()
    payload_hash = hashlib.sha256(
        "\n".join(f"{r[0]}|{r[1]}|{r[2]}|{r[3]}|{r[4]}|{r[5]}" for r in rows).encode("utf-8", errors="ignore")
    ).hexdigest()

    try:
        inserted = feature_store_insert_earnings(con, rows)
        duration_ms = int((time.perf_counter() - t0) * 1000)
        if feature_store_log_ingest_event:
            try:
                feature_store_log_ingest_event(
                    con,
                    as_of=datetime.now(timezone.utc).date(),
                    source=f"earnings:{provider_eff}:{log_tag}",
                    row_count=int(inserted),
                    sha256=payload_hash,
                    duration_ms=duration_ms,
                    ok=True,
                )
            except Exception as exc:
                logger.debug("log_ingest_event earnings ok failed: %s", exc)
        if EARNINGS_INGESTED_COUNTER is not None:
            try:
                EARNINGS_INGESTED_COUNTER.inc(max(0, int(inserted)))
            except Exception:
                pass
        log_json(
            "info",
            msg="earnings_ingest_ok",
            symbol=symbol_norm,
            source=provider_eff,
            rows=int(inserted),
            duration_ms=duration_ms,
            tag=log_tag,
        )
        return {"rows": int(inserted), "symbol": symbol_norm}
    except Exception as exc:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        if feature_store_log_ingest_event:
            try:
                feature_store_log_ingest_event(
                    con,
                    as_of=datetime.now(timezone.utc).date(),
                    source=f"earnings:{provider_eff}:{log_tag}",
                    row_count=len(rows),
                    sha256=payload_hash,
                    duration_ms=duration_ms,
                    ok=False,
                    error=str(exc),
                )
            except Exception:
                pass
        log_json("error", msg="earnings_ingest_fail", symbol=symbol_norm, error=str(exc), tag=log_tag)
        raise HTTPException(status_code=500, detail=f"failed to insert earnings: {exc}") from exc
    finally:
        try:
            con.close()
        except Exception:
            pass


async def _fred_fetch_series(series_id: str, api_key: str, limit: int) -> list[tuple[date, float]]:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": int(max(1, limit)),
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json() or {}
    obs = data.get("observations", []) or []
    rows: list[tuple[date, float]] = []
    for item in obs:
        val = (item.get("value") or "").strip()
        if not val or val == ".":
            continue
        try:
            value = float(val)
        except Exception:
            continue
        try:
            d = date.fromisoformat(str(item.get("date")))
        except Exception:
            continue
        rows.append((d, value))
    return rows


async def _fetch_macro_rows(provider: str) -> list[tuple]:
    provider = (provider or "fred").strip().lower()
    if provider != "fred":
        raise HTTPException(status_code=400, detail=f"Unsupported macro provider '{provider}'.")
    api_key = (os.getenv("PT_FRED_API_KEY") or os.getenv("FRED_API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="FRED API key missing (set PT_FRED_API_KEY).")
    rff_series, cpi_series, u_series = await asyncio.gather(
        _fred_fetch_series("DGS3MO", api_key, limit=15),
        _fred_fetch_series("CPIAUCSL", api_key, limit=30),
        _fred_fetch_series("UNRATE", api_key, limit=15),
    )

    def _latest(series: list[tuple[date, float]]) -> tuple[date | None, float | None]:
        series_sorted = sorted(series, key=lambda x: x[0], reverse=True)
        for d, v in series_sorted:
            return d, v
        return (None, None)

    rff_date, rff_val = _latest(rff_series)
    u_date, u_val = _latest(u_series)

    cpi_sorted = sorted(cpi_series, key=lambda x: x[0], reverse=True)
    cpi_date, cpi_val = _latest(cpi_sorted)
    prev_val = None
    if cpi_date:
        for d, v in cpi_sorted[1:]:
            delta_months = (cpi_date.year - d.year) * 12 + (cpi_date.month - d.month)
            if delta_months >= 12:
                prev_val = v
                break
    cpi_yoy = None
    if cpi_val is not None and prev_val not in (None, 0.0):
        denom = abs(prev_val)
        if denom > 1e-9:
            cpi_yoy = ((cpi_val - prev_val) / denom) * 100.0

    candidates = [d for d in (rff_date, cpi_date, u_date) if d is not None]
    if not candidates:
        return []
    as_of = max(candidates)
    return [(as_of, rff_val, cpi_yoy, u_val)]


async def ingest_macro(
    provider: Optional[str] = None,
    *,
    log_tag: str = "manual",
) -> dict[str, Any]:
    if not callable(feature_store_connect) or feature_store_upsert_macro is None:
        raise HTTPException(status_code=503, detail="Feature store unavailable for macro ingest.")

    provider_eff = _determine_macro_provider(provider)
    try:
        rows = await _fetch_macro_rows(provider_eff)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("ingest_macro_failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"macro fetch failed: {exc}") from exc

    if not rows:
        log_json("info", msg="macro_ingest_empty", source=provider_eff, tag=log_tag)
        return {"rows": 0, "source": provider_eff}

    try:
        con = feature_store_connect()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Feature store connect failed: {exc}") from exc

    t0 = time.perf_counter()
    payload_hash = hashlib.sha256(
        "\n".join(f"{r[0]}|{r[1]}|{r[2]}|{r[3]}" for r in rows).encode("utf-8", errors="ignore")
    ).hexdigest()

    try:
        inserted = feature_store_upsert_macro(con, rows)
        duration_ms = int((time.perf_counter() - t0) * 1000)
        if feature_store_log_ingest_event:
            try:
                feature_store_log_ingest_event(
                    con,
                    as_of=rows[0][0],
                    source=f"macro:{provider_eff}:{log_tag}",
                    row_count=int(inserted),
                    sha256=payload_hash,
                    duration_ms=duration_ms,
                    ok=True,
                )
            except Exception as exc:
                logger.debug("log_ingest_event macro ok failed: %s", exc)
        if MACRO_UPSERTS_COUNTER is not None:
            try:
                MACRO_UPSERTS_COUNTER.inc(max(0, int(inserted)))
            except Exception:
                pass
        log_json(
            "info",
            msg="macro_ingest_ok",
            source=provider_eff,
            rows=int(inserted),
            duration_ms=duration_ms,
            tag=log_tag,
        )
        return {"rows": int(inserted), "source": provider_eff}
    except Exception as exc:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        if feature_store_log_ingest_event:
            try:
                feature_store_log_ingest_event(
                    con,
                    as_of=rows[0][0],
                    source=f"macro:{provider_eff}:{log_tag}",
                    row_count=len(rows),
                    sha256=payload_hash,
                    duration_ms=duration_ms,
                    ok=False,
                    error=str(exc),
                )
            except Exception:
                pass
        log_json("error", msg="macro_ingest_fail", source=provider_eff, error=str(exc), tag=log_tag)
        raise HTTPException(status_code=500, detail=f"failed to upsert macro snapshot: {exc}") from exc
    finally:
        try:
            con.close()
        except Exception:
            pass


async def fetch_hist_prices(symbol: str, window_days: int | None = None, *, key: str | None = None) -> list[float]:
    polygon_symbol = _to_polygon_symbol(symbol)
    api_key = (key or _polygon_api_key()).strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="Polygon key missing for historical price fetch.")

    days = settings.lookback_days_max if window_days is None else max(1, int(window_days))
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)

    url = f"https://api.polygon.io/v2/aggs/ticker/{polygon_symbol}/range/1/day/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": "50000"}
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            payload = resp.json() or {}
    except httpx.HTTPStatusError as exc:
        if exc.response is not None and exc.response.status_code == 429:
            raise HTTPException(status_code=429, detail="Rate limited by Polygon") from exc
        logger.exception("polygon_hist_prices_http_error: symbol=%s error=%s", polygon_symbol, exc)
        return []
    except Exception as exc:
        logger.exception("polygon_hist_prices_failed: symbol=%s error=%s", polygon_symbol, exc)
        return []

    results = payload.get("results") or []
    closes: list[float] = []
    for item in results:
        if isinstance(item, dict) and "c" in item:
            try:
                closes.append(float(item.get("c")))
            except (TypeError, ValueError):
                continue
    return closes


async def fetch_cached_hist_prices(
    symbol: str,
    window_days: int,
    redis: Redis | None = None,
) -> list[float]:
    window = int(max(1, window_days))
    today_str = datetime.now(timezone.utc).date().isoformat()
    app_symbol = _to_app_symbol(symbol)
    cache_key = f"hist_prices:{app_symbol}:{window}:{today_str}"

    redis_client: Redis | None = redis or REDIS
    if redis_client:
        try:
            cached = await redis_client.get(cache_key)
        except Exception:
            cached = None
        if cached:
            try:
                if isinstance(cached, (bytes, bytearray)):
                    cached_text = cached.decode("utf-8", errors="ignore")
                else:
                    cached_text = str(cached)
                data = json.loads(cached_text)
                if isinstance(data, list) and data and all(isinstance(x, (int, float)) for x in data):
                    return [float(x) for x in data][-window:]
            except Exception:
                pass

    closes: list[float] = []
    if callable(feature_store_connect):
        try:
            con = feature_store_connect()
        except Exception as exc:
            logger.debug("hist_price_cache_fs_connect_failed: %s", exc)
            con = None
        if con is not None:
            try:
                rows = con.execute(
                    "SELECT ts, close FROM bars_daily WHERE symbol = ? ORDER BY ts ASC",
                    [app_symbol],
                ).fetchall()
            except Exception as exc:
                logger.debug("hist_price_cache_query_failed: symbol=%s error=%s", app_symbol, exc)
                rows = []
            finally:
                try:
                    con.close()
                except Exception:
                    pass
            if rows:
                closes = [float(r[1]) for r in rows][-window:]

    min_required = min(30, window // 2 or 1)
    free_mode = os.getenv("PT_FREE_MODE", "0") == "1"
    if (not closes or len(closes) < min_required) and not free_mode:
        key = _polygon_api_key()
        if not key:
            if closes:
                return closes
            raise HTTPException(status_code=400, detail="Polygon key missing for historical price fetch.")
        closes = await fetch_hist_prices(app_symbol, window, key=key)

    if redis_client:
        try:
            await redis_client.setex(cache_key, 1800, json.dumps(closes))
        except Exception:
            pass

    return closes


async def ingest_grouped_daily(
    as_of: date,
    *,
    log_tag: str = "manual",
) -> dict[str, Any]:
    if isinstance(as_of, datetime):
        as_of = as_of.date()

    if not callable(feature_store_connect):
        raise HTTPException(status_code=503, detail="Feature store unavailable for grouped daily ingest.")

    api_key = _polygon_api_key()
    if not api_key:
        raise HTTPException(status_code=400, detail="Polygon key missing for grouped daily ingest.")

    day_iso = as_of.isoformat()
    params = {"adjusted": "true", "apiKey": api_key}

    async def _fetch_json(client: httpx.AsyncClient, url: str, params: dict[str, str]) -> dict[str, Any]:
        base_delays = (0.8, 1.6, 3.2, 6.4)
        for delay in (*base_delays, None):
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json() or {}
                if isinstance(data, dict):
                    return data
                return {}
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status in (429, 500, 502, 503, 504) and delay is not None:
                    retry_after = exc.response.headers.get("Retry-After") if exc.response is not None else None
                    reset = exc.response.headers.get("X-RateLimit-Reset") if exc.response is not None else None
                    sleep_seconds: float | None = None
                    if retry_after:
                        try:
                            sleep_seconds = float(retry_after)
                        except Exception:
                            sleep_seconds = None
                    if sleep_seconds is None and reset:
                        try:
                            sleep_seconds = max(0.0, float(reset) - time.time())
                        except Exception:
                            sleep_seconds = None
                    if sleep_seconds is None:
                        jitter = random.random()
                        sleep_seconds = delay + jitter
                    await asyncio.sleep(min(15.0, max(0.5, sleep_seconds)))
                    continue
                raise
            except httpx.HTTPError:
                if delay is not None:
                    await asyncio.sleep(delay)
                    continue
                raise
        return {}

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        stocks = await _fetch_json(
            client,
            f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{day_iso}",
            params,
        )
        crypto = await _fetch_json(
            client,
            f"https://api.polygon.io/v2/aggs/grouped/locale/global/market/crypto/{day_iso}",
            params,
        )

    rows_stocks = stocks.get("results") or []
    rows_crypto = crypto.get("results") or []

    to_upsert: list[tuple[str, str, float, float, float, float, float]] = []
    for row in rows_stocks:
        if not isinstance(row, dict):
            continue
        ticker = (row.get("T") or "").strip().upper()
        if not ticker:
            continue
        to_upsert.append(
            (
                ticker,
                day_iso,
                float(row.get("o") or 0.0),
                float(row.get("h") or 0.0),
                float(row.get("l") or 0.0),
                float(row.get("c") or 0.0),
                float(row.get("v") or 0.0),
            )
        )

    for row in rows_crypto:
        if not isinstance(row, dict):
            continue
        raw = (row.get("T") or "").strip()
        ticker = _poly_crypto_to_app(raw)
        if not ticker or "-USD" not in ticker:
            continue
        to_upsert.append(
            (
                ticker,
                day_iso,
                float(row.get("o") or 0.0),
                float(row.get("h") or 0.0),
                float(row.get("l") or 0.0),
                float(row.get("c") or 0.0),
                float(row.get("v") or 0.0),
            )
        )

    if not to_upsert:
        log_json("info", msg="grouped_daily_ingest_empty", date=day_iso, tag=log_tag)
        return {"ok": True, "upserted": 0, "date": day_iso}

    payload_hash = hashlib.sha256(
        json.dumps(to_upsert, separators=(",", ":"), sort_keys=True).encode("utf-8", errors="ignore")
    ).hexdigest()

    conn = feature_store_connect()
    t_start = time.perf_counter()
    try:
        conn.execute("BEGIN")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bars_daily (
                symbol TEXT,
                ts DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY(symbol, ts)
            )
            """
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO bars_daily
            (symbol, ts, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            to_upsert,
        )
        conn.execute("COMMIT")
    except Exception as exc:
        conn.execute("ROLLBACK")
        duration_ms = int((time.perf_counter() - t_start) * 1000)
        if feature_store_log_ingest_event:
            try:
                feature_store_log_ingest_event(
                    conn,
                    as_of=day_iso,
                    source=f"grouped_daily:{log_tag}",
                    row_count=len(to_upsert),
                    sha256=payload_hash,
                    duration_ms=duration_ms,
                    ok=False,
                    error=str(exc),
                )
            except Exception:
                pass
        log_json(
            "error",
            msg="grouped_daily_ingest_fail",
            date=day_iso,
            rows=len(to_upsert),
            error=str(exc),
            tag=log_tag,
        )
        raise HTTPException(status_code=500, detail=f"Failed to upsert grouped daily bars: {exc}") from exc
    else:
        duration_ms = int((time.perf_counter() - t_start) * 1000)
        if feature_store_log_ingest_event:
            try:
                feature_store_log_ingest_event(
                    conn,
                    as_of=day_iso,
                    source=f"grouped_daily:{log_tag}",
                    row_count=len(to_upsert),
                    sha256=payload_hash,
                    duration_ms=duration_ms,
                    ok=True,
                )
            except Exception as exc:
                logger.debug("log_ingest_event grouped_daily ok failed: %s", exc)
        log_json(
            "info",
            msg="grouped_daily_ingest_ok",
            date=day_iso,
            rows=len(to_upsert),
            duration_ms=duration_ms,
            tag=log_tag,
        )
        return {"ok": True, "upserted": len(to_upsert), "date": day_iso}
    finally:
        try:
            conn.close()
        except Exception:
            pass


__all__ = [
    "ingest_news",
    "score_news",
    "fetch_recent_news",
    "fetch_social",
    "load_option_surface",
    "load_futures_curve",
    "ingest_earnings",
    "ingest_macro",
    "fetch_hist_prices",
    "fetch_cached_hist_prices",
    "ingest_grouped_daily",
]
