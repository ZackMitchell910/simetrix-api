from __future__ import annotations

import logging
import math
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import numpy as np

from src.services.context import get_earnings_features, get_sentiment_features

logger = logging.getLogger("simetrix.services.quant_context")


def safe_sentiments() -> dict[str, float | int]:
    return {"avg_sent_7d": 0.0, "last24h": 0.0, "n_news": 0}


def safe_earnings() -> dict[str, float | int | None]:
    return {"surprise_last": 0.0, "guidance_delta": 0.0, "days_since_earn": None, "days_to_next": None}


def mu_bias_from_context(ctx: Mapping[str, Any]) -> float:
    sent = ctx.get("sentiment") or {}
    earn = ctx.get("earnings") or {}
    avg_sent = float(sent.get("avg_sent_7d") or 0.0)
    earn_surprise = float(earn.get("surprise_last") or 0.0)
    return float(np.clip(0.15 * avg_sent + 0.05 * earn_surprise, -0.15, 0.15))


def detect_regime(px: np.ndarray) -> dict[str, float]:
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
        name = "neutral"
        score = float(np.clip(m - 0.3 * v, -0.4, 0.4))

    return {"name": name, "score": float(np.clip(score, -1.0, 1.0))}


async def build_context_loader(
    cache: dict[str, dict[str, Any]] | None = None,
    *,
    get_sentiment: Callable[[str], Awaitable[dict[str, Any]]] = get_sentiment_features,
    get_earnings: Callable[[str], Awaitable[dict[str, Any]]] = get_earnings_features,
    default_sentiments: Callable[[], Mapping[str, Any]] = safe_sentiments,
    default_earnings: Callable[[], Mapping[str, Any]] = safe_earnings,
) -> Callable[[str], Awaitable[dict[str, Any]]]:
    context_cache = cache if cache is not None else {}

    async def _loader(symbol: str) -> dict[str, Any]:
        key = (symbol or "").upper().strip()
        if not key:
            return {"sentiment": dict(default_sentiments()), "earnings": dict(default_earnings())}
        if key in context_cache:
            return context_cache[key]
        try:
            sent = await get_sentiment(key)
        except Exception as exc:  # pragma: no cover - telemetry only
            logger.debug("daily_quant_sentiment_fail %s: %s", key, exc)
            sent = default_sentiments()
        try:
            earn = await get_earnings(key)
        except Exception as exc:  # pragma: no cover - telemetry only
            logger.debug("daily_quant_earnings_fail %s: %s", key, exc)
            earn = default_earnings()
        ctx = {
            "sentiment": sent or default_sentiments(),
            "earnings": earn or default_earnings(),
        }
        context_cache[key] = ctx
        return ctx

    return _loader


__all__ = [
    "safe_sentiments",
    "safe_earnings",
    "mu_bias_from_context",
    "detect_regime",
    "build_context_loader",
]

