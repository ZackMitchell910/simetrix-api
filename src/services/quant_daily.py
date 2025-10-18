from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
from fastapi import HTTPException

from src.core import REDIS
from src.services.inference import get_ensemble_prob
from src.services.labeling import to_polygon_ticker
from src.services.quant_utils import ewma_sigma, horizon_shrink, simulate_gbm_student_t, winsorize
from . import ingestion as ingestion_service

logger = logging.getLogger("simetrix.services.quant_daily")

TTL_DAY_SECONDS = 27 * 3600
TTL_MINIMAL_SECONDS = 6 * 3600


@dataclass
class MinimalSummary:
    symbol: str
    horizon_days: int
    prob_up: float
    base_price: float
    predicted_price: float
    bullish_price: float
    spot_price: float
    median_return_pct: float
    blurb: str | None = None

    def as_dict(self) -> Dict[str, Any]:
        price_pred = self.predicted_price if math.isfinite(self.predicted_price) else self.base_price
        bullish = self.bullish_price if math.isfinite(self.bullish_price) else self.base_price
        spot = self.spot_price if math.isfinite(self.spot_price) and self.spot_price > 0 else price_pred
        median_ret = self.median_return_pct
        if not math.isfinite(median_ret):
            try:
                median_ret = float(((self.base_price / spot) - 1.0) * 100.0)
            except Exception:
                median_ret = float("nan")
        blurb = (self.blurb or "").strip()
        if not blurb:
            blurb = _format_blurb(self.symbol, self.horizon_days, self.prob_up, median_ret)
        payload = {
            "symbol": self.symbol,
            "horizon_days": int(self.horizon_days),
            "prob_up_end": float(self.prob_up),
            "median_return_pct": round(median_ret, 3) if math.isfinite(median_ret) else None,
            "prob_up_30d": round(self.prob_up, 4),
            "base_price": round(self.base_price, 2),
            "predicted_price": round(price_pred, 2),
            "bullish_price": round(bullish, 2),
            "spot_price": round(spot, 2) if math.isfinite(spot) else None,
            "blurb": blurb,
            "p95": round(bullish, 2),  # legacy alias
        }
        return payload


def _format_blurb(symbol: str, horizon_days: int, prob_up: float, median_return_pct: float) -> str:
    sym = symbol.upper().strip()
    prob_pct = prob_up * 100.0
    if math.isfinite(median_return_pct):
        return f"{sym}: {horizon_days}d outlook P(up)≈{prob_pct:.0f}%, median ≈{median_return_pct:.1f}%"
    return f"{sym}: {horizon_days}d outlook P(up)≈{prob_pct:.0f}%"


def _median_return_pct(base_price: Optional[float], spot_price: Optional[float]) -> Optional[float]:
    if base_price is None or spot_price is None or spot_price <= 0:
        return None
    try:
        return float(((float(base_price) / float(spot_price)) - 1.0) * 100.0)
    except Exception:
        return None


def _to_number(value: Any) -> Optional[float]:
    try:
        num = float(value)
        if math.isfinite(num):
            return num
    except (TypeError, ValueError):
        return None
    return None


def _normalize_prob(value: Any) -> Optional[float]:
    num = _to_number(value)
    if num is None:
        return None
    if num > 1.0:
        # Accept percentage-style inputs (e.g., 62.5) and normalise to 0-1.
        num = num / 100.0
    return float(np.clip(num, 0.0, 1.0))


def _normalize_horizon(value: Any, fallback: Optional[int]) -> Optional[int]:
    try:
        if value is None:
            raise ValueError("missing")
        hz = int(value)
        if hz <= 0:
            raise ValueError("non-positive horizon")
        return hz
    except Exception:
        if fallback is None:
            return None
        return max(1, int(fallback))


def _default_blurb(symbol: str, horizon: Optional[int], prob: Optional[float], median: Optional[float], regime: str) -> str:
    horizon_str = f"{int(horizon)}d" if isinstance(horizon, int) else "multi-day"
    prob_str = f"≈{prob:.0%}" if isinstance(prob, (int, float)) else "mixed odds"
    med_str = f"median ≈ {median:.1f}%" if isinstance(median, (int, float)) else "median path uncertain"
    regime = regime or "neutral"
    return f"{symbol}: {horizon_str} outlook {prob_str}, {med_str}. Regime {regime}."


def normalize_pick_document(
    doc: Mapping[str, Any] | None,
    *,
    fallback_horizon: Optional[int] = None,
    required_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return a copy of *doc* with the daily-card fields coerced and defaults filled.

    The frontend relies on ``prob_up_end``, ``median_return_pct``, ``horizon_days`` and
    ``blurb`` being present. This helper enforces those invariants so that stale cache
    entries or external payloads do not break the DailyQuant card UI.
    """

    if not isinstance(doc, Mapping):
        return {}

    data = {k: v for k, v in doc.items()}
    symbol = str(data.get("symbol") or "").upper().strip()
    if not symbol:
        return {}

    prob = _normalize_prob(data.get("prob_up_end") or data.get("prob_up_30d"))
    if prob is not None:
        data["prob_up_end"] = prob

    median_pct = _to_number(data.get("median_return_pct"))
    if median_pct is None:
        base = _to_number(data.get("base_price"))
        spot = _to_number(data.get("spot")) or _to_number(data.get("spot0"))
        if spot not in (None, 0):
            try:
                median_pct = float(((base or spot) / spot - 1.0) * 100.0)
            except Exception:
                median_pct = None
    if median_pct is not None:
        data["median_return_pct"] = float(median_pct)

    horizon = _normalize_horizon(data.get("horizon_days"), fallback_horizon)
    if horizon is not None:
        data["horizon_days"] = horizon

    regime = str(data.get("regime") or "neutral").strip() or "neutral"
    if not data.get("blurb"):
        data["blurb"] = _default_blurb(symbol, horizon, prob, median_pct, regime)

    required = set(required_keys or ("prob_up_end", "median_return_pct", "horizon_days", "blurb"))
    missing = [key for key in required if key not in data]
    for key in missing:
        if key == "prob_up_end" and prob is None:
            data[key] = 0.5
        elif key == "median_return_pct" and median_pct is None:
            data[key] = 0.0
        elif key == "horizon_days" and horizon is None:
            data[key] = fallback_horizon or 30
        elif key == "blurb":
            data[key] = _default_blurb(symbol, data.get("horizon_days"), data.get("prob_up_end"), data.get("median_return_pct"), regime)

    data["symbol"] = symbol
    return data


def _cache_key(symbol: str, horizon_days: int, day_iso: Optional[str] = None) -> str:
    as_of = day_iso or datetime.now(timezone.utc).date().isoformat()
    return f"quant:daily:{as_of}:summary:{symbol}:{horizon_days}"


async def fetch_minimal_summary(symbol: str, horizon_days: int) -> Dict[str, Any]:
    symbol_clean = symbol.strip().upper()
    if not symbol_clean:
        raise HTTPException(status_code=400, detail="Symbol cannot be blank")

    redis = REDIS
    cache_key = _cache_key(symbol_clean, horizon_days)
    if redis:
        try:
            cached = await redis.get(cache_key)
            if cached:
                doc = json.loads(cached)
                prob = _to_number(doc.get("prob_up_30d"))
                base = _to_number(doc.get("base_price"))
                pred = _to_number(doc.get("predicted_price", base))
                bull = _to_number(doc.get("bullish_price", doc.get("p95")))
                if None not in (prob, base, bull):
                    if pred is None:
                        pred = base
                    spot = _to_number(doc.get("spot_price") or doc.get("spot0") or doc.get("spot"))
                    spot_val = float(spot) if spot is not None and math.isfinite(spot) else float(base)
                    median_pct = _to_number(doc.get("median_return_pct"))
                    if median_pct is None or not math.isfinite(median_pct):
                        maybe_med = _median_return_pct(base, spot_val)
                        median_pct = maybe_med if maybe_med is not None else float("nan")
                    summary = MinimalSummary(
                        symbol=symbol_clean,
                        horizon_days=horizon_days,
                        prob_up=prob,
                        base_price=base,
                        predicted_price=pred,
                        bullish_price=bull,
                        spot_price=spot_val,
                        median_return_pct=float(median_pct),
                        blurb=str(doc.get("blurb") or ""),
                    )
                    return summary.as_dict()
        except Exception:
            pass

    summary = await _compute_minimal_summary(symbol_clean, horizon_days)
    if redis:
        try:
            await redis.setex(cache_key, TTL_MINIMAL_SECONDS, json.dumps(summary.as_dict()))
        except Exception:
            pass
    return summary.as_dict()


async def _compute_minimal_summary(symbol: str, horizon_days: int) -> MinimalSummary:
    sym_fetch = to_polygon_ticker(symbol)
    window_days = max(60, min(540, int(horizon_days * 12)))
    prices = await ingestion_service.fetch_cached_hist_prices(sym_fetch, window_days, REDIS)
    if not prices or len(prices) < 30:
        raise HTTPException(status_code=503, detail=f"Insufficient history for {symbol}")

    arr = np.asarray(prices, dtype=float)
    s0 = float(arr[-1])
    history_window = min(len(arr) - 1, 260)
    rets = np.diff(np.log(arr[-(history_window + 1):]))
    if rets.size == 0 or not np.isfinite(rets).all():
        raise HTTPException(status_code=503, detail=f"Insufficient history for {symbol}")
    rets = winsorize(rets)

    scale = 252.0
    sigma_ann = float(np.clip(ewma_sigma(rets) * math.sqrt(scale), 1e-4, 5.0))
    mu_ann = float(np.clip(np.mean(rets) * scale, -1.5, 1.5))
    mu_ann *= horizon_shrink(horizon_days)

    try:
        p_up_ml = await asyncio.wait_for(get_ensemble_prob(symbol, REDIS, horizon_days), timeout=0.6)
        mu_ann = float(mu_ann + 0.30 * (2.0 * p_up_ml - 1.0) * sigma_ann)
    except Exception:
        pass

    paths = simulate_gbm_student_t(
        S0=s0,
        mu_ann=mu_ann,
        sigma_ann=sigma_ann,
        horizon_days=int(horizon_days),
        n_paths=4000,
        df_t=7,
        antithetic=True,
        seed=None,
    )
    terminal = paths[:, -1].astype(float)
    prob_up = float(np.mean(terminal > s0))
    base_price = float(np.median(terminal))
    bullish_price = float(np.percentile(terminal, 95))
    median_return_pct = float(np.median(terminal / s0 - 1.0) * 100.0)

    predicted_price = base_price
    valid = terminal[np.isfinite(terminal) & (terminal > 0)]
    if valid.size >= 10:
        log_t = np.log(valid)
        iqr = float(np.subtract(*np.percentile(log_t, [75, 25])))
        bw = 2.0 * iqr * (log_t.size ** (-1.0 / 3.0))
        if not np.isfinite(bw) or bw <= 1e-9:
            std = float(np.std(log_t))
            bw = max(1e-6, std * (log_t.size ** (-1.0 / 5.0)) if std > 0 else 1e-3)
        bins = int(np.clip(np.ceil((log_t.max() - log_t.min()) / max(1e-9, bw)), 30, 120))
        counts, edges = np.histogram(log_t, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        predicted_price = float(np.exp(centers[np.argmax(counts)]))
    if not math.isfinite(predicted_price):
        predicted_price = base_price

    return MinimalSummary(
        symbol=symbol,
        horizon_days=horizon_days,
        prob_up=prob_up,
        base_price=base_price,
        predicted_price=predicted_price,
        bullish_price=bullish_price,
        spot_price=float(s0),
        median_return_pct=median_return_pct,
    )


async def prefill_minimal_summary(
    symbol: str,
    doc: Optional[dict],
    as_of: str,
    horizon_days: int,
) -> None:
    redis = REDIS
    if not redis:
        return

    payload = await _payload_from_doc(symbol, doc, horizon_days)
    if payload is None:
        return

    cache_key = _cache_key(payload["symbol"], horizon_days, as_of)
    try:
        await redis.setex(cache_key, TTL_DAY_SECONDS, json.dumps(payload))
    except Exception:
        pass


async def _payload_from_doc(symbol: str, doc: Optional[dict], horizon_days: int) -> Optional[Dict[str, Any]]:
    info = doc or {}
    symbol_clean = symbol.strip().upper()
    if not symbol_clean:
        return None

    prob = _to_number(info.get("prob_up_end", info.get("prob_up_30d")))
    base = _to_number(info.get("base_price"))
    predicted = _to_number(info.get("predicted_price", info.get("most_likely_price")))
    bullish = _to_number(info.get("bullish_price", info.get("p95")))
    spot = _to_number(info.get("spot_price") or info.get("spot0") or info.get("spot"))
    median_pct = _to_number(info.get("median_return_pct"))
    blurb = str(info.get("blurb") or "").strip()

    if base is None:
        spot = _to_number(info.get("spot0") or info.get("spot"))
        med_pct = _to_number(info.get("median_return_pct"))
        if spot is not None and med_pct is not None:
            base = float(spot * (1.0 + med_pct / 100.0))

    if predicted is None:
        predicted = base

    spot_val: float
    median_val: Optional[float]

    if prob is None or base is None or bullish is None:
        refreshed = await _refresh_summary(symbol_clean, horizon_days)
        if refreshed is None:
            return None
        prob = refreshed.prob_up
        base = refreshed.base_price
        predicted = refreshed.predicted_price
        bullish = refreshed.bullish_price
        spot_val = float(refreshed.spot_price)
        median_val = float(refreshed.median_return_pct)
        if not blurb:
            blurb = refreshed.blurb or ""
    else:
        if predicted is None:
            predicted = base
        if not math.isfinite(bullish) and info.get("run_id"):
            try:
                raw_art = await REDIS.get(f"artifact:{info.get('run_id')}")
                if raw_art:
                    art_doc = json.loads(raw_art)
                    term_prices = art_doc.get("terminal_prices") or []
                    if term_prices:
                        bullish = _to_number(np.percentile(np.asarray(term_prices, dtype=float), 95))
                    if base is None:
                        med_path = art_doc.get("median_path") or []
                        if med_path:
                            base = _to_number(med_path[-1][1])
                    if predicted is None:
                        predicted = _to_number(art_doc.get("most_likely_price"))
            except Exception:
                pass
        if spot is not None and math.isfinite(spot):
            spot_val = float(spot)
        else:
            spot_val = float(base)
        if median_pct is not None and math.isfinite(median_pct):
            median_val = float(median_pct)
        else:
            median_val = _median_return_pct(base, spot_val)

    if predicted is None:
        predicted = base
    if None in (prob, base, predicted, bullish):
        return None

    if median_val is None:
        median_val = _median_return_pct(base, spot_val)
    if median_val is None:
        median_val = float("nan")

    summary = MinimalSummary(
        symbol=symbol_clean,
        horizon_days=horizon_days,
        prob_up=prob,
        base_price=base,
        predicted_price=predicted,
        bullish_price=bullish,
        spot_price=spot_val,
        median_return_pct=float(median_val),
        blurb=blurb,
    )
    return summary.as_dict()


async def _refresh_summary(symbol: str, horizon_days: int) -> Optional[MinimalSummary]:
    try:
        summary = await _compute_minimal_summary(symbol, horizon_days)
        return summary
    except HTTPException:
        raise
    except Exception as exc:
        logger.debug("refresh_summary failed for %s: %s", symbol, exc)
        return None
