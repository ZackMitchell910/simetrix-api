from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
from fastapi import HTTPException

from .. import predictive_service as svc  # type: ignore
from . import ingestion as ingestion_service

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

    def as_dict(self) -> Dict[str, Any]:
        price_pred = self.predicted_price if math.isfinite(self.predicted_price) else self.base_price
        bullish = self.bullish_price if math.isfinite(self.bullish_price) else self.base_price
        return {
            "symbol": self.symbol,
            "prob_up_30d": round(self.prob_up, 4),
            "base_price": round(self.base_price, 2),
            "predicted_price": round(price_pred, 2),
            "bullish_price": round(bullish, 2),
            "p95": round(bullish, 2),  # legacy alias
        }


def _cache_key(symbol: str, horizon_days: int, day_iso: Optional[str] = None) -> str:
    as_of = day_iso or datetime.now(timezone.utc).date().isoformat()
    return f"quant:daily:{as_of}:summary:{symbol}:{horizon_days}"


def _to_number(value: Any) -> Optional[float]:
    try:
        num = float(value)
        if math.isfinite(num):
            return num
    except (TypeError, ValueError):
        return None
    return None


async def fetch_minimal_summary(symbol: str, horizon_days: int) -> Dict[str, Any]:
    symbol_clean = symbol.strip().upper()
    if not symbol_clean:
        raise HTTPException(status_code=400, detail="Symbol cannot be blank")

    redis = svc.REDIS
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
                    summary = MinimalSummary(
                        symbol=symbol_clean,
                        horizon_days=horizon_days,
                        prob_up=prob,
                        base_price=base,
                        predicted_price=pred,
                        bullish_price=bull,
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
    sym_fetch = svc._to_polygon_ticker(symbol)
    window_days = max(60, min(540, int(horizon_days * 12)))
    prices = await ingestion_service.fetch_cached_hist_prices(sym_fetch, window_days, svc.REDIS)
    if not prices or len(prices) < 30:
        raise HTTPException(status_code=503, detail=f"Insufficient history for {symbol}")

    arr = np.asarray(prices, dtype=float)
    s0 = float(arr[-1])
    history_window = min(len(arr) - 1, 260)
    rets = np.diff(np.log(arr[-(history_window + 1):]))
    if rets.size == 0 or not np.isfinite(rets).all():
        raise HTTPException(status_code=503, detail=f"Insufficient history for {symbol}")
    rets = svc._winsorize(rets)

    scale = 252.0
    sigma_ann = float(np.clip(svc._ewma_sigma(rets) * math.sqrt(scale), 1e-4, 5.0))
    mu_ann = float(np.clip(np.mean(rets) * scale, -1.5, 1.5))
    mu_ann *= svc._horizon_shrink(horizon_days)

    try:
        p_up_ml = await asyncio.wait_for(svc.get_ensemble_prob(symbol, svc.REDIS, horizon_days), timeout=0.6)
        mu_ann = float(mu_ann + 0.30 * (2.0 * p_up_ml - 1.0) * sigma_ann)
    except Exception:
        pass

    paths = svc.simulate_gbm_student_t(
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
    )


async def prefill_minimal_summary(
    symbol: str,
    doc: Optional[dict],
    as_of: str,
    horizon_days: int,
) -> None:
    redis = svc.REDIS
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

    if base is None:
        spot = _to_number(info.get("spot0") or info.get("spot"))
        med_pct = _to_number(info.get("median_return_pct"))
        if spot is not None and med_pct is not None:
            base = float(spot * (1.0 + med_pct / 100.0))

    if predicted is None:
        predicted = base

    if prob is None or base is None or bullish is None:
        refreshed = await _refresh_summary(symbol_clean, horizon_days)
        if refreshed is None:
            return None
        prob = refreshed.prob_up
        base = refreshed.base_price
        predicted = refreshed.predicted_price
        bullish = refreshed.bullish_price
    else:
        if predicted is None:
            predicted = base
        if not math.isfinite(bullish) and info.get("run_id"):
            try:
                raw_art = await svc.REDIS.get(f"artifact:{info.get('run_id')}")
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

    if predicted is None:
        predicted = base
    if None in (prob, base, predicted, bullish):
        return None

    summary = MinimalSummary(
        symbol=symbol_clean,
        horizon_days=horizon_days,
        prob_up=prob,
        base_price=base,
        predicted_price=predicted,
        bullish_price=bullish,
    )
    return summary.as_dict()


async def _refresh_summary(symbol: str, horizon_days: int) -> Optional[MinimalSummary]:
    try:
        res = await svc._mc_summary(symbol, horizon_days)
    except HTTPException:
        raise
    except Exception as exc:
        svc.logger.debug("mc_summary refresh failed for %s: %s", symbol, exc)
        return None

    prob = _to_number(res.get("prob_up_end", res.get("prob_up_30d")))
    base = _to_number(res.get("base_price"))
    predicted = _to_number(res.get("predicted_price", res.get("most_likely_price")))
    bullish = _to_number(res.get("bullish_price", res.get("p95")))

    if base is None:
        spot = _to_number(res.get("spot0") or res.get("spot"))
        med_pct = _to_number(res.get("median_return_pct"))
        if spot is not None and med_pct is not None:
            base = float(spot * (1.0 + med_pct / 100.0))
    if predicted is None:
        predicted = base

    if None in (prob, base, predicted, bullish):
        return None

    return MinimalSummary(
        symbol=symbol,
        horizon_days=horizon_days,
        prob_up=prob,
        base_price=base,
        predicted_price=predicted,
        bullish_price=bullish,
    )
