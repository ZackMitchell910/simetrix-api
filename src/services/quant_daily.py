from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

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


async def ensure_daily_pick_contract(
    doc: Mapping[str, Any] | None,
    horizon_days: int,
    *,
    symbol: str | None = None,
    allow_refresh: bool = True,
) -> Dict[str, Any]:
    """Ensure the daily pick document exposes the frontend contract."""

    info: Dict[str, Any] = dict(doc or {})
    sym = (symbol or info.get("symbol") or "").strip().upper()
    if sym:
        info["symbol"] = sym

    horizon_val = _to_number(info.get("horizon_days"))
    hz = int(horizon_val) if horizon_val is not None else int(horizon_days or 30)
    info["horizon_days"] = hz

    prob = _to_number(info.get("prob_up_end"))
    if prob is None:
        prob = _to_number(info.get("prob_up_30d"))

    median_pct = _to_number(info.get("median_return_pct"))
    base_price = _to_number(info.get("base_price"))
    predicted_price = _to_number(info.get("predicted_price"))
    bullish_price = _to_number(info.get("bullish_price", info.get("p95")))

    refresh_payload: Dict[str, Any] | None = None
    if allow_refresh and sym and (prob is None or median_pct is None or base_price is None or bullish_price is None):
        try:
            refresh_payload = await fetch_minimal_summary(sym, hz)
        except HTTPException:
            refresh_payload = None
        except Exception:
            refresh_payload = None

    if refresh_payload:
        base_price = base_price if base_price is not None else _to_number(refresh_payload.get("base_price"))
        predicted_price = (
            predicted_price if predicted_price is not None else _to_number(refresh_payload.get("predicted_price"))
        )
        bullish_price = bullish_price if bullish_price is not None else _to_number(refresh_payload.get("bullish_price"))
        if prob is None:
            prob = _to_number(refresh_payload.get("prob_up_30d"))

    info["prob_up_end"] = float(prob if prob is not None else 0.5)
    info["median_return_pct"] = float(median_pct if median_pct is not None else 0.0)

    if base_price is not None:
        info["base_price"] = float(base_price)
    if predicted_price is not None:
        info["predicted_price"] = float(predicted_price)
    elif base_price is not None:
        info["predicted_price"] = float(base_price)
    if bullish_price is not None:
        info["bullish_price"] = float(bullish_price)

    regime = str(info.get("regime") or "neutral")
    info["regime"] = regime

    blurb = str(info.get("blurb") or "").strip()
    if not blurb and sym:
        prob_pct = info["prob_up_end"] * 100.0
        med = info["median_return_pct"]
        blurb = (
            f"{sym}: {hz}d MC P(up)≈{prob_pct:.1f}%, "
            f"median ≈ {med:.1f}% • regime {regime}."
        )
    info["blurb"] = blurb

    return info


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
    info: Mapping[str, Any] | None = doc
    symbol_clean = symbol.strip().upper()
    if not symbol_clean:
        return None

    ensured = await ensure_daily_pick_contract(info, horizon_days, symbol=symbol_clean, allow_refresh=False)
    prob = _to_number(ensured.get("prob_up_end"))
    base = _to_number(ensured.get("base_price"))
    predicted = _to_number(ensured.get("predicted_price") or ensured.get("most_likely_price"))
    bullish = _to_number(ensured.get("bullish_price") or ensured.get("p95"))

    if base is None:
        spot = _to_number(ensured.get("spot0") or ensured.get("spot"))
        med_pct = _to_number(ensured.get("median_return_pct"))
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
        if not math.isfinite(bullish) and ensured.get("run_id"):
            try:
                raw_art = await REDIS.get(f"artifact:{ensured.get('run_id')}")
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
        summary = await _compute_minimal_summary(symbol, horizon_days)
        return summary
    except HTTPException:
        raise
    except Exception as exc:
        logger.debug("refresh_summary failed for %s: %s", symbol, exc)
        return None


async def fetch_daily_snapshot(horizon_days: int) -> Dict[str, Any]:
    """Return the daily winners ensuring the consumer contract is respected."""

    today_iso = datetime.now(timezone.utc).date().isoformat()
    payload: Dict[str, Any] = {"as_of": today_iso}
    redis = REDIS
    equity_doc: Dict[str, Any] | None = None
    crypto_doc: Dict[str, Any] | None = None
    macro_ctx: Dict[str, Any] | None = None

    if redis:
        try:
            base_key = f"quant:daily:{today_iso}"
            eq_raw, cr_raw, macro_raw = await asyncio.gather(
                redis.get(f"{base_key}:equity"),
                redis.get(f"{base_key}:crypto"),
                redis.get(f"{base_key}:macro"),
            )
            if eq_raw:
                equity_doc = json.loads(eq_raw)
            if cr_raw:
                crypto_doc = json.loads(cr_raw)
            if macro_raw:
                try:
                    macro_ctx = json.loads(macro_raw)
                except Exception:
                    macro_ctx = None

            eq_top_raw, cr_top_raw = await asyncio.gather(
                redis.get(f"{base_key}:equity_top"),
                redis.get(f"{base_key}:crypto_top"),
            )
            if eq_top_raw:
                payload["equity_top"] = json.loads(eq_top_raw)
            if cr_top_raw:
                payload["crypto_top"] = json.loads(cr_top_raw)

            eq_final_raw, cr_final_raw = await asyncio.gather(
                redis.get(f"{base_key}:equity_finalists"),
                redis.get(f"{base_key}:crypto_finalists"),
            )
            if eq_final_raw:
                payload["equity_finalists"] = json.loads(eq_final_raw)
            if cr_final_raw:
                payload["crypto_finalists"] = json.loads(cr_final_raw)
        except Exception:
            equity_doc = equity_doc or None
            crypto_doc = crypto_doc or None

    if macro_ctx:
        payload["macro"] = macro_ctx

    if equity_doc:
        payload["equity"] = await ensure_daily_pick_contract(equity_doc, horizon_days)
    if crypto_doc:
        payload["crypto"] = await ensure_daily_pick_contract(crypto_doc, horizon_days)

    if "equity" not in payload or "crypto" not in payload:
        try:
            from src.services.quant_scheduler import run_daily_quant

            computed = await run_daily_quant(horizon_days=horizon_days)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("run_daily_quant fallback failed: %s", exc)
            return payload

        payload["as_of"] = str(computed.get("as_of") or today_iso)
        if computed.get("macro"):
            payload["macro"] = computed.get("macro")
        eq_computed = computed.get("equity")
        cr_computed = computed.get("crypto")
        if eq_computed:
            payload["equity"] = await ensure_daily_pick_contract(eq_computed, horizon_days, allow_refresh=False)
        if cr_computed:
            payload["crypto"] = await ensure_daily_pick_contract(cr_computed, horizon_days, allow_refresh=False)
        if computed.get("equity_top"):
            payload["equity_top"] = computed["equity_top"]
        if computed.get("crypto_top"):
            payload["crypto_top"] = computed["crypto_top"]
        if computed.get("equity_finalists"):
            payload["equity_finalists"] = computed["equity_finalists"]
        if computed.get("crypto_finalists"):
            payload["crypto_finalists"] = computed["crypto_finalists"]

    return payload
