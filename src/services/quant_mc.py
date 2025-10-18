from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, Iterable
from uuid import uuid4

import numpy as np
from redis.asyncio import Redis

from src.adapters.base import FeedFrame, FeedRecord
from src.core import REDIS, settings
from src.scenarios.schema import EventShock
from src.services.context import get_macro_features
from src.services.labeling import to_polygon_ticker
from src.services.simulation import RunState, SimRequest, run_simulation
from src.services.quant_context import (
    build_context_loader,
    mu_bias_from_context as default_mu_bias,
    safe_earnings as default_safe_earnings,
    safe_sentiments as default_safe_sentiments,
)
from src.state.contracts import StateVector
from src.state.estimation import fuse_state

try:  # pragma: no cover - optional dependency during bootstrap/tests
    from src.feature_store import connect as fs_connect, get_latest_mc_metric as fs_get_latest_mc_metric  # type: ignore
except Exception:  # pragma: no cover - feature store disabled
    fs_connect = None  # type: ignore[assignment]
    fs_get_latest_mc_metric = None  # type: ignore[assignment]

logger = logging.getLogger("simetrix.services.quant_mc")

LegacyScope = Mapping[str, Any]
LegacyCallable = Callable[[str, int], Any]

LEGACY_MC_FUNCTION_NAMES: tuple[str, ...] = (
    "_simulate_and_summarize",
    "_simulate_mc_summary",
    "_monte_carlo_summary",
    "_simulate_symbol",
    "_simulate_mc",
    "_monte_carlo",
)


def _ensure_redis(redis: Redis | None) -> Redis:
    if redis is not None:
        return redis
    if REDIS is None:
        raise RuntimeError("Redis unavailable for Monte Carlo execution")
    return REDIS


async def run_mc_for(
    symbol: str,
    horizon: int | None = None,
    paths: int = 6000,
    *,
    redis: Redis | None = None,
    quant_allow: Callable[[Redis, int], Awaitable[bool]] | None = None,
    quant_consume: Callable[[Redis], Awaitable[None]] | None = None,
    quantum_enabled: bool | None = None,
    max_quant_calls_per_day: int | None = None,
    mode: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Execute a Monte-Carlo simulation inline and return the summary artifacts.
    """
    redis_conn = _ensure_redis(redis)
    horizon_days = int(horizon or settings.daily_quant_horizon_days)
    req = SimRequest(
        symbol=to_polygon_ticker(symbol),
        horizon_days=horizon_days,
        paths=int(paths),
        mode=str(mode or "quick"),
    )
    run_id = uuid4().hex
    run_state = RunState(
        run_id=run_id,
        status="queued",
        progress=0.0,
        symbol=req.symbol,
        horizon_days=req.horizon_days,
        paths=req.paths,
        startedAt=datetime.now(timezone.utc).isoformat(),
        status_detail="Queued",
        owner="system:quant",
    )
    await redis_conn.setex(f"run:{run_id}", settings.run_ttl_seconds, run_state.model_dump_json())
    await run_simulation(run_id, req, redis_conn)

    raw_artifact = await redis_conn.get(f"artifact:{run_id}")
    artifact = json.loads(raw_artifact) if raw_artifact else {}

    base_price = None
    median_return_pct = 0.0
    try:
        median_path = artifact.get("median_path") or []
        if median_path:
            s0 = float(median_path[0][1])
            sh = float(median_path[-1][1])
            base_price = float(sh)
            median_return_pct = float((sh / s0 - 1) * 100) if s0 else 0.0
    except Exception:  # pragma: no cover - defensive
        base_price = None
        median_return_pct = 0.0

    predicted_price = None
    try:
        predicted_val = artifact.get("most_likely_price")
        if predicted_val is not None:
            predicted_price = float(predicted_val)
    except Exception:  # pragma: no cover - defensive
        predicted_price = None
    if predicted_price is None:
        predicted_price = base_price

    summary: dict[str, Any] = {
        "symbol": symbol,
        "run_id": run_id,
        "horizon_days": int(artifact.get("horizon_days", horizon_days)),
        "prob_up_end": float(artifact.get("prob_up_end", 0.5)),
        "median_return_pct": float(median_return_pct),
        "var95": (artifact.get("var_es") or {}).get("var95"),
        "es95": (artifact.get("var_es") or {}).get("es95"),
        "regime": (artifact.get("model_info") or {}).get("regime", "neutral"),
        "base_price": base_price,
        "predicted_price": predicted_price,
    }

    symbol_upper = symbol.strip().upper()
    horizon_for_state = int(summary.get("horizon_days") or horizon_days)
    model_info = artifact.get("model_info") or {}
    art_context = artifact.get("context") or {}
    macro_ctx = dict(art_context.get("macro") or {})
    sentiment_ctx = dict(art_context.get("sentiment") or {})
    events = []
    for event in art_context.get("events", []):
        try:
            events.append(EventShock.from_dict(event))
        except Exception:
            continue

    mu_prior = float(model_info.get("mu_ann") or 0.0)
    if not mu_prior and median_return_pct is not None and horizon_for_state > 0:
        try:
            mu_prior = math.log1p(float(median_return_pct) / 100.0) * (252.0 / max(horizon_for_state, 1))
        except Exception:
            mu_prior = 0.0

    sigma_prior = float(model_info.get("sigma_ann") or 0.0)
    if sigma_prior <= 0.0:
        try:
            base_val = float(summary.get("base_price") or 0.0)
            bull_val = float(summary.get("bullish_price") or 0.0)
            if base_val > 0 and bull_val > 0 and horizon_for_state > 0:
                pct95 = bull_val / base_val - 1.0
                sigma_prior = abs(pct95) * math.sqrt(252.0 / max(horizon_for_state, 1)) / 1.65
        except Exception:
            sigma_prior = 0.0
    sigma_prior = float(max(0.05, min(5.0, sigma_prior if sigma_prior else 0.35)))

    jump_params = artifact.get("jump_params") or {}
    state_prior = StateVector(
        t=datetime.now(timezone.utc),
        spot=float(artifact.get("spot") or summary.get("base_price") or 0.0),
        drift_annual=mu_prior,
        vol_annual=sigma_prior,
        jump_intensity=float(jump_params.get("lambda", 0.0)),
        jump_mean=float(jump_params.get("mu", 0.0)),
        jump_vol=float(jump_params.get("sigma", 0.0)),
        regime=str(summary.get("regime") or art_context.get("regime") or "neutral"),
        macro=macro_ctx,
        sentiment=sentiment_ctx,
        events=events,
        cross=dict(art_context.get("cross") or {}),
        provenance={
            "run_id": run_id,
            "source": "quant_mc",
            "artifact_version": artifact.get("version"),
        },
    )

    social_frame: FeedFrame | None = None
    if sentiment_ctx:
        social_frame = FeedFrame(
            [
                FeedRecord(
                    symbol=symbol_upper,
                    asof=datetime.now(timezone.utc),
                    source="sentiment",
                    confidence=0.5,
                    payload={
                        "last24h": sentiment_ctx.get("last24h", 0.0),
                        "last7d": sentiment_ctx.get("last7d", 0.0),
                    },
                )
            ]
        )

    fused_state, fusion_diag = fuse_state(
        state_prior,
        sentiment_scores=sentiment_ctx,
        macro_context=macro_ctx,
        social=social_frame,
    )

    summary.setdefault("diagnostics", {})
    summary["diagnostics"]["context"] = fused_state.to_dict()
    summary["diagnostics"]["fusion"] = fusion_diag

    try:
        term_prices = artifact.get("terminal_prices") or []
        if term_prices:
            terminal_arr = np.asarray(term_prices, dtype=float)
            if terminal_arr.size:
                bullish_val = float(np.percentile(terminal_arr, 95))
                summary["bullish_price"] = bullish_val
                summary["p95"] = bullish_val
                summary.setdefault("base_price", float(np.median(terminal_arr)))
                if summary.get("predicted_price") is None:
                    summary["predicted_price"] = summary.get("base_price")
    except Exception:  # pragma: no cover - defensive
        summary.setdefault("bullish_price", None)
        if summary.get("predicted_price") is None:
            summary["predicted_price"] = summary.get("base_price")
        summary.setdefault("p95", summary.get("bullish_price"))

    try:
        quantum_flag = quantum_enabled
        if quantum_flag is None:
            quantum_flag = os.getenv("PT_USE_QUANTUM", "0").strip() == "1"
        max_calls = int(max_quant_calls_per_day or int(os.getenv("PT_QUANT_MAX_CALLS", "1")))
        if (
            quantum_flag
            and quant_allow is not None
            and quant_consume is not None
            and await quant_allow(redis_conn, max_calls)
        ):
            from src.quantum_engine import prob_up_from_terminal

            terminal_prices = artifact.get("terminal_prices") or []
            s0 = float(artifact.get("spot") or 0.0)
            if terminal_prices and s0 > 0:
                q_prob = prob_up_from_terminal(terminal_prices, s0, bins=int(os.getenv("PT_QUANT_BINS", "64")))
                if 0.0 <= q_prob <= 1.0:
                    summary["prob_up_end_q"] = float(q_prob)
                    if os.getenv("PT_QUANT_OVERRIDE", "1") == "1":
                        summary["prob_up_end"] = float(q_prob)
            await quant_consume(redis_conn)
    except Exception as exc:  # pragma: no cover - telemetry only
        logger.info("Quantum refinement skipped: %s", exc)

    return symbol, summary


async def run_mc_batch(
    candidates: Sequence[Mapping[str, Any]],
    horizon: int | None = None,
    paths: int = 8000,
    *,
    limit: int | None = None,
    redis: Redis | None = None,
    quant_allow: Callable[[Redis, int], Awaitable[bool]] | None = None,
    quant_consume: Callable[[Redis], Awaitable[None]] | None = None,
    mode: str | None = None,
) -> list[dict[str, Any]]:
    picks = list(candidates)[: limit or 3]
    if not picks:
        return []
    hz = int(horizon or settings.daily_quant_horizon_days)
    tasks = [
        asyncio.create_task(
            run_mc_for(
                str(item.get("symbol")),
                hz,
                paths,
                redis=redis,
                quant_allow=quant_allow,
                quant_consume=quant_consume,
                mode=mode,
            )
        )
        for item in picks
        if item.get("symbol")
    ]
    out: list[dict[str, Any]] = []
    for coro in asyncio.as_completed(tasks):
        _sym, summary = await coro
        out.append(summary)
    out.sort(key=lambda s: (float(s.get("prob_up_end") or 0.0), float(s.get("median_return_pct") or 0.0)), reverse=True)
    return out


def combine_mc_results(prefinal: Iterable[Mapping[str, Any]], computed: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    combined: list[dict[str, Any]] = []
    for row in prefinal:
        sym = str(row.get("symbol") or "").upper()
        if sym and sym not in seen:
            combined.append(dict(row))
            seen.add(sym)
    for row in computed:
        if not isinstance(row, Mapping) or not row.get("ok", True):
            continue
        sym = str(row.get("symbol") or "").upper()
        if sym and sym not in seen:
            combined.append(dict(row))
            seen.add(sym)
    return combined


def select_winner(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    return max(
        (dict(item) for item in results if isinstance(item, Mapping)),
        key=lambda r: (
            float(r.get("prob_up_end") or 0.0),
            float(r.get("median_return_pct") or 0.0),
        ),
    )


async def enrich_results_list(
    items: Sequence[Mapping[str, Any]],
    *,
    load_context: Callable[[str], Awaitable[dict[str, Any]]] | None = None,
    macro_context: Mapping[str, Any] | None = None,
    safe_sentiments: Callable[[], Mapping[str, Any]] | None = None,
    safe_earnings: Callable[[], Mapping[str, Any]] | None = None,
    mu_bias_from_context: Callable[[dict[str, Any]], float] | None = None,
) -> list[dict[str, Any]]:
    if not items:
        return []

    safe_sentiments = safe_sentiments or default_safe_sentiments
    safe_earnings = safe_earnings or default_safe_earnings
    mu_bias = mu_bias_from_context or default_mu_bias
    macro_ctx = dict(macro_context) if macro_context is not None else await get_macro_features()

    if load_context is None:
        context_cache: dict[str, dict[str, Any]] = {}
        load_context = await build_context_loader(
            context_cache,
            default_sentiments=safe_sentiments,
            default_earnings=safe_earnings,
        )

    enriched: list[dict[str, Any]] = []
    for item in items:
        sym = str(item.get("symbol") or "").upper()
        ctx = await load_context(sym)
        mu_val = float(mu_bias(ctx))
        sent = ctx.get("sentiment") or safe_sentiments()
        result = dict(item)
        result.setdefault("context", {})
        result["context"].update(
            {
                "sentiment": sent,
                "earnings": ctx.get("earnings") or safe_earnings(),
                "macro": macro_ctx,
                "mu_bias": mu_val,
            }
        )
        if "prob_up_end" in result and result.get("prob_up_end") is not None:
            prob_raw = float(result.get("prob_up_end") or 0.5)
            prob_adj = float(
                np.clip(
                    prob_raw + 0.35 * mu_val + 0.10 * float((sent or {}).get("last24h") or 0.0),
                    0.0,
                    1.0,
                )
            )
            result["prob_up_end_raw"] = prob_raw
            result["prob_up_end"] = prob_adj
            result["score_ctx"] = prob_adj
        else:
            result["score_ctx"] = float(result.get("score") or 0.0)
        if "median_return_pct" in result and result.get("median_return_pct") is not None:
            med_raw = float(result.get("median_return_pct") or 0.0)
            med_adj = float(med_raw + mu_val * 100.0 * 0.2)
            result["median_return_pct_raw"] = med_raw
            result["median_return_pct"] = med_adj
        enriched.append(result)

    enriched.sort(
        key=lambda r: (
            float(r.get("score_ctx") or r.get("prob_up_end") or 0.0),
            float(r.get("median_return_pct") or 0.0),
        ),
        reverse=True,
    )
    return enriched


def latest_mc_metric(symbol: str, horizon_days: int) -> dict[str, Any] | None:
    if not fs_connect or not fs_get_latest_mc_metric:
        return None
    con = fs_connect()
    try:
        return fs_get_latest_mc_metric(con, symbol.upper(), int(horizon_days))
    except Exception as exc:  # pragma: no cover - telemetry only
        logger.debug("get_latest_mc_metric failed for %s/%s: %s", symbol, horizon_days, exc)
        return None
    finally:
        con.close()


async def mc_summary(
    symbol: str,
    horizon_days: int,
    *,
    legacy_scopes: Sequence[LegacyScope] | None = None,
    redis: Redis | None = None,
    quant_allow: Callable[[Redis, int], Awaitable[bool]] | None = None,
    quant_consume: Callable[[Redis], Awaitable[None]] | None = None,
    run_mc: Callable[[str, int], Awaitable[tuple[str, dict[str, Any]]]] | None = None,
) -> dict[str, Any]:
    scopes = legacy_scopes or ()
    for scope in scopes:
        for name in LEGACY_MC_FUNCTION_NAMES:
            fn = scope.get(name)
            if not fn:
                continue
            try:
                if inspect.iscoroutinefunction(fn):
                    res = await fn(symbol, horizon_days)
                else:
                    res = await asyncio.to_thread(fn, symbol, horizon_days)
            except Exception:  # pragma: no cover - legacy helper failure
                continue

            if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], Mapping):
                summary = dict(res[1])
            elif isinstance(res, Mapping):
                summary = dict(res)
            else:
                continue

            summary.setdefault("symbol", symbol)
            summary.setdefault("horizon_days", horizon_days)
            summary.setdefault("ok", True)
            return summary

    execute = run_mc or (
        lambda sym, hz: run_mc_for(
            sym,
            hz,
            redis=redis,
            quant_allow=quant_allow,
            quant_consume=quant_consume,
        )
    )
    _sym, summary = await execute(symbol, horizon_days)
    result = dict(summary)
    result.setdefault("symbol", symbol)
    result.setdefault("horizon_days", horizon_days)
    result.setdefault("ok", True)
    return result


__all__ = [
    "run_mc_for",
    "run_mc_batch",
    "combine_mc_results",
    "select_winner",
    "enrich_results_list",
    "latest_mc_metric",
    "mc_summary",
    "LEGACY_MC_FUNCTION_NAMES",
]
