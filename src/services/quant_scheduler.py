from __future__ import annotations

import asyncio
import logging
import os
import random
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from redis.asyncio import Redis

from src.core import REDIS, settings
from src.observability import log_json
from src.services.context import get_macro_features
from src.services.ingestion import fetch_cached_hist_prices
from src.services.quant_candidates import (
    llm_shortlist,
    load_precomputed_quant,
    normalize_precomputed_result,
    quick_score,
    rank_candidates,
    write_quant_results,
)
from src.services.quant_context import (
    build_context_loader,
    detect_regime,
    mu_bias_from_context,
    safe_earnings,
    safe_sentiments,
)
from src.services.quant_mc import (
    combine_mc_results,
    enrich_results_list as mc_enrich_results_list,
    mc_summary,
    run_mc_batch,
    select_winner,
)
from src.services.training import _feat_from_prices

logger = logging.getLogger("simetrix.services.quant_scheduler")

EQUITY_WATCH: Sequence[str] = []
CRYPTO_WATCH: Sequence[str] = []


def _parse_hhmm(value: str) -> tuple[int, int]:
    try:
        hh, mm = value.strip().split(":")
        h = max(0, min(23, int(hh)))
        m = max(0, min(59, int(mm)))
        return h, m
    except Exception:
        return (8, 45)


try:
    DAILY_QUANT_TZ = ZoneInfo(settings.daily_quant_timezone)
except Exception:
    DAILY_QUANT_TZ = timezone.utc
    logger.warning(
        "Falling back to UTC for daily quant scheduler (invalid timezone: %s)",
        settings.daily_quant_timezone,
    )


DAILY_QUANT_HOUR, DAILY_QUANT_MINUTE = _parse_hhmm(settings.daily_quant_hhmm)
_LAST_SCHEDULED_QUANT: str | None = None
_HEALTH_QUANT_LOCK = asyncio.Lock()
_HEALTH_QUANT_TASK: asyncio.Task | None = None


def quant_budget_key() -> str:
    return f"quant:budget:{datetime.now(timezone.utc).date().isoformat()}"


async def quant_allow(redis: Redis, max_calls_per_day: int = 1) -> bool:
    try:
        k = quant_budget_key()
        calls = await redis.get(k)
        n = int(calls or "0")
        return n < max_calls_per_day
    except Exception:
        return False


async def quant_consume(redis: Redis) -> None:
    try:
        k = quant_budget_key()
        pipe = await redis.pipeline()
        await pipe.incr(k)
        await pipe.expire(k, 3 * 24 * 3600)
        await pipe.execute()
    except Exception:
        pass


async def _quick_score(symbol: str) -> dict[str, Any]:
    return await quick_score(
        symbol,
        fetch_hist=fetch_cached_hist_prices,
        compute_features=_feat_from_prices,
        detect_regime=detect_regime,
        redis=REDIS,
    )


async def _rank_candidates(symbols: Sequence[str], top_k: int = 8) -> list[dict[str, Any]]:
    return await rank_candidates(symbols, top_k, quick_score_func=_quick_score)


async def _mc_batch(
    cands: list[dict[str, Any]],
    horizon: int | None = None,
    paths: int = 8000,
) -> list[dict[str, Any]]:
    return await run_mc_batch(
        cands,
        horizon=horizon,
        paths=paths,
        limit=3,
        redis=REDIS,
        quant_allow=quant_allow,
        quant_consume=quant_consume,
    )


async def run_daily_quant(
    horizon_days: int | None = None,
    *,
    legacy_scopes: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    today_str = datetime.now(timezone.utc).date().isoformat()
    horizon = int(horizon_days or settings.daily_quant_horizon_days)
    seed_k = int(os.getenv("PT_DAILY_SEED_K", "6"))
    rank_k = int(os.getenv("PT_DAILY_RANK_K", "3"))
    mc_conc = int(os.getenv("PT_MC_CONC", "4"))

    top_n = max(1, int(os.getenv("PT_TOP_N", "1")))
    expose_finalists = os.getenv("PT_EXPOSE_FINALISTS", "0") == "1"

    eq_watch = getattr(settings, "equity_watch", None) or list(EQUITY_WATCH)
    cr_watch = getattr(settings, "crypto_watch", None) or list(CRYPTO_WATCH)

    if not eq_watch or not cr_watch:
        logger.warning(
            "Watchlists are empty or missing; equity=%d crypto=%d",
            len(eq_watch or []),
            len(cr_watch or []),
        )

    macro_ctx = await get_macro_features()
    context_cache: dict[str, dict[str, Any]] = {}
    load_context = await build_context_loader(context_cache)
    safe_sent = safe_sentiments
    safe_earn = safe_earnings
    mu_bias_fn = mu_bias_from_context

    async def enrich_candidate(item: dict[str, Any]) -> dict[str, Any]:
        sym = str(item.get("symbol") or "").upper()
        ctx = await load_context(sym)
        sent = ctx.get("sentiment") or safe_sent()
        earn = ctx.get("earnings") or safe_earn()
        mu_bias = mu_bias_fn(ctx)
        last24 = float(sent.get("last24h") or 0.0)
        score_raw = float(item.get("score") or 0.0)
        score_adj = float(np.clip(score_raw + 0.5 * mu_bias + 0.25 * last24, 0.0, 1.0))
        enriched = dict(item)
        enriched["score_raw"] = score_raw
        enriched["score"] = score_adj
        enriched["context"] = {
            "sentiment": sent,
            "earnings": earn,
            "macro": macro_ctx,
            "mu_bias": mu_bias,
        }
        return enriched

    async def enrich_candidate_list(cands: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        if not cands:
            return []
        tasks = [asyncio.create_task(enrich_candidate(dict(item))) for item in cands]
        enriched = [await t for t in tasks]
        enriched.sort(
            key=lambda r: (
                float(r.get("score") or 0.0),
                float(r.get("score_raw") or 0.0),
            ),
            reverse=True,
        )
        return enriched

    async def enrich_results_list(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        return await mc_enrich_results_list(
            items,
            load_context=load_context,
            macro_context=macro_ctx,
            safe_sentiments=safe_sent,
            safe_earnings=safe_earn,
            mu_bias_from_context=mu_bias_fn,
        )

    precomp_eq = await load_precomputed_quant("equity", today_str, redis=REDIS)
    precomp_cr = await load_precomputed_quant("crypto", today_str, redis=REDIS)

    eq_prefinal = [
        normalize_precomputed_result(item, horizon)
        for item in precomp_eq
        if "prob_up_end" in item
    ]
    cr_prefinal = [
        normalize_precomputed_result(item, horizon)
        for item in precomp_cr
        if "prob_up_end" in item
    ]
    eq_precandidates = [item for item in precomp_eq if "prob_up_end" not in item]
    cr_precandidates = [item for item in precomp_cr if "prob_up_end" not in item]

    if eq_precandidates:
        eq_rank: list[dict[str, Any]] = [
            {"symbol": item["symbol"], "score": float(item.get("score", 1.0)), "source": "external"}
            for item in eq_precandidates
        ]
        log_json("info", msg="quant_candidates_external", kind="equity", items=len(eq_rank))
    else:
        eq_seed = await llm_shortlist("equity", list(eq_watch), top_k=seed_k, horizon_days=horizon)
        eq_rank = await _rank_candidates(eq_seed, top_k=rank_k)

    if cr_precandidates:
        cr_rank: list[dict[str, Any]] = [
            {"symbol": item["symbol"], "score": float(item.get("score", 1.0)), "source": "external"}
            for item in cr_precandidates
        ]
        log_json("info", msg="quant_candidates_external", kind="crypto", items=len(cr_rank))
    else:
        cr_seed = await llm_shortlist("crypto", list(cr_watch), top_k=seed_k, horizon_days=horizon)
        cr_rank = await _rank_candidates(cr_seed, top_k=rank_k)

    eq_rank = await enrich_candidate_list(eq_rank)
    cr_rank = await enrich_candidate_list(cr_rank)

    try:
        if eq_rank:
            logger.info(
                "Equity prescreen finalists: %s",
                [(r.get("symbol"), round(r.get("score") or 0.0, 4)) for r in eq_rank],
            )
        if cr_rank:
            logger.info(
                "Crypto prescreen finalists: %s",
                [(r.get("symbol"), round(r.get("score") or 0.0, 4)) for r in cr_rank],
            )
    except Exception:
        pass

    async def _mc_summary(symbol: str, hz: int) -> dict[str, Any]:
        summary = await mc_summary(
            symbol,
            hz,
            legacy_scopes=legacy_scopes,
            redis=REDIS,
            quant_allow=quant_allow,
            quant_consume=quant_consume,
        )
        return summary

    async def mc_for_finalists(finalists: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        sem = asyncio.Semaphore(max(1, mc_conc))

        async def _one(item: dict[str, Any]) -> dict[str, Any]:
            sym = item.get("symbol")
            async with sem:
                try:
                    res = await _mc_summary(sym, horizon)
                    res.setdefault("symbol", sym)
                    res.setdefault("horizon_days", horizon)
                    res.setdefault("ok", True)
                    return res
                except Exception as exc:
                    logger.warning("MC failed for %s: %s", sym, exc)
                    return {"ok": False, "symbol": sym, "error": f"{type(exc).__name__}: {exc}"}

        return await asyncio.gather(*(_one(x) for x in finalists))

    eq_mc = await mc_for_finalists(eq_rank) if eq_rank else []
    cr_mc = await mc_for_finalists(cr_rank) if cr_rank else []

    eq_mc_ok = combine_mc_results(eq_prefinal, eq_mc)
    cr_mc_ok = combine_mc_results(cr_prefinal, cr_mc)

    eq_mc_ok = await enrich_results_list(eq_mc_ok)
    cr_mc_ok = await enrich_results_list(cr_mc_ok)

    eq_top = eq_mc_ok[:top_n]
    cr_top = cr_mc_ok[:top_n]
    eq_win = select_winner(eq_mc_ok)
    cr_win = select_winner(cr_mc_ok)

    payload: dict[str, Any] = {
        "as_of": today_str,
        "equity": eq_win,
        "crypto": cr_win,
        "macro": macro_ctx,
    }
    if top_n > 1:
        payload["equity_top"] = eq_top
        payload["crypto_top"] = cr_top
    if expose_finalists:
        payload["equity_finalists"] = eq_mc_ok
        payload["crypto_finalists"] = cr_mc_ok

    await write_quant_results(
        as_of=today_str,
        horizon_days=horizon,
        equity_pick=eq_win,
        crypto_pick=cr_win,
        equity_top=eq_top if top_n > 1 else None,
        crypto_top=cr_top if top_n > 1 else None,
        equity_finalists=eq_mc_ok if expose_finalists else None,
        crypto_finalists=cr_mc_ok if expose_finalists else None,
        top_n=top_n,
        expose_finalists=expose_finalists,
        redis=REDIS,
    )

    return payload


async def trigger_daily_quant_from_health(
    reason: str = "health",
    *,
    register_task: Callable[[asyncio.Task], None] | None = None,
    legacy_scopes: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    global _LAST_SCHEDULED_QUANT, _HEALTH_QUANT_TASK

    today_str = datetime.now(timezone.utc).date().isoformat()
    if _LAST_SCHEDULED_QUANT == today_str:
        return

    async def has_done_flag() -> bool:
        if not REDIS:
            return False
        try:
            return bool(await REDIS.exists(f"quant:daily:{today_str}:done"))
        except Exception:
            return False

    if await has_done_flag():
        _LAST_SCHEDULED_QUANT = today_str
        return

    async with _HEALTH_QUANT_LOCK:
        if _LAST_SCHEDULED_QUANT == today_str:
            return
        if _HEALTH_QUANT_TASK and not _HEALTH_QUANT_TASK.done():
            return
        if await has_done_flag():
            _LAST_SCHEDULED_QUANT = today_str
            return

        async def runner() -> None:
            global _LAST_SCHEDULED_QUANT, _HEALTH_QUANT_TASK
            try:
                payload = await run_daily_quant(
                    horizon_days=settings.daily_quant_horizon_days,
                    legacy_scopes=legacy_scopes,
                )
                as_of = str(payload.get("as_of") or today_str)
                _LAST_SCHEDULED_QUANT = as_of
                log_json("info", msg="quant_health_trigger_ok", reason=reason, as_of=as_of)
            except Exception as exc:
                log_json("error", msg="quant_health_trigger_fail", reason=reason, error=str(exc))
            finally:
                _HEALTH_QUANT_TASK = None

        task = asyncio.create_task(runner(), name="quant_health_trigger")
        _HEALTH_QUANT_TASK = task
        if register_task:
            register_task(task)
        log_json("info", msg="quant_health_trigger", reason=reason, today=today_str)


async def daily_quant_scheduler(
    *,
    legacy_scopes: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    global _LAST_SCHEDULED_QUANT
    await asyncio.sleep(random.uniform(5.0, 25.0))
    while True:
        now_utc = datetime.now(timezone.utc)
        now_local = now_utc.astimezone(DAILY_QUANT_TZ)
        today_str = now_local.date().isoformat()
        target_local = now_local.replace(
            hour=DAILY_QUANT_HOUR,
            minute=DAILY_QUANT_MINUTE,
            second=0,
            microsecond=0,
        )

        done = False
        if REDIS:
            try:
                done = bool(await REDIS.exists(f"quant:daily:{today_str}:done"))
            except Exception:
                done = False
        if _LAST_SCHEDULED_QUANT == today_str:
            done = True

        if not done and now_local >= target_local:
            try:
                payload = await run_daily_quant(
                    horizon_days=settings.daily_quant_horizon_days,
                    legacy_scopes=legacy_scopes,
                )
                as_of = payload.get("as_of") or today_str
                _LAST_SCHEDULED_QUANT = as_of
                if REDIS and as_of != today_str:
                    try:
                        await REDIS.setex(f"quant:daily:{as_of}:done", 27 * 3600, "1")
                    except Exception:
                        pass
                log_json("info", msg="quant_daily_scheduler_ok", as_of=as_of)
            except Exception as exc:
                log_json("error", msg="quant_daily_scheduler_fail", error=str(exc))
                await asyncio.sleep(300)
            else:
                await asyncio.sleep(90)
            continue

        if now_local < target_local:
            delay = (target_local - now_local).total_seconds()
        else:
            next_target = target_local + timedelta(days=1)
            delay = (next_target - now_local).total_seconds()
        await asyncio.sleep(max(60, min(delay, 3600)))


__all__ = [
    "quant_budget_key",
    "quant_allow",
    "quant_consume",
    "run_daily_quant",
    "trigger_daily_quant_from_health",
    "daily_quant_scheduler",
]
