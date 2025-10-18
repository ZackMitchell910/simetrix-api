from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import httpx
import numpy as np
from redis.asyncio import Redis

from src.core import REDIS
from src.observability import log_json
from src.services.inference import get_ensemble_prob_light
from src.services.labeling import to_polygon_ticker
from src.services.quant_daily import prefill_minimal_summary

logger = logging.getLogger("simetrix.services.quant_candidates")


def _llm_auto_enabled() -> bool:
    val = os.getenv("PT_LLM_AUTO") or os.getenv("PT_LLM_BACKGROUND") or "0"
    return val.strip().lower() in {"1", "true", "yes", "on"}


async def load_precomputed_quant(kind: str, day: str, *, redis: Redis | None = None) -> list[dict[str, Any]]:
    """Fetch cached Monte-Carlo output or pre-ranked candidates from Redis."""
    redis = redis or REDIS
    if not redis:
        return []
    key = f"quant:calc:{day}:{kind}"
    try:
        raw = await redis.get(key)
        if not raw:
            return []
        data = json.loads(raw)
        if isinstance(data, list):
            norm = [item for item in data if isinstance(item, dict) and item.get("symbol")]
            if norm:
                log_json("info", msg="quant_preload", kind=kind, items=len(norm))
            return norm
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to load precomputed quant data for %s: %s", key, exc)
    return []


def normalize_precomputed_result(item: Mapping[str, Any], horizon: int) -> dict[str, Any]:
    """Ensure cached MC output carries the expected structure."""
    out = dict(item)
    out.setdefault("horizon_days", horizon)
    out.setdefault("ok", True)
    return out


async def quick_score(
    symbol: str,
    *,
    fetch_hist: Callable[[str, int, Redis | None], Awaitable[Sequence[float]]],
    compute_features: Callable[[str, Sequence[float]], Awaitable[Mapping[str, float]]],
    detect_regime: Callable[[np.ndarray], Mapping[str, Any]],
    redis: Redis | None,
) -> dict[str, Any]:
    """Fast prescreen: small history fetch + simple features + light tilt."""
    try:
        sym_fetch = to_polygon_ticker(symbol)
        prices = await fetch_hist(sym_fetch, 180, redis)
        if not prices or len(prices) < 30:
            return {"symbol": symbol, "ok": False}
        feats = await compute_features(symbol, prices)
        regime = detect_regime(np.asarray(prices, dtype=float))
        try:
            p_lin = await asyncio.wait_for(get_ensemble_prob_light(symbol, redis, 1), timeout=0.6)
        except Exception:
            p_lin = 0.5
        mom = float(feats.get("mom_20", 0.0))
        mom_n = float(np.tanh(mom / 0.10))
        score = float(
            np.clip(
                0.55 * p_lin
                + 0.30 * (0.5 * (mom_n + 1))
                + 0.15 * (0.5 * (regime.get("score", 0) + 1)),
                0,
                1,
            )
        )
        return {
            "symbol": symbol,
            "ok": True,
            "p_quick": p_lin,
            "mom_20": mom,
            "regime": regime.get("name", "neutral"),
            "score": score,
        }
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.info("quick_score failed for %s: %s", symbol, exc)
        return {"symbol": symbol, "ok": False}


async def rank_candidates(
    symbols: Sequence[str],
    top_k: int = 8,
    *,
    quick_score_func: Callable[[str], Awaitable[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    max_tasks = int(os.getenv("PT_QS_TASKS", "6"))
    concurrency = int(os.getenv("PT_POLY_CONC", "1"))
    jitter_ms = int(os.getenv("PT_QS_JITTER_MS", "350"))

    seed = os.getenv("PT_DAILY_SEED") or datetime.utcnow().strftime("%Y%m%d")
    rng = random.Random(f"{seed}-rank")
    pool = list(symbols)
    rng.shuffle(pool)
    todo = pool[: max(1, max_tasks)]

    sem = asyncio.Semaphore(max(1, concurrency))

    async def guarded(sym: str) -> Mapping[str, Any]:
        async with sem:
            if jitter_ms:
                await asyncio.sleep(rng.random() * (jitter_ms / 1000.0))
            try:
                return await quick_score_func(sym)
            except Exception as exc:  # pragma: no cover - defensive logging
                return {"ok": False, "symbol": sym, "error": f"{type(exc).__name__}: {exc}"}

    results = await asyncio.gather(*(guarded(s) for s in todo))
    good = [r for r in results if r.get("ok")]

    def sort_key(entry: Mapping[str, Any]) -> tuple[float, float]:
        return (float(entry.get("score") or 0.0), rng.random())

    good.sort(key=sort_key, reverse=True)

    try:
        logger.info(
            "Prescreen finalists: %s",
            [(g.get("symbol"), round(float(g.get("score") or 0.0), 4)) for g in good[:top_k]],
        )
    except Exception:
        pass

    return [dict(item) for item in good[:top_k]]


async def llm_shortlist(
    kind: str,
    symbols: Sequence[str],
    top_k: int = 20,
    *,
    horizon_days: int | None = None,
) -> list[str]:
    key = os.getenv("XAI_API_KEY", "").strip()
    base = [s.upper() for s in symbols]
    if not base:
        return []
    top_k = max(1, min(int(top_k), len(base)))

    if not key:
        return base[:top_k]

    horizon_phrase = (
        f"over the next {int(horizon_days)} days"
        if isinstance(horizon_days, (int, float)) and horizon_days > 0
        else "in the near term"
    )

    prompt = {
        "role": "user",
        "content": (
            f"Given this watchlist, pick the TOP symbols most likely to outperform {horizon_phrase}. "
            "Consider catalysts, liquidity, momentum, and risk. Return JSON {list:[...]} with up to "
            f"{top_k} tickers only. Watchlist ({kind}): {', '.join(base)}"
        ),
    }

    attempts = max(1, int(os.getenv("PT_LLM_RETRY_ATTEMPTS", "2")))
    backoff = float(os.getenv("PT_LLM_RETRY_BACKOFF", "1.5"))

    for attempt in range(1, attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as cli:
                response = await cli.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                    "model": os.getenv("XAI_MODEL", "grok-4-latest"),
                        "messages": [prompt],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.1,
                    },
                )
                response.raise_for_status()
                body = response.json()
                content = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
                parsed = json.loads(content)
                picks = parsed.get("list") or []
                log_json(
                    "info",
                    msg="llm_shortlist_result",
                    provider="xai",
                    kind=kind,
                    attempt=attempt,
                    requested=top_k,
                    returned=len(picks),
                    items=picks,
                )
                selected: list[str] = []
                seen: set[str] = set()
                allowed = set(base)
                for item in picks:
                    sym = str(item).upper().strip()
                    if sym in allowed and sym not in seen:
                        selected.append(sym)
                        seen.add(sym)
                    if len(selected) >= top_k:
                        break
                if len(selected) < top_k:
                    for sym in base:
                        if sym not in seen:
                            selected.append(sym)
                            seen.add(sym)
                        if len(selected) >= top_k:
                            break
                if selected:
                    return selected[:top_k]
                raise ValueError("empty shortlist after filtering")
        except Exception as exc:
            logger.warning("llm_shortlist attempt %s/%s failed: %s", attempt, attempts, exc)
            if attempt < attempts:
                await asyncio.sleep(backoff * attempt)
                continue
            log_json("warning", msg="llm_shortlist_fallback", kind=kind, attempts=attempts, error=str(exc))
            return base[:top_k]

    return base[:top_k]


async def llm_select_and_write(kind: str, summaries: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    """
    Return (symbol, blurb). Uses xAI if LLM_PROVIDER=xai or OpenAI missing.
    Falls back to a deterministic heuristic if no key / parse failure.
    """
    summaries = list(summaries)
    if not summaries:
        return "", ""

    best = max(
        summaries,
        key=lambda s: (float(s.get("prob_up_end") or 0.0), float(s.get("median_return_pct") or 0.0)),
    )
    default_blurb = (
        f"{best['symbol']}: {best.get('horizon_days', 30)}d MC P(up)≈{best.get('prob_up_end', 0):.2%}, "
        f"median ≈ {best.get('median_return_pct', 0):.1f}% • regime {best.get('regime', 'neutral')}."
    )

    if not _llm_auto_enabled():
        return best.get("symbol", ""), default_blurb

    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    oai_key = os.getenv("OPENAI_API_KEY", "").strip()
    xai_key = os.getenv("XAI_API_KEY", "").strip()
    prefer_xai = provider == "xai" or (not oai_key and bool(xai_key))

    async def call_xai() -> str:
        model = os.getenv("XAI_MODEL", "grok-4-latest")
        async with httpx.AsyncClient(timeout=20.0) as cli:
            resp = await cli.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {xai_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "Be concise, factual, compliance-safe."},
                        {
                            "role": "user",
                            "content": (
                                "You are a risk-aware quantitative analyst. Given JSON candidate summaries, "
                                "pick ONE ticker with the best risk-adjusted outlook for the stated horizon. "
                                "Respond as JSON with keys {symbol, blurb}. The blurb must be ≤280 chars, "
                                "objective, no guarantees, and mention 1-2 drivers.\n\n"
                                f"Candidates ({kind}):\n{json.dumps(summaries, ensure_ascii=False)}"
                            ),
                        },
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "QuantPick",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "symbol": {"type": "string"},
                                    "blurb": {"type": "string", "maxLength": 280},
                                },
                                "required": ["symbol", "blurb"],
                                "additionalProperties": False,
                            },
                            "strict": True,
                        },
                    },
                    "temperature": 0.2,
                },
            )
            resp.raise_for_status()
            return (resp.json().get("choices") or [{}])[0].get("message", {}).get("content", "") or ""

    async def call_openai() -> str:
        async with httpx.AsyncClient(timeout=20.0) as cli:
            resp = await cli.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {oai_key}"},
                json={
                    "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    "messages": [
                        {"role": "system", "content": "Be concise, factual, compliance-safe."},
                        {
                            "role": "user",
                            "content": (
                                "You are a risk-aware quantitative analyst. Given JSON candidate summaries, "
                                "pick ONE ticker with the best risk-adjusted outlook for the stated horizon. "
                                "Respond as JSON with keys {symbol, blurb}. The blurb must be ≤280 chars, "
                                "objective, no guarantees, and mention 1-2 drivers.\n\n"
                                f"Candidates ({kind}):\n{json.dumps(summaries, ensure_ascii=False)}"
                            ),
                        },
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2,
                },
            )
            resp.raise_for_status()
            return (resp.json().get("choices") or [{}])[0].get("message", {}).get("content", "") or ""

    try:
        payload = ""
        if prefer_xai and xai_key:
            payload = await call_xai()
        elif oai_key:
            payload = await call_openai()
        elif xai_key:
            payload = await call_xai()
        else:
            return best["symbol"], default_blurb

        doc = json.loads(payload.strip())
        sym = (doc.get("symbol") or best["symbol"]).upper().strip()
        blurb = (doc.get("blurb") or default_blurb).strip()
        return sym, blurb
    except Exception as exc:  # pragma: no cover - fallback safety
        logger.info("LLM pick failed; using heuristic: %s", exc)
        return best["symbol"], default_blurb


async def write_quant_results(
    *,
    as_of: str,
    horizon_days: int,
    equity_pick: Mapping[str, Any] | None,
    crypto_pick: Mapping[str, Any] | None,
    equity_top: Sequence[Mapping[str, Any]] | None = None,
    crypto_top: Sequence[Mapping[str, Any]] | None = None,
    equity_finalists: Sequence[Mapping[str, Any]] | None = None,
    crypto_finalists: Sequence[Mapping[str, Any]] | None = None,
    top_n: int = 1,
    expose_finalists: bool = False,
    redis: Redis | None = None,
    prefill_fn: Callable[[str, Mapping[str, Any] | None, str, int], Awaitable[None]] = prefill_minimal_summary,
    extra_symbols: Sequence[str] | None = None,
) -> None:
    redis = redis or REDIS
    base = f"quant:daily:{as_of}"

    equity_pick_dict = dict(equity_pick) if equity_pick else {}
    crypto_pick_dict = dict(crypto_pick) if crypto_pick else {}
    eq_top_list = [dict(item) for item in (equity_top or [])]
    cr_top_list = [dict(item) for item in (crypto_top or [])]
    eq_final_list = [dict(item) for item in (equity_finalists or [])]
    cr_final_list = [dict(item) for item in (crypto_finalists or [])]

    if redis:
        try:
            await redis.set(f"{base}:equity", json.dumps(equity_pick_dict))
            await redis.set(f"{base}:crypto", json.dumps(crypto_pick_dict))
            if top_n > 1:
                await redis.set(f"{base}:equity_top", json.dumps(eq_top_list))
                await redis.set(f"{base}:crypto_top", json.dumps(cr_top_list))
            if expose_finalists:
                await redis.set(f"{base}:equity_finalists", json.dumps(eq_final_list))
                await redis.set(f"{base}:crypto_finalists", json.dumps(cr_final_list))

            publish_pairs: list[tuple[str, Mapping[str, Any] | None]] = []
            for winner in (equity_pick_dict, crypto_pick_dict):
                if winner.get("symbol"):
                    publish_pairs.append((str(winner["symbol"]).upper(), winner))
            if top_n > 1:
                for item in eq_top_list + cr_top_list:
                    if item.get("symbol"):
                        publish_pairs.append((str(item["symbol"]).upper(), item))
            if expose_finalists:
                for item in eq_final_list + cr_final_list:
                    if item.get("symbol"):
                        publish_pairs.append((str(item["symbol"]).upper(), item))

            publish_tasks: list[asyncio.Task] = []
            publish_symbols: list[str] = []
            seen_symbols: set[str] = set()

            for sym, doc in publish_pairs:
                if not sym or sym in seen_symbols:
                    continue
                seen_symbols.add(sym)
                publish_symbols.append(sym)
                publish_tasks.append(asyncio.create_task(prefill_fn(sym, doc, as_of, horizon_days)))

            parsed_extra: list[str] = list(extra_symbols or [])
            if not parsed_extra:
                try:
                    raw_extra = os.getenv("SIMETRIX_DAILY_SYMBOLS", "")
                    if raw_extra.strip():
                        if raw_extra.strip().startswith(("[", "(")):
                            from ast import literal_eval

                            parsed_val = literal_eval(raw_extra.strip())
                        else:
                            parsed_val = [raw_extra]
                        if isinstance(parsed_val, (list, tuple)):
                            parsed_extra = [str(x).strip().upper() for x in parsed_val if str(x).strip()]
                except Exception:
                    parsed_extra = []

            for sym in parsed_extra:
                if sym and sym not in seen_symbols:
                    seen_symbols.add(sym)
                    publish_symbols.append(sym)
                    publish_tasks.append(asyncio.create_task(prefill_fn(sym, None, as_of, horizon_days)))

            if publish_tasks:
                results = await asyncio.gather(*publish_tasks, return_exceptions=True)
                for sym, outcome in zip(publish_symbols, results):
                    if isinstance(outcome, Exception):
                        logger.debug("prefill summary skipped for %s: %s", sym, outcome)

            await redis.setex(f"{base}:done", 27 * 3600, "1")
        except Exception as exc:
            logger.warning("Failed to cache daily results in Redis: %s", exc)


__all__ = [
    "load_precomputed_quant",
    "normalize_precomputed_result",
    "quick_score",
    "rank_candidates",
    "llm_shortlist",
    "llm_select_and_write",
    "write_quant_results",
]
