from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from redis.asyncio import Redis

from src.core import REDIS
from src.services.quant_candidates import (
    load_precomputed_quant as service_load_precomputed_quant,
    normalize_precomputed_result as service_normalize_precomputed_result,
    quick_score as service_quick_score,
    rank_candidates as service_rank_candidates,
    llm_select_and_write as service_llm_select_and_write,
)
from src.services.quant_mc import (
    run_mc_for as service_run_mc_for,
    run_mc_batch as service_run_mc_batch,
    combine_mc_results as service_combine_mc_results,
)

__all__ = [
    "load_precomputed_quant",
    "normalize_precomputed_result",
    "combine_mc_results",
    "quick_score",
    "rank_candidates",
    "run_mc_for",
    "run_mc_batch",
    "llm_select_and_write",
]


async def load_precomputed_quant(kind: str, day: str, *, redis: Redis | None = None) -> list[dict[str, Any]]:
    """Compatibility wrapper for legacy imports."""
    return await service_load_precomputed_quant(kind, day, redis=redis or REDIS)


def normalize_precomputed_result(item: Mapping[str, Any], horizon: int) -> dict[str, Any]:
    """Compatibility wrapper for legacy imports."""
    return service_normalize_precomputed_result(item, horizon)


def combine_mc_results(prefinal: Sequence[Mapping[str, Any]], computed: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compatibility wrapper for legacy imports."""
    return service_combine_mc_results(prefinal, computed)


async def quick_score(
    symbol: str,
    *,
    fetch_hist: Callable[[str, int, Redis | None], Any],
    compute_features: Callable[[str, Sequence[float]], Any],
    detect_regime: Callable[[Any], Any],
    redis: Redis | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper for legacy imports."""
    return await service_quick_score(
        symbol,
        fetch_hist=fetch_hist,
        compute_features=compute_features,
        detect_regime=detect_regime,
        redis=redis or REDIS,
    )


async def rank_candidates(
    symbols: Sequence[str],
    top_k: int = 8,
    *,
    quick_score_func: Callable[[str], Any],
) -> list[dict[str, Any]]:
    """Compatibility wrapper for legacy imports."""
    return await service_rank_candidates(symbols, top_k, quick_score_func=quick_score_func)


async def run_mc_for(
    symbol: str,
    horizon: int | None = None,
    paths: int = 6000,
    *,
    redis: Redis | None = None,
    quant_allow: Callable[[Redis, int], Any] | None = None,
    quant_consume: Callable[[Redis], Any] | None = None,
    mode: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Compatibility wrapper for legacy imports."""
    return await service_run_mc_for(
        symbol,
        horizon=horizon,
        paths=paths,
        redis=redis or REDIS,
        quant_allow=quant_allow,
        quant_consume=quant_consume,
        mode=mode,
    )


async def run_mc_batch(
    candidates: Sequence[Mapping[str, Any]],
    horizon: int | None = None,
    paths: int = 8000,
    *,
    limit: int | None = None,
    redis: Redis | None = None,
    quant_allow: Callable[[Redis, int], Any] | None = None,
    quant_consume: Callable[[Redis], Any] | None = None,
    mode: str | None = None,
) -> list[dict[str, Any]]:
    """Compatibility wrapper for legacy imports."""
    return await service_run_mc_batch(
        candidates,
        horizon=horizon,
        paths=paths,
        limit=limit,
        redis=redis or REDIS,
        quant_allow=quant_allow,
        quant_consume=quant_consume,
        mode=mode,
    )


async def llm_select_and_write(kind: str, summaries: list[dict[str, Any]]) -> tuple[str, str]:
    """Compatibility wrapper for legacy imports."""
    return await service_llm_select_and_write(kind, summaries)
