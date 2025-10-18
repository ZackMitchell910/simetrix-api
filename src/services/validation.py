from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.core import REDIS, settings
from src.feature_store import insert_mc_metrics, upsert_mc_params, connect as feature_store_connect
from src.sim_validation import rollforward_validation
from src.services import ingestion as ingestion_service


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    return values


VALIDATION_LOOKBACKS = _parse_int_list(settings.validation_lookbacks) or [20, 60, 120]
VALIDATION_TARGET_MAPE = float(settings.validation_target_mape or 5.0)
VALIDATION_MAX_SAMPLES = int(max(20, min(settings.validation_max_samples, 500)))
VALIDATION_BARS_PER_YEAR = 252.0


async def validate_mc_paths(symbols: list[str], days: int, n_paths: int) -> dict[str, Any]:
    today = datetime.now(timezone.utc).date()
    as_of = today.isoformat()
    seeded_by = "symbol|horizon"

    results: list[dict[str, Any]] = []
    params_rows: list[tuple[str, float, float, int, int]] = []
    metric_rows: list[tuple[Any, ...]] = []

    for sym in [s.strip().upper() for s in symbols if s.strip()]:
        try:
            window_days = max(400, days + max(VALIDATION_LOOKBACKS) + 50)
            px = await ingestion_service.fetch_cached_hist_prices(sym, window_days=window_days, redis=REDIS)
            arr = np.asarray([p for p in px if isinstance(p, (int, float)) and math.isfinite(p)], float)
        except Exception as exc:
            results.append({"symbol": sym, "skipped": True, "reason": f"history_fetch_failed:{exc}"})
            continue

        if arr.size < (days + min(VALIDATION_LOOKBACKS) + 5):
            results.append({"symbol": sym, "skipped": True, "reason": "insufficient_history"})
            continue

        try:
            tune = rollforward_validation(
                arr,
                horizon_days=days,
                lookbacks=VALIDATION_LOOKBACKS,
                n_paths=n_paths,
                target_mape=VALIDATION_TARGET_MAPE,
                max_samples=VALIDATION_MAX_SAMPLES,
                bars_per_year=VALIDATION_BARS_PER_YEAR,
            )
        except ValueError as exc:
            results.append({"symbol": sym, "skipped": True, "reason": str(exc)})
            continue

        best = tune.best
        mu_daily = float(best.mu_ann / VALIDATION_BARS_PER_YEAR)
        sigma_daily = float(best.sigma_ann / math.sqrt(VALIDATION_BARS_PER_YEAR))

        params_rows.append((sym, mu_daily, sigma_daily, int(best.lookback), int(best.lookback)))

        variants = [
            {
                "lookback": r.lookback,
                "samples": r.samples,
                "mape": r.mape,
                "mdape": r.mdape,
                "mu_ann": r.mu_ann,
                "sigma_ann": r.sigma_ann,
            }
            for r in tune.results
        ]

        results.append(
            {
                "symbol": sym,
                "horizon_days": int(days),
                "mape": best.mape,
                "mdape": best.mdape,
                "n": int(best.samples),
                "mu": mu_daily,
                "sigma": sigma_daily,
                "mu_ann": best.mu_ann,
                "sigma_ann": best.sigma_ann,
                "lookback": int(best.lookback),
                "recommended_n_paths": int(tune.recommended_n_paths),
                "target_mape": VALIDATION_TARGET_MAPE,
                "candidates": variants,
            }
        )

        metric_rows.append(
            (
                as_of,
                sym,
                int(days),
                float(best.mape),
                float(best.mdape),
                int(best.samples),
                mu_daily,
                sigma_daily,
                int(tune.recommended_n_paths),
                int(best.lookback),
                int(best.lookback),
                seeded_by,
            )
        )

    con = feature_store_connect()
    try:
        for sym, mu_final, sig_final, lb_mu, lb_sig in params_rows:
            upsert_mc_params(con, sym, mu_final, sig_final, lb_mu, lb_sig)
        if metric_rows:
            insert_mc_metrics(con, metric_rows)
    finally:
        con.close()

    return {
        "as_of": as_of,
        "horizon_days": int(days),
        "n_paths": int(n_paths),
        "target_mape": VALIDATION_TARGET_MAPE,
        "lookbacks": VALIDATION_LOOKBACKS,
        "items": results,
    }


__all__ = [
    "VALIDATION_LOOKBACKS",
    "VALIDATION_TARGET_MAPE",
    "VALIDATION_MAX_SAMPLES",
    "VALIDATION_BARS_PER_YEAR",
    "validate_mc_paths",
]
