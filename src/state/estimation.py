from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

from src.scenarios.schema import EventShock
from src.state.contracts import StateVector


def fuse_dynamic_factors(
    *,
    t: datetime,
    spot: float,
    mu_pre: float,
    sigma_pre: float,
    regime: str,
    macro: Mapping[str, Any] | None,
    sentiment: Mapping[str, Any] | None,
    earnings: Mapping[str, Any] | None,
    options_iv: float | None = None,
    futures_curve: Mapping[str, Any] | None = None,
    events: Sequence[EventShock] | None = None,
    cache_diags: Sequence[Mapping[str, Any]] | None = None,
    bounds: tuple[float, float, float, float] = (-1.5, 1.5, 1e-4, 5.0),
    apply_bias: bool = True,
) -> tuple[float, float, StateVector, dict[str, Any]]:
    """Blend prior drift/vol with exogenous signals, returning the fused state."""

    mu_min, mu_max, sigma_min, sigma_max = bounds
    mu_post = float(mu_pre)
    sigma_post = float(sigma_pre)

    sentiment = dict(sentiment or {})
    earnings = dict(earnings or {})
    macro = dict(macro or {})
    futures_curve = dict(futures_curve or {})

    sent_avg = float(sentiment.get("avg_sent_7d") or 0.0)
    sent_last24 = float(sentiment.get("last24h") or 0.0)
    earn_surprise = float(earnings.get("surprise_last") or 0.0)
    guidance_delta = float(earnings.get("guidance_delta") or 0.0)
    futures_bias = float(futures_curve.get("drift") or futures_curve.get("basis") or 0.0)

    adjustments: dict[str, float] = {}

    if apply_bias:
        mu_bias = float(max(-0.20, min(0.20, 0.15 * sent_avg + 0.05 * earn_surprise + 0.02 * guidance_delta)))
        mu_post = float(min(mu_max, max(mu_min, mu_post + mu_bias + futures_bias)))
        adjustments["mu_bias"] = mu_bias
        adjustments["futures_bias"] = futures_bias
        if sent_last24 > 0.20:
            sigma_post *= 0.97
            adjustments["sentiment_damp"] = 0.97
    else:
        adjustments["mu_bias"] = 0.0
        adjustments["futures_bias"] = futures_bias if futures_bias else 0.0

    if options_iv is not None:
        sigma_post = max(sigma_post, float(options_iv))
        adjustments["options_iv_floor"] = float(options_iv)

    sigma_post = float(min(sigma_max, max(sigma_min, sigma_post)))
    mu_post = float(min(mu_max, max(mu_min, mu_post)))

    state = StateVector(
        t=t,
        spot=float(spot),
        drift_annual=mu_post,
        vol_annual=sigma_post,
        jump_intensity=0.0,
        jump_mean=0.0,
        jump_vol=0.0,
        regime=str(regime or "neutral"),
        macro=macro,
        sentiment=sentiment,
        events=list(events or []),
        cross={
            "mu_pre": float(mu_pre),
            "sigma_pre": float(sigma_pre),
            "sentiment_bias": adjustments.get("mu_bias", 0.0),
            "futures_bias": adjustments.get("futures_bias", 0.0),
            **({"options_iv": float(options_iv)} if options_iv is not None else {}),
        },
        provenance={
            "estimator": "state.estimation.fuse_dynamic_factors",
            "apply_bias": bool(apply_bias),
            "bounds": {
                "mu": [mu_min, mu_max],
                "sigma": [sigma_min, sigma_max],
            },
        },
    )

    if cache_diags:
        cache_list = [dict(item) for item in cache_diags]
        state.provenance["cache"] = cache_list
    else:
        cache_list = []

    diag: dict[str, Any] = {
        "use_sim_bias": bool(apply_bias),
        "mu_ann_pre": float(mu_pre),
        "mu_ann_post": mu_post,
        "sigma_ann_pre": float(sigma_pre),
        "sigma_ann_post": sigma_post,
        "adjustments": adjustments,
        "context": state.as_dict(),
    }
    if cache_list:
        diag["cache_stats"] = cache_list

    return mu_post, sigma_post, state, diag


__all__ = ["fuse_dynamic_factors"]
