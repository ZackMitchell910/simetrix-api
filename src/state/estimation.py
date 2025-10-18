"""Dynamic factor fusion utilities."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from src.adapters.base import FeedFrame
from src.state.contracts import StateVector


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def fuse_state(
    prior: StateVector,
    *,
    news: FeedFrame | None = None,
    sentiment_scores: Mapping[str, Any] | None = None,
    options_surface: FeedFrame | None = None,
    futures_curve: FeedFrame | None = None,
    macro_context: Mapping[str, Any] | None = None,
    social: FeedFrame | None = None,
) -> tuple[StateVector, dict[str, Any]]:
    """Fuse heterogeneous signals into an updated state vector."""

    mu_pre = float(prior.drift_annual)
    sigma_pre = float(prior.vol_annual)

    mu_signal = 0.0
    if sentiment_scores:
        mu_signal += 0.25 * float(sentiment_scores.get("last24h", 0.0))
        mu_signal += 0.15 * float(sentiment_scores.get("last7d", 0.0))
    if news:
        mu_signal += 0.05 * news.weighted_mean("sentiment_score", default=0.0)
    if social:
        mu_signal += 0.10 * social.weighted_mean("last24h", default=0.0)
    if macro_context:
        mu_signal -= 0.0005 * float(macro_context.get("rates_delta_bp", 0.0))
        mu_signal += 0.0008 * float(macro_context.get("credit_spread_bp", 0.0))
    if futures_curve:
        forward = futures_curve.weighted_mean("annualized_forward", default=mu_pre)
        mu_signal += 0.5 * (forward - mu_pre)

    sigma_signal = 0.0
    if options_surface:
        implied_vol = options_surface.weighted_mean("implied_vol", default=sigma_pre)
        sigma_signal += 0.6 * (implied_vol - sigma_pre)
        sigma_signal += 0.1 * options_surface.weighted_mean("skew", default=0.0)
    if news:
        sigma_signal += 0.15 * abs(news.weighted_mean("sentiment_score", default=0.0))
    if social:
        sigma_signal += 0.10 * abs(social.weighted_mean("last24h", default=0.0))

    mu_post = _clamp(mu_pre + mu_signal, -1.8, 1.8)
    sigma_post = _clamp(sigma_pre + sigma_signal, 0.01, 5.0)

    jump_intensity = _clamp(prior.jump_intensity + max(0.0, sigma_signal) * 0.2, 0.0, 5.0)
    jump_mean = prior.jump_mean + 0.1 * mu_signal
    jump_vol = _clamp(prior.jump_vol + 0.05 * abs(sigma_signal), 0.0, 3.0)

    fused = replace(
        prior,
        drift_annual=mu_post,
        vol_annual=sigma_post,
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_vol=jump_vol,
    )

    diagnostics = {
        "mu_pre": mu_pre,
        "mu_post": mu_post,
        "sigma_pre": sigma_pre,
        "sigma_post": sigma_post,
        "mu_signal": mu_signal,
        "sigma_signal": sigma_signal,
        "jump_intensity": jump_intensity,
    }

    return fused, diagnostics


__all__ = ["fuse_state"]
