from __future__ import annotations

from __future__ import annotations

import math
from typing import Any, Mapping


def _to_float(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _kalman_blend(prior: float, measurement: float, prior_var: float, meas_var: float) -> tuple[float, float, float]:
    gain = prior_var / (prior_var + meas_var)
    posterior = prior + gain * (measurement - prior)
    posterior_var = (1.0 - gain) * prior_var
    return posterior, posterior_var, gain


def fuse_dynamic_factors(
    mu_prior: float,
    sigma_prior: float,
    *,
    news: Mapping[str, Any] | None = None,
    sentiment: Mapping[str, Any] | None = None,
    earnings: Mapping[str, Any] | None = None,
    macro: Mapping[str, Any] | None = None,
    options: Mapping[str, Any] | None = None,
    futures: Mapping[str, Any] | None = None,
    use_exogenous: bool = True,
    mu_bounds: tuple[float, float] = (-1.5, 1.5),
    sigma_bounds: tuple[float, float] = (1e-4, 5.0),
) -> tuple[float, float, dict[str, Any]]:
    """Blend heterogeneous signals into drift/volatility updates.

    The function applies simple Kalman-like updates for priors (options/futures) and
    linear tilts for exogenous signals (news, sentiment, macro). Diagnostics capture
    each contribution so downstream consumers can audit the adjustments.
    """

    mu_low, mu_high = mu_bounds
    sig_low, sig_high = sigma_bounds

    mu_post = float(mu_prior)
    sigma_post = float(sigma_prior)

    diagnostics: dict[str, Any] = {
        "mu_pre": float(mu_prior),
        "sigma_pre": float(sigma_prior),
        "contributors": {},
        "jumps": {"intensity": 0.02, "mean": 0.0, "vol": 0.0},
    }

    contributors = diagnostics["contributors"]

    if options:
        iv = _to_float(options.get("iv") or options.get("sigma"))
        if iv is not None and iv > 0:
            sigma_post, posterior_var, gain = _kalman_blend(sigma_post, iv, prior_var=0.10, meas_var=0.20)
            sigma_post = float(max(sig_low, min(sig_high, sigma_post)))
            contributors["options"] = {"iv": iv, "kalman_gain": float(gain), "posterior_var": float(posterior_var)}

    if futures:
        drift_hint = _to_float(futures.get("drift") or futures.get("annualized_return"))
        if drift_hint is not None:
            mu_post, _, gain = _kalman_blend(mu_post, drift_hint, prior_var=0.05, meas_var=0.12)
            contributors["futures"] = {"drift": drift_hint, "kalman_gain": float(gain)}

    sent_last24 = 0.0
    sent_avg = 0.0
    if use_exogenous and sentiment:
        sent_last24 = _to_float(sentiment.get("last24h")) or 0.0
        sent_avg = _to_float(sentiment.get("avg_sent_7d")) or 0.0
        mu_shift_raw = sent_last24 * 0.12 + sent_avg * 0.05
        mu_shift = float(max(-0.2, min(0.2, mu_shift_raw)))
        mu_post += mu_shift
        contributors["sentiment"] = {
            "mu_shift": mu_shift,
            "avg_sent_7d": sent_avg,
            "last24h": sent_last24,
        }

    if use_exogenous and earnings:
        surprise = _to_float(earnings.get("surprise_last")) or 0.0
        guidance = _to_float(earnings.get("guidance_delta")) or 0.0
        mu_shift = 0.04 * surprise + 0.02 * guidance
        mu_post += mu_shift
        contributors["earnings"] = {
            "mu_shift": mu_shift,
            "surprise_last": surprise,
            "guidance_delta": guidance,
        }

    if use_exogenous and macro:
        rff = _to_float(macro.get("rff"))
        u_rate = _to_float(macro.get("u_rate"))
        macro_tilt = 0.0
        if rff is not None:
            macro_tilt += -0.02 * max(0.0, rff - 0.04)
        if u_rate is not None:
            macro_tilt += -0.03 * max(0.0, u_rate - 0.06)
        mu_post += macro_tilt
        contributors["macro"] = {"mu_shift": macro_tilt, "rff": rff, "u_rate": u_rate}

    if news and use_exogenous:
        try:
            sentiment_signal = float(news.get("sentiment"))
        except Exception:
            sentiment_signal = 0.0
        vol_shift = 0.05 * abs(sentiment_signal)
        sigma_post += vol_shift
        contributors["news"] = {"vol_shift": vol_shift, "sentiment": sentiment_signal}

    mu_post = float(max(mu_low, min(mu_high, mu_post)))
    sigma_post = float(max(sig_low, min(sig_high, sigma_post)))

    jump_intensity = diagnostics["jumps"]["intensity"] + abs(sent_last24) * 0.05
    if news and use_exogenous:
        jump_intensity += 0.05
    diagnostics["jumps"].update(
        {
            "intensity": float(max(0.0, jump_intensity)),
            "mean": float(sent_avg * 0.02),
            "vol": float(max(0.0, 0.10 + abs(sent_last24) * 0.05)),
        }
    )

    diagnostics["mu_post"] = mu_post
    diagnostics["sigma_post"] = sigma_post
    diagnostics["use_exogenous"] = bool(use_exogenous)

    return mu_post, sigma_post, diagnostics


__all__ = ["fuse_dynamic_factors"]
