from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

LONG_HORIZON_DAYS = 365
LONG_HORIZON_SHRINK_FLOOR = 0.25  # never shrink below 25%


def horizon_shrink(h_days: int) -> float:
    """Shrink drift as horizon increases; floored to avoid zeroing out."""
    if h_days <= LONG_HORIZON_DAYS:
        return 1.0
    s = (LONG_HORIZON_DAYS / float(max(1, min(h_days, 3650)))) ** 0.5
    return max(LONG_HORIZON_SHRINK_FLOOR, min(1.0, s))


def winsorize(arr: np.ndarray, p_lo: float = 0.005, p_hi: float = 0.995) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo, hi = np.quantile(arr, [p_lo, p_hi])
    return np.clip(arr, lo, hi)


def ewma_sigma(returns: np.ndarray, lam: float = 0.94) -> float:
    """EWMA of bar returns; returns per-bar sigma (annualize outside)."""
    if returns.size == 0:
        return 0.2
    var = 0.0
    for r in returns[::-1]:
        var = lam * var + (1 - lam) * (float(r) * float(r))
    return math.sqrt(max(var, 1e-12))


def rel_levels_from_expected_move(expected_move: float, *, kind: str = "equity") -> list[float]:
    em = float(max(0.005, expected_move))
    if kind == "crypto":
        multipliers = [-3.5, -2.5, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.5, 3.5]
    else:
        multipliers = [-2.5, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.5]
    levels = [float(round(m * em, 6)) for m in multipliers]
    if 0.0 not in levels:
        levels.insert(len(levels) // 2, 0.0)
    return levels


def auto_rel_levels(
    sigma_ann: Optional[float],
    horizon_days: int,
    *,
    kind: str = "equity",
) -> Tuple[float, ...]:
    """Volatility-aware ladder (percent moves from S0)."""
    if sigma_ann is None or not math.isfinite(sigma_ann):
        sigma_ann = 0.30 if kind == "equity" else 0.80

    scale = float(np.clip(sigma_ann, 0.05, 1.50)) * math.sqrt(max(int(horizon_days), 1) / 252.0)
    base = max(0.05, min(0.40, 1.5 * scale))
    rungs = (-2 * base, -base, 0.0, base, 2 * base)
    return tuple(float(x) for x in rungs)


def _student_t_noise(df: int, size: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    if df <= 0:
        raise ValueError("df must be positive")
    z = rng.standard_normal(size)
    u = rng.chisquare(df, size)
    return z * np.sqrt(float(df) / np.maximum(u, 1e-9))


def simulate_gbm_student_t(
    S0: float,
    mu_ann: float,
    sigma_ann: float,
    horizon_days: int,
    n_paths: int,
    df_t: int = 7,
    antithetic: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """Return price paths shape (n_paths, H+1) including S0."""
    H = int(horizon_days)
    rng = np.random.default_rng(seed)
    mu_d = mu_ann / 252.0
    sig_d = sigma_ann / math.sqrt(252.0)
    half = n_paths // 2 if antithetic else n_paths
    noises = _student_t_noise(df_t, size=(half, H), rng=rng)
    if antithetic:
        noises = np.vstack([noises, -noises])
        if noises.shape[0] < n_paths:
            extra = _student_t_noise(df_t, size=(1, H), rng=rng)
            noises = np.vstack([noises, extra])
        noises = noises[:n_paths, :]
    dlogS = (mu_d - 0.5 * sig_d * sig_d) + sig_d * noises
    log_paths = np.cumsum(dlogS, axis=1)
    paths = S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))
    return paths


__all__ = [
    "LONG_HORIZON_DAYS",
    "LONG_HORIZON_SHRINK_FLOOR",
    "auto_rel_levels",
    "ewma_sigma",
    "horizon_shrink",
    "rel_levels_from_expected_move",
    "simulate_gbm_student_t",
    "winsorize",
]
