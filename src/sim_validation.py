from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np


def _as_float_array(prices: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(prices), dtype=float)
    if arr.ndim != 1:
        raise ValueError("prices must be a 1-D sequence of floats")
    if arr.size < 8:
        raise ValueError("prices history too short to validate")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError("prices must be positive finite values")
    return arr


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-12
    denom = np.abs(y_true) + eps
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def _mdape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-12
    denom = np.abs(y_true) + eps
    return float(np.median(np.abs(y_true - y_pred) / denom) * 100.0)


@dataclass(slots=True)
class RollForwardResult:
    lookback: int
    samples: int
    mape: float
    mdape: float
    mu_ann: float
    sigma_ann: float


@dataclass(slots=True)
class AutoTuneResult:
    horizon_days: int
    target_mape: float
    n_paths_input: int
    recommended_n_paths: int
    best: RollForwardResult
    results: List[RollForwardResult]


def _median_terminal_price(s0: float, mu_d: float, sigma_d: float, horizon_days: int) -> float:
    """
    Closed-form median terminal price for lognormal GBM with per-step drift mu_d and volatility sigma_d.
    """
    drift = (mu_d - 0.5 * sigma_d * sigma_d) * float(horizon_days)
    return float(s0 * math.exp(drift))


def _recommend_n_paths(current: int, mdape: float, target_mape: float) -> int:
    """
    Heuristic: increase paths when error >> target, allow mild reduction when well below target.
    """
    current = int(max(100, current))
    if not math.isfinite(mdape) or target_mape <= 0:
        return current

    ratio = mdape / target_mape
    if ratio <= 0.65:
        # comfortably below target -> allow ~30% savings but never less than 500
        return max(500, int(round(current * 0.7)))
    if ratio <= 1.05:
        # near target -> keep as-is
        return current

    # Above target: scale up but clamp to 4x current (and overall 200k)
    scale = min(4.0, max(1.2, ratio ** 1.4))
    recommended = int(round(current * scale))
    return int(min(200_000, max(current, recommended)))


def rollforward_validation(
    prices: Iterable[float],
    *,
    horizon_days: int,
    lookbacks: Sequence[int] = (20, 60, 120),
    n_paths: int = 4_000,
    target_mape: float = 5.0,
    max_samples: int = 120,
    bars_per_year: float = 252.0,
) -> AutoTuneResult:
    """
    Evaluate GBM-based forecasts on rolling windows and derive auto-tuned parameters.

    Parameters
    ----------
    prices:
        Sequence of historical closing prices (oldest->newest).
    horizon_days:
        Forecast horizon in trading days.
    lookbacks:
        Candidate lookback window sizes to evaluate (in trading days).
    n_paths:
        Current Monte Carlo path count. Used as baseline for recommendations.
    target_mape:
        Desired Mean Absolute Percentage Error (percentage, e.g., 5.0 for 5%).
    max_samples:
        Max number of roll-forward samples per lookback to keep runtime bounded.
    bars_per_year:
        Annualization factor (defaults to 252 trading days).
    """
    arr = _as_float_array(prices)
    horizon = int(max(1, horizon_days))
    samples_cap = int(max(8, max_samples))
    lookbacks = [int(l) for l in lookbacks if int(l) > 4]
    if not lookbacks:
        raise ValueError("lookbacks must contain at least one window > 4")

    results: list[RollForwardResult] = []

    for look in lookbacks:
        if look + horizon >= arr.size:
            continue

        preds: list[float] = []
        actuals: list[float] = []

        upper = arr.size - horizon
        for start in range(0, upper - look):
            window = arr[start : start + look + 1]
            if window.size <= look:
                continue
            rets = np.diff(np.log(window))
            if rets.size == 0 or not np.all(np.isfinite(rets)):
                continue

            mu_d = float(np.mean(rets))
            sigma_d = float(np.std(rets))

            s0 = float(window[-1])
            preds.append(_median_terminal_price(s0, mu_d, sigma_d, horizon))
            actuals.append(float(arr[start + look + horizon]))

            if len(preds) >= samples_cap:
                break

        if not preds:
            continue

        pred_arr = np.asarray(preds, dtype=float)
        actual_arr = np.asarray(actuals, dtype=float)
        mape = _mape(actual_arr, pred_arr)
        mdape = _mdape(actual_arr, pred_arr)

        recent = arr[-(look + 1) :]
        rets_recent = np.diff(np.log(recent))
        if rets_recent.size == 0:
            continue
        mu_ann = float(np.mean(rets_recent) * bars_per_year)
        sigma_ann = float(np.std(rets_recent) * math.sqrt(bars_per_year))

        results.append(
            RollForwardResult(
                lookback=look,
                samples=len(preds),
                mape=mape,
                mdape=mdape,
                mu_ann=mu_ann,
                sigma_ann=sigma_ann,
            )
        )

    if not results:
        raise ValueError("insufficient data for validation")

    # Best defined by lowest MDAPE; tie-breaker on smaller lookback (faster adaptation)
    best = min(results, key=lambda r: (r.mdape, r.lookback))
    recommended_paths = _recommend_n_paths(n_paths, best.mdape, target_mape)

    return AutoTuneResult(
        horizon_days=horizon,
        target_mape=float(target_mape),
        n_paths_input=int(n_paths),
        recommended_n_paths=recommended_paths,
        best=best,
        results=results,
    )
