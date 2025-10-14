import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sim_validation import rollforward_validation


def _gbm_series(mu: float, sigma: float, n_steps: int, s0: float = 100.0, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    prices = [float(s0)]
    for _ in range(n_steps):
        ret = rng.normal(mu / 252.0, sigma / math.sqrt(252.0))
        prices.append(prices[-1] * math.exp(ret))
    return prices


def test_rollforward_validation_returns_reasonable_metrics():
    prices = _gbm_series(mu=0.10, sigma=0.25, n_steps=500, seed=42)
    result = rollforward_validation(prices, horizon_days=15, n_paths=2000, target_mape=5.0)

    assert result.best.samples > 20
    assert len(result.results) >= 1
    assert 0.0 < result.best.mape < 15.0
    assert 0.0 < result.best.mdape < 12.0
    assert 500 <= result.recommended_n_paths <= 2000


def test_rollforward_validation_scales_paths_when_error_high():
    prices = _gbm_series(mu=0.0, sigma=0.60, n_steps=400, seed=123)
    result = rollforward_validation(prices, horizon_days=30, n_paths=1000, target_mape=2.0)

    assert result.best.mdape > 5.0
    assert result.recommended_n_paths > 1000  # auto-tune should scale up path budget


def test_rollforward_validation_requires_sufficient_history():
    short_series = [100.0, 101.0, 102.0]
    with pytest.raises(ValueError):
        rollforward_validation(short_series, horizon_days=10)
