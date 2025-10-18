import os
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import numpy as np

# Ensure required env vars are present before importing the service module
os.environ.setdefault("PT_POLYGON_KEY", "test-key")
os.environ.setdefault("PT_SKIP_ONNXRUNTIME", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import predictive_api as svc
from src.services import training as training_service


class DummyRedis:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    async def get(self, key: str) -> Any:
        return self.store.get(key)

    async def set(self, key: str, value: Any, ex: int | None = None) -> None:
        self.store[key] = value

    async def setex(self, key: str, ttl: int, value: Any) -> None:
        self.store[key] = value


@pytest.mark.asyncio
async def test_ensure_trained_models_skips_when_fresh(monkeypatch):
    dummy = DummyRedis()
    key = await svc._model_key("AAPL_linear")
    dummy.store[key] = json.dumps(
        {
            "lookback_days": 200,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "profile": "quick",
        }
    )

    monkeypatch.setattr(svc, "REDIS", dummy, raising=True)
    monkeypatch.setattr(training_service, "REDIS", dummy, raising=True)

    called: list[tuple[str, int, str | None]] = []

    async def fake_train(symbol: str, *, lookback_days: int, profile: str | None = None, **_: Any) -> None:
        called.append((symbol, lookback_days, profile))

    monkeypatch.setattr(training_service, "_train_models", fake_train, raising=True)

    await svc._ensure_trained_models("AAPL", required_lookback=120)
    assert called == [], "Expected training to be skipped when cached model is fresh enough"


@pytest.mark.asyncio
async def test_ensure_trained_models_retrains_on_low_lookback(monkeypatch):
    dummy = DummyRedis()
    key = await svc._model_key("MSFT_linear")
    dummy.store[key] = json.dumps(
        {
            "lookback_days": 60,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "profile": "quick",
        }
    )

    monkeypatch.setattr(svc, "REDIS", dummy, raising=True)
    monkeypatch.setattr(training_service, "REDIS", dummy, raising=True)

    called: list[tuple[str, int, str | None]] = []

    async def fake_train(symbol: str, *, lookback_days: int, profile: str | None = None, **_: Any) -> None:
        called.append((symbol, lookback_days, profile))

    monkeypatch.setattr(training_service, "_train_models", fake_train, raising=True)

    await svc._ensure_trained_models("MSFT", required_lookback=120)
    assert called == [("MSFT", 120, "quick")], "Expected training to run when cached lookback is insufficient"


@pytest.mark.asyncio
async def test_ensure_trained_models_retrains_when_stale(monkeypatch):
    if svc.TRAIN_REFRESH_DELTA is None:
        pytest.skip("Training refresh not enabled")

    dummy = DummyRedis()
    key = await svc._model_key("TSLA_linear")
    stale_at = datetime.now(timezone.utc) - svc.TRAIN_REFRESH_DELTA - timedelta(seconds=1)
    dummy.store[key] = json.dumps(
        {
            "lookback_days": 365,
            "trained_at": stale_at.isoformat(),
            "profile": "quick",
        }
    )

    monkeypatch.setattr(svc, "REDIS", dummy, raising=True)
    monkeypatch.setattr(training_service, "REDIS", dummy, raising=True)

    called: list[tuple[str, int, str | None]] = []

    async def fake_train(symbol: str, *, lookback_days: int, profile: str | None = None, **_: Any) -> None:
        called.append((symbol, lookback_days, profile))

    monkeypatch.setattr(training_service, "_train_models", fake_train, raising=True)

    await svc._ensure_trained_models("TSLA", required_lookback=120)
    assert called == [("TSLA", 120, "quick")], "Expected training to refresh when cached model is stale"


@pytest.mark.asyncio
async def test_ensure_trained_models_uses_profile_specific_cache(monkeypatch):
    dummy = DummyRedis()
    quick_key = await svc._model_key("NFLX_linear")
    deep_key = await svc._model_key("NFLX_linear:DEEP")
    now = datetime.now(timezone.utc)

    dummy.store[quick_key] = json.dumps(
        {
            "lookback_days": 180,
            "trained_at": now.isoformat(),
            "profile": "quick",
        }
    )

    monkeypatch.setattr(svc, "REDIS", dummy, raising=True)
    monkeypatch.setattr(training_service, "REDIS", dummy, raising=True)

    called: list[tuple[str, int, str | None]] = []

    async def fake_train(symbol: str, *, lookback_days: int, profile: str | None = None, **_: Any) -> None:
        called.append((symbol, lookback_days, profile))

    monkeypatch.setattr(training_service, "_train_models", fake_train, raising=True)

    # Deep profile should ignore quick cache and trigger training
    await svc._ensure_trained_models("NFLX", required_lookback=3650, profile="deep")
    assert called == [("NFLX", 3650, "deep")]

    # Simulate deep profile cache being populated
    dummy.store[deep_key] = json.dumps(
        {
            "lookback_days": 3650,
            "trained_at": now.isoformat(),
            "profile": "deep",
        }
    )
    called.clear()

    # Deep profile call should now skip training
    await svc._ensure_trained_models("NFLX", required_lookback=3650, profile="deep")
    assert called == []

    # Quick profile still retrains if its own cache is insufficient
    dummy.store[quick_key] = json.dumps(
        {
            "lookback_days": 60,
            "trained_at": now.isoformat(),
            "profile": "quick",
        }
    )
    called.clear()
    await svc._ensure_trained_models("NFLX", required_lookback=120, profile="quick")
    assert called == [("NFLX", 120, "quick")]


def test_build_training_dataset_sliding_features():
    prices = [100.0 + i * 0.5 for i in range(250)]
    features, X, y, dataset_hash = svc._build_training_dataset(prices, horizon=1)

    assert features == ["mom_20", "rvol_20", "autocorr_5"]
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == len(features)
    assert len(dataset_hash) == 40
    # All targets should be either 0 or 1
    assert set(np.unique(y)).issubset({0.0, 1.0})
