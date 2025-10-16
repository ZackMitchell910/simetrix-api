import os
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

# Ensure required env vars are present before importing the service module
os.environ.setdefault("PT_POLYGON_KEY", "test-key")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import predictive_service as svc


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
        }
    )

    monkeypatch.setattr(svc, "REDIS", dummy, raising=True)

    called: list[tuple[str, int]] = []

    async def fake_train(symbol: str, lookback_days: int) -> None:
        called.append((symbol, lookback_days))

    monkeypatch.setattr(svc, "_train_models", fake_train, raising=True)

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
        }
    )

    monkeypatch.setattr(svc, "REDIS", dummy, raising=True)

    called: list[tuple[str, int]] = []

    async def fake_train(symbol: str, lookback_days: int) -> None:
        called.append((symbol, lookback_days))

    monkeypatch.setattr(svc, "_train_models", fake_train, raising=True)

    await svc._ensure_trained_models("MSFT", required_lookback=120)
    assert called == [("MSFT", 120)], "Expected training to run when cached lookback is insufficient"


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
        }
    )

    monkeypatch.setattr(svc, "REDIS", dummy, raising=True)

    called: list[tuple[str, int]] = []

    async def fake_train(symbol: str, lookback_days: int) -> None:
        called.append((symbol, lookback_days))

    monkeypatch.setattr(svc, "_train_models", fake_train, raising=True)

    await svc._ensure_trained_models("TSLA", required_lookback=120)
    assert called == [("TSLA", 120)], "Expected training to refresh when cached model is stale"
