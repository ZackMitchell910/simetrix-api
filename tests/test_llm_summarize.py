import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Ensure required env vars so predictive_service imports cleanly
os.environ.setdefault("PT_POLYGON_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("XAI_API_KEY", "dummy")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import predictive_service as svc  # noqa: E402


class _DummyResponse:
    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    def raise_for_status(self) -> None:  # noqa: D401 - mimic httpx.Response
        return None

    def json(self) -> Dict[str, Any]:
        return self._data


class _DummyAsyncClient:
    def __init__(self, data: Dict[str, Any]) -> None:
        self._data = data

    async def __aenter__(self) -> "_DummyAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, *args, **kwargs) -> _DummyResponse:
        return _DummyResponse(self._data)


def _make_dummy_client(data: Dict[str, Any]):
    def factory(*args, **kwargs):
        return _DummyAsyncClient(data)

    return factory


_COMMON_PAYLOAD = {
    "summary": "Outlook stays constructive with balanced risks.",
    "what_it_means": ["Median path tilts upward", "Volatility remains contained"],
    "risks": ["Macro shocks could derail upside"],
    "watch": ["Upcoming earnings"],
    "confidence": "medium",
    "metrics": {"prob_up_end": 0.55, "median_return_pct": 4.2},
}


@pytest.mark.asyncio
async def test_llm_summarize_async_parses_structured_dict(monkeypatch):
    data = {"choices": [{"message": {"content": dict(_COMMON_PAYLOAD)}}]}
    monkeypatch.setattr(svc.httpx, "AsyncClient", _make_dummy_client(data), raising=False)

    result = await svc.llm_summarize_async(
        {"role": "user", "content": "context"},
        prefer_xai=True,
        xai_key="dummy",
        oai_key=None,
    )

    assert result == _COMMON_PAYLOAD


@pytest.mark.asyncio
async def test_llm_summarize_async_parses_markdown_fenced_json(monkeypatch):
    fenced = "```json\n" + svc.json.dumps(_COMMON_PAYLOAD) + "\n```"
    data = {"choices": [{"message": {"content": fenced}}]}
    monkeypatch.setattr(svc.httpx, "AsyncClient", _make_dummy_client(data), raising=False)

    result = await svc.llm_summarize_async(
        {"role": "user", "content": "context"},
        prefer_xai=True,
        xai_key="dummy",
        oai_key=None,
    )

    assert result == _COMMON_PAYLOAD
