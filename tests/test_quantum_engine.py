import os
from typing import Any

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import src.quantum_engine as qe  # noqa: E402


class _StubResultElement:
    def __init__(self, counts: dict[str, int]):
        self.data = type(
            "D",
            (),
            {"meas": type("M", (), {"get_counts": lambda self: counts})()},
        )()


class _StubResultSequence:
    def __init__(self, counts: dict[str, int]):
        self._counts = counts

    def __getitem__(self, idx: int):
        return _StubResultElement(self._counts)


class _StubRunHandle:
    def __init__(self, counts: dict[str, int]):
        self._counts = counts

    def result(self):
        return _StubResultSequence(self._counts)


class _RecordingSampler:
    def __init__(self, counts: dict[str, int]):
        self.counts = counts
        self.calls: list[dict[str, Any]] = []

    def run(self, circuits, shots=None):
        self.calls.append({"shots": shots, "n_circuits": len(circuits)})
        return _StubRunHandle(self.counts)


def test_estimate_indicator_probability_uses_ibm(monkeypatch):
    monkeypatch.setenv("PT_QUANT_TARGET", "ibm")
    sampler = _RecordingSampler({"00": 25, "01": 75, "10": 25, "11": 75})
    monkeypatch.setattr(qe, "_build_ibm_sampler", lambda **kwargs: sampler, raising=True)

    prob = qe.estimate_indicator_probability([1, 2, 3, 4], [False, False, True, True], shots=2048)

    assert pytest.approx(prob, rel=1e-6) == 0.75
    assert sampler.calls, "IBM sampler should have been invoked"
    assert sampler.calls[0]["shots"] == 2048


def test_estimate_indicator_probability_falls_back_to_aer(monkeypatch):
    monkeypatch.setenv("PT_QUANT_TARGET", "ibm")
    monkeypatch.setattr(qe, "_build_ibm_sampler", lambda **kwargs: None, raising=True)

    aer_sampler = _RecordingSampler({"0": 50, "1": 50})

    monkeypatch.setattr(qe, "AerSampler", lambda backend=None: aer_sampler, raising=True)

    prob = qe.estimate_indicator_probability([1, 1], [False, True], shots=512)

    assert pytest.approx(prob, rel=1e-6) == 0.5
    assert aer_sampler.calls, "Aer sampler should have been invoked as fallback"
    assert aer_sampler.calls[0]["shots"] == 512
