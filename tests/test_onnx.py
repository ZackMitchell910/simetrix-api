import math
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import predictive_api as svc  # noqa: E402


@pytest.mark.skipif(not svc._onnx_supported(), reason="ONNX runtime not available")
def test_linear_onnx_export_and_infer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    symbol = "TEST"
    weights = [0.15, 0.25, -0.35]
    features = ["f1", "f2"]

    path = svc._export_linear_onnx(symbol, weights, features)
    assert path is not None
    assert path.exists()

    vector = [0.4, -0.2]
    prob = svc._run_linear_onnx(symbol, vector)
    assert prob is not None

    expected = 1.0 / (1.0 + math.exp(-(weights[0] + weights[1] * vector[0] + weights[2] * vector[1])))
    assert prob == pytest.approx(expected, rel=1e-6)

    # Clear cache so subsequent tests start fresh
    svc._ONNX_SESSION_CACHE.clear()
