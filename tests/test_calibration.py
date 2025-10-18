import os
import sys
import datetime as dt
from pathlib import Path
from typing import Any, List

import duckdb
import numpy as np
import pytest

os.environ.setdefault("PT_POLYGON_KEY", "test-key")
os.environ.setdefault("PT_SKIP_ONNXRUNTIME", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import feature_store as fs
from src import predictive_api as svc


def test_fit_residual_distribution_quantiles():
    rng = np.random.default_rng(123)
    sample = rng.standard_t(df=5, size=800) * 0.025
    fit = svc._fit_residual_distribution(sample, horizon_days=5)

    assert fit["sample_std"] > 0
    assert fit["sigma_ann"] > 0

    q05_true = np.percentile(sample, 5)
    q50_true = np.percentile(sample, 50)
    q95_true = np.percentile(sample, 95)

    assert fit["q05"] == pytest.approx(q05_true, abs=0.01)
    assert fit["q50"] == pytest.approx(q50_true, abs=0.01)
    assert fit["q95"] == pytest.approx(q95_true, abs=0.02)


def test_metrics_rollup_brier_and_coverage():
    con = duckdb.connect(":memory:")
    con.execute(fs.DDL)

    day = dt.date(2024, 1, 1)
    issued_at = dt.datetime(2024, 1, 1, 15, 30, tzinfo=dt.timezone.utc)
    realized_at = dt.datetime(2024, 1, 6, 15, 30, tzinfo=dt.timezone.utc)

    pred_rows: List[tuple[Any, ...]] = [
        (
            "run-1",
            "ensemble-v1",
            "AAPL",
            issued_at,
            5,
            101.0,
            0.7,
            95.0,
            100.0,
            110.0,
            None,
            "{}",
        ),
        (
            "run-2",
            "ensemble-v1",
            "AAPL",
            issued_at,
            5,
            99.0,
            0.3,
            90.0,
            99.0,
            108.0,
            None,
            "{}",
        ),
    ]
    outcome_rows = [
        ("run-1", "AAPL", realized_at, 108.0),
        ("run-2", "AAPL", realized_at, 92.0),
    ]

    con.executemany(
        """
        INSERT INTO predictions
        (run_id, model_id, symbol, issued_at, horizon_days, yhat_mean, prob_up, q05, q50, q95, uncertainty, features_ref)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        pred_rows,
    )
    con.executemany(
        """
        INSERT INTO outcomes (run_id, symbol, realized_at, y)
        VALUES (?, ?, ?, ?)
        """,
        outcome_rows,
    )

    rows = fs.compute_and_upsert_metrics_daily(con, day=day)
    assert rows == 1

    row = con.execute(
        "SELECT brier, p90_cov FROM metrics_daily WHERE date = ?",
        [day],
    ).fetchone()
    assert row is not None
    brier, p90_cov = row

    # Expected Brier: average of (0.7-1)^2 and (0.3-0)^2 = 0.09
    assert brier == pytest.approx(0.09, rel=1e-2, abs=1e-3)
    assert p90_cov == pytest.approx(1.0, rel=1e-2)

    con.close()
