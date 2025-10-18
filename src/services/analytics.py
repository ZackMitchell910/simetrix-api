from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import HTTPException

from src.core import EXPORT_ROOT
from src.feature_store import (
    connect as feature_store_connect,
    export_outcomes_parquet,
    export_predictions_parquet,
    fetch_metrics_daily_arrow,
    generate_accuracy_statements,
)
from src.services.export_utils import maybe_upload_to_s3


def _parse_day(day: Optional[str]) -> date:
    if not day:
        return datetime.now(timezone.utc).date()
    try:
        return date.fromisoformat(day)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="day must be YYYY-MM-DD") from exc


def export_predictions(
    *,
    day: Optional[str],
    symbol: Optional[str],
    limit: int,
    push_to_s3: bool,
) -> dict[str, Any]:
    if not callable(feature_store_connect):
        raise HTTPException(status_code=503, detail="feature_store_unavailable")

    day_obj = _parse_day(day)
    dest_dir = EXPORT_ROOT / "manual" / "predictions"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"predictions_{day_obj.isoformat()}.parquet"

    try:
        path = export_predictions_parquet(dest, symbol=symbol, day=day_obj, limit=limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    s3_uri = maybe_upload_to_s3(Path(path)) if push_to_s3 else None
    return {"path": str(path), "s3_uri": s3_uri}


def accuracy_statements(limit: int) -> dict[str, Any]:
    if not callable(feature_store_connect):
        raise HTTPException(status_code=503, detail="feature_store_unavailable")

    con = feature_store_connect()
    try:
        statements = generate_accuracy_statements(con, limit=limit)
    finally:
        try:
            con.close()
        except Exception:
            pass
    return {"statements": statements}


def metrics_summary(
    *,
    symbol: Optional[str],
    horizon_days: Optional[int],
    limit: int,
) -> dict[str, Any]:
    if not callable(feature_store_connect):
        raise HTTPException(status_code=503, detail="feature_store_unavailable")

    con = feature_store_connect()
    try:
        table = fetch_metrics_daily_arrow(
            con,
            symbol=symbol,
            horizon_days=horizon_days,
            limit=limit,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        try:
            con.close()
        except Exception:
            pass

    if table.num_rows == 0:
        return {"summary": {"count": 0}, "rows": []}

    rows = table.to_pylist()
    total_n = sum(int(r.get("n") or 0) for r in rows)
    avg_brier = None
    avg_rmse = None
    avg_mape = None
    coverage = None

    if rows:
        briers = [float(r["brier"]) for r in rows if r.get("brier") is not None]
        rmses = [float(r["rmse"]) for r in rows if r.get("rmse") is not None]
        mapes = [float(r["mape"]) for r in rows if r.get("mape") is not None]
        if briers:
            avg_brier = float(np.mean(briers))
        if rmses:
            avg_rmse = float(np.mean(rmses))
        if mapes:
            avg_mape = float(np.mean(mapes))

    if total_n:
        coverage = sum(
            float(r.get("p90_cov") or 0.0) * int(r.get("n") or 0) for r in rows
        ) / total_n

    summary = {
        "count": len(rows),
        "total_samples": total_n,
        "avg_brier": avg_brier,
        "avg_rmse": avg_rmse,
        "avg_mape": avg_mape,
        "weighted_p90_cov": coverage,
    }
    return {"summary": summary, "rows": rows}


def export_outcomes(
    *,
    day: Optional[str],
    symbol: Optional[str],
    limit: int,
    push_to_s3: bool,
) -> dict[str, Any]:
    if not callable(feature_store_connect):
        raise HTTPException(status_code=503, detail="feature_store_unavailable")

    day_obj = _parse_day(day)
    dest_dir = EXPORT_ROOT / "manual" / "outcomes"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"outcomes_{day_obj.isoformat()}.parquet"

    try:
        path = export_outcomes_parquet(dest, symbol=symbol, day=day_obj, limit=limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    s3_uri = maybe_upload_to_s3(Path(path)) if push_to_s3 else None
    return {"path": str(path), "s3_uri": s3_uri}
