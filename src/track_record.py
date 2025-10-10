# backend/src/track_record.py
from __future__ import annotations
import json, hashlib
from datetime import datetime, date, timedelta, timezone
from typing import List, Dict, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

# Use the same DuckDB connection & file as the rest of your app
from .duck import get_conn  # <— reuse pt.duckdb and single connection

router = APIRouter(prefix="/track", tags=["track"])

# ---------- Schema (created once) ----------
def _ensure_schema():
    con = get_conn()
    con.execute("""
    CREATE TABLE IF NOT EXISTS cohorts (
      cohort_id     BIGINT PRIMARY KEY,
      cohort_date   DATE,
      horizon_days  INTEGER,
      model_version TEXT,
      seed          BIGINT,
      artifact_hash TEXT,
      created_at    TIMESTAMP DEFAULT now()
    );
    """)
    con.execute("""
    CREATE TABLE IF NOT EXISTS track_predictions (
      cohort_id      BIGINT,
      symbol         TEXT,
      start_price    DOUBLE,
      tgt_date       DATE,
      forecast_med   DOUBLE,
      forecast_mean  DOUBLE,
      band_p05       DOUBLE,
      band_p95       DOUBLE,
      expect_ret     DOUBLE,
      hit_prob       DOUBLE,
      sim_meta       JSON,
      PRIMARY KEY (cohort_id, symbol)
    );
    """)
    con.execute("""
    CREATE TABLE IF NOT EXISTS track_realized (
      cohort_id      BIGINT,
      symbol         TEXT,
      realized_price DOUBLE,
      realized_ret   DOUBLE,
      abs_pct_err    DOUBLE,  -- Absolute Percentage Error
      within_90pc    BOOLEAN,
      computed_at    TIMESTAMP DEFAULT now(),
      PRIMARY KEY (cohort_id, symbol)
    );
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_track_preds_tgt ON track_predictions(tgt_date);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_track_preds_sym ON track_predictions(symbol);")
    con.execute("CREATE INDEX IF NOT EXISTS idx_track_realized_cohort ON track_realized(cohort_id);")

_ensure_schema()

# ---------- Helpers ----------
def _poly_key() -> str:
    import os
    return (os.getenv("PT_POLYGON_KEY") or os.getenv("POLYGON_KEY") or "").strip()

def make_cohort_id(d: date, horizon: int) -> int:
    return int(d.strftime("%Y%m%d")) * 100 + int(horizon)

def merkleish_hash(rows: List[Dict]) -> str:
    # Simplified hash for artifact verification
    data = json.dumps(rows, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()[:16]

# ---------- Models ----------
class CohortRow(BaseModel):
    symbol: str
    start_price: float
    tgt_date: date
    forecast_med: float
    band_p05: Optional[float] = None
    band_p95: Optional[float] = None
    realized_price: Optional[float] = None
    abs_pct_err: Optional[float] = None  # Now as percentage
    within_90pc: Optional[bool] = None

class CohortDetail(BaseModel):
    cohort_id: int
    cohort_date: date
    horizon_days: int
    model_version: str
    artifact_hash: str
    picks: List[CohortRow]

class CohortSummary(BaseModel):
    cohort_id: int
    cohort_date: date
    horizon_days: int
    model_version: str
    artifact_hash: str
    n: int
    win_rate: float
    mdape: float  # Median Absolute Percentage Error
    coverage_90: float

# ---------- Endpoints ----------
@router.post("/cohort")
def create_cohort(
    horizon_days: int = 30,
    model_version: str = "v1",
    symbols: Optional[List[str]] = None,
    n_symbols: int = 50,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    # Implementation unchanged for brevity; integrate if needed
    pass

@router.get("/cohort-summary/{cohort_id}", response_model=CohortSummary)
def cohort_summary(cohort_id: int):
    con = get_conn()
    # Fetch and compute summary with MAPE/MDAPE
    # Example query enhancement
    rows = con.execute("""
        SELECT abs_pct_err, within_90pc
        FROM track_realized WHERE cohort_id = ?
    """, [cohort_id]).fetchall()
    
    if not rows:
        raise HTTPException(404, "No data for cohort")

    apes = [r[0] for r in rows if r[0] is not None]
    covs = [r[1] for r in rows if r[1] is not None]
    
    n = len(apes)
    win_rate = sum(1 for ape in apes if ape < 5) / n if n else 0  # Example: <5% as "win"
    mdape = np.median(apes) if apes else 0
    coverage = sum(covs) / len(covs) if covs else 0

    # Fetch header
    head = con.execute("""
      SELECT cohort_date, horizon_days, model_version, artifact_hash
      FROM cohorts WHERE cohort_id = ?
    """, [cohort_id]).fetchone()

    return CohortSummary(
        cohort_id=cohort_id,
        cohort_date=head[0],
        horizon_days=head[1],
        model_version=head[2],
        artifact_hash=head[3],
        n=n,
        win_rate=win_rate,
        mdape=mdape,
        coverage_90=coverage,
    )

@router.get("/cohort/{cohort_id}", response_model=CohortDetail)
def cohort_detail(cohort_id: int):
    con = get_conn()
    head = con.execute("""
      SELECT cohort_date, horizon_days, model_version, artifact_hash
      FROM cohorts WHERE cohort_id = ?
    """, [cohort_id]).fetchone()
    if not head:
        raise HTTPException(status_code=404, detail="Cohort not found.")
    cohort_date, horizon_days, model_version, ahash = head

    con.execute("""
      SELECT
        p.symbol, p.start_price, p.tgt_date, p.forecast_med, p.band_p05, p.band_p95,
        r.realized_price, r.abs_pct_err, r.within_90pc
      FROM track_predictions p
      LEFT JOIN track_realized r USING (cohort_id, symbol)
      WHERE p.cohort_id = ?
      ORDER BY p.expect_ret DESC
    """, [cohort_id])
    rows = con.fetchall()

    picks = [
        CohortRow(
            symbol=sym, start_price=float(sp), tgt_date=tgt,
            forecast_med=float(fmed),
            band_p05=(float(p05) if p05 is not None else None),
            band_p95=(float(p95) if p95 is not None else None),
            realized_price=(float(rz) if rz is not None else None),
            abs_pct_err=(float(ape) if ape is not None else None),
            within_90pc=within,
        )
        for (sym, sp, tgt, fmed, p05, p95, rz, ape, within) in rows
    ]

    return CohortDetail(
        cohort_id=cohort_id,
        cohort_date=cohort_date,
        horizon_days=int(horizon_days),
        model_version=model_version,
        artifact_hash=ahash,
        picks=picks
    )

@router.get("/cohorts")
def list_cohorts(limit: int = Query(30, ge=1, le=365)):
    con = get_conn()
    con.execute("""
      SELECT cohort_id, cohort_date, horizon_days, model_version, artifact_hash, created_at
      FROM cohorts ORDER BY cohort_date DESC LIMIT ?
    """, [limit])
    rows = con.fetchall()
    return [
        {
            "cohort_id": int(cid),
            "cohort_date": cdate,
            "horizon_days": int(hz),
            "model_version": mv,
            "artifact_hash": ah,
            "created_at": cat,
        } for (cid, cdate, hz, mv, ah, cat) in rows
    ]