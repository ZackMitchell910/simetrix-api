# backend/src/feature_store.py
"""
PathPanda DuckDB Feature Store
- Tables: predictions, outcomes, metrics_daily (+ indexes)
- Write adapters: single + batch upserts
- Redis→DuckDB migration helper
- Daily metrics rollup (RMSE/MAE/Brier/CRPS/MAPE, P80/P90 coverage)
"""

from __future__ import annotations
import os, json, datetime as dt
from typing import Iterable, Optional, Mapping, Any, List, Tuple

import duckdb
import numpy as np

DB_PATH = os.getenv("PT_DB_PATH", "data/pathpanda.duckdb")

# ---------- Schema ----------

DDL = r"""
PRAGMA threads=4;

CREATE TABLE IF NOT EXISTS predictions (
  run_id        TEXT,
  model_id      TEXT,
  symbol        TEXT,
  issued_at     TIMESTAMP,
  horizon_days  INTEGER,
  yhat_mean     DOUBLE,
  prob_up       DOUBLE,
  q05           DOUBLE,
  q50           DOUBLE,
  q95           DOUBLE,
  uncertainty   DOUBLE,
  features_ref  TEXT,
  PRIMARY KEY (run_id, model_id)
);

CREATE TABLE IF NOT EXISTS outcomes (
  run_id        TEXT PRIMARY KEY,
  symbol        TEXT,
  realized_at   TIMESTAMP,
  y             DOUBLE
);

CREATE TABLE IF NOT EXISTS metrics_daily (
  date          DATE,
  model_id      TEXT,
  symbol        TEXT,
  horizon_days  INTEGER,
  rmse          DOUBLE,
  mae           DOUBLE,
  brier         DOUBLE,
  crps          DOUBLE,
  mape          DOUBLE,  -- New: Mean Absolute Percentage Error
  p80_cov       DOUBLE,
  p90_cov       DOUBLE,
  n             INTEGER,
  PRIMARY KEY (date, model_id, symbol, horizon_days)
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_pred_symbol_time ON predictions(symbol, issued_at DESC);
CREATE INDEX IF NOT EXISTS idx_pred_model_time  ON predictions(model_id, issued_at DESC);
CREATE INDEX IF NOT EXISTS idx_out_symbol_time  ON outcomes(symbol, realized_at DESC);

CREATE TABLE IF NOT EXISTS mc_params (
  symbol         TEXT PRIMARY KEY,
  mu             DOUBLE,
  sigma          DOUBLE,
  lookback_mu    INTEGER,
  lookback_sigma INTEGER,
  updated_at     TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS mc_metrics (
  as_of          DATE,
  symbol         TEXT,
  horizon_days   INTEGER,
  mape           DOUBLE,
  mdape          DOUBLE,
  n              INTEGER,
  mu             DOUBLE,
  sigma          DOUBLE,
  n_paths        INTEGER,
  lookback_mu    INTEGER,
  lookback_sigma INTEGER,
  seeded_by      TEXT,
  PRIMARY KEY (as_of, symbol, horizon_days)
);

CREATE INDEX IF NOT EXISTS idx_fs_mc_metrics_symbol_time
  ON mc_metrics(symbol, as_of DESC, horizon_days);

CREATE SEQUENCE IF NOT EXISTS ingest_log_id_seq START 1;

CREATE TABLE IF NOT EXISTS ingest_log (
  id           BIGINT PRIMARY KEY DEFAULT nextval('ingest_log_id_seq'),
  as_of        DATE NOT NULL,
  source       TEXT NOT NULL,
  row_count    INTEGER NOT NULL,
  sha256       TEXT NOT NULL,
  duration_ms  INTEGER,
  ok           BOOLEAN NOT NULL DEFAULT TRUE,
  error        TEXT,
  created_at   TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ingest_log_asof
  ON ingest_log(as_of DESC, id DESC);
"""

# ---------- Connection / bootstrap ----------

def connect() -> duckdb.DuckDBPyConnection:
    folder = os.path.dirname(DB_PATH) or "."
    os.makedirs(folder, exist_ok=True)
    con = duckdb.connect(DB_PATH)
    con.execute(DDL)
    return con

def get_con() -> duckdb.DuckDBPyConnection:
    return connect()

# ---------- Normalization helpers ----------

_PRED_COLS = (
    "run_id","model_id","symbol","issued_at","horizon_days",
    "yhat_mean","prob_up","q05","q50","q95","uncertainty","features_ref"
)
def upsert_mc_params(con, symbol: str, mu: float, sigma: float,
                     lookback_mu: int, lookback_sigma: int) -> None:
    con.execute("""
        INSERT OR REPLACE INTO mc_params
        (symbol, mu, sigma, lookback_mu, lookback_sigma, updated_at)
        VALUES (?, ?, ?, ?, ?, now())
    """, (symbol, mu, sigma, lookback_mu, lookback_sigma))

def insert_mc_metrics(con, rows: list[tuple]) -> int:
    """
    rows: [(as_of, symbol, horizon_days, mape, mdape, n, mu, sigma, n_paths,
            lookback_mu, lookback_sigma, seeded_by), ...]
    """
    con.executemany("""
        INSERT OR REPLACE INTO mc_metrics
        (as_of, symbol, horizon_days, mape, mdape, n, mu, sigma, n_paths,
         lookback_mu, lookback_sigma, seeded_by)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    return con.rowcount

def log_ingest_event(
    con,
    *,
    as_of: dt.date | str,
    source: str,
    row_count: int,
    sha256: str,
    duration_ms: int | None = None,
    ok: bool = True,
    error: str | None = None,
) -> None:
    as_of_val = str(as_of)
    con.execute(
        """
        INSERT INTO ingest_log (as_of, source, row_count, sha256, duration_ms, ok, error)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (as_of_val, source, int(row_count), sha256, duration_ms, bool(ok), error),
    )

def get_mc_metrics(con, symbol: str | None = None, limit: int = 200) -> list[dict]:
    where = "WHERE symbol = ?" if symbol else ""
    args  = [symbol] if symbol else []
    rs = con.execute(f"""
        SELECT as_of, symbol, horizon_days, mape, mdape, n, mu, sigma, n_paths,
               lookback_mu, lookback_sigma, seeded_by
        FROM mc_metrics
        {where}
        ORDER BY as_of DESC, horizon_days
        LIMIT {int(max(1, min(limit, 1000)))}
    """, args).fetchall()
    cols = ["as_of","symbol","horizon_days","mape","mdape","n","mu","sigma",
            "n_paths","lookback_mu","lookback_sigma","seeded_by"]
    return [dict(zip(cols, r)) for r in rs]
def get_latest_mc_metric(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    horizon_days: int,
) -> dict | None:
    row = con.execute(
        """
        SELECT as_of, mape, mdape, n, mu, sigma, n_paths, lookback_mu, lookback_sigma, seeded_by
        FROM mc_metrics
        WHERE symbol = ? AND horizon_days = ?
        ORDER BY as_of DESC
        LIMIT 1
        """,
        [symbol.upper(), int(horizon_days)],
    ).fetchone()
    if not row:
        return None
    cols = ["as_of", "mape", "mdape", "n", "mu", "sigma", "n_paths", "lookback_mu", "lookback_sigma", "seeded_by"]
    return dict(zip(cols, row))
def _pred_tuple(pred: Mapping) -> tuple:
    return tuple(pred.get(c) for c in _PRED_COLS)

def _pred_tuples(preds: Iterable[Mapping]) -> list[tuple]:
    return [_pred_tuple(p) for p in preds]

# ---------- Write adapters ----------

def log_prediction(pred: Mapping[str, Any]) -> None:
    """Single-row upsert for a prediction artifact."""
    con = get_con()
    con.execute(f"""
        INSERT OR REPLACE INTO predictions ({','.join(_PRED_COLS)})
        VALUES ({','.join(['?']*len(_PRED_COLS))})
    """, _pred_tuple(pred))

def log_predictions(preds: Iterable[Mapping[str, Any]]) -> int:
    """Batch upsert for prediction artifacts."""
    con = get_con()
    con.executemany(f"""
        INSERT OR REPLACE INTO predictions ({','.join(_PRED_COLS)})
        VALUES ({','.join(['?']*len(_PRED_COLS))})
    """, _pred_tuples(preds))
    return con.rowcount

def insert_outcome(outcome: Mapping[str, Any]) -> None:
    """Single-row upsert for realized outcome (y)."""
    con = get_con()
    con.execute("""
        INSERT OR REPLACE INTO outcomes (run_id, symbol, realized_at, y)
        VALUES (?, ?, ?, ?)
    """, (outcome["run_id"], outcome["symbol"], outcome["realized_at"], outcome["y"]))

# ---------- Retrieval ----------

def matured_predictions(limit: int = 5000) -> list[tuple]:
    """Matured but unlabeled predictions: (run_id, issued_at, symbol, horizon_days)."""
    con = get_con()
    return con.execute("""
        SELECT p.run_id, p.issued_at, p.symbol, p.horizon_days
        FROM predictions p
        LEFT JOIN outcomes o USING (run_id)
        WHERE o.run_id IS NULL
          AND p.issued_at + (p.horizon_days * INTERVAL 1 DAY) <= now()
        ORDER BY p.issued_at DESC
        LIMIT ?
    """, [limit]).fetchall()

# ---------- Metrics Rollup ----------

def _brier(p: float, y: int) -> float:
    return (p - y)**2

def _crps_approx(y: float, q05: float, q50: float, q95: float) -> float:
    # Simplified CRPS approximation using quantiles
    if q05 is None or q95 is None:
        return abs(q50 - y) if q50 is not None else abs(y)
    # More accurate formula can be implemented; placeholder
    return 0.5 * (abs(y - q50) + (q95 - q05) / 2)

def compute_and_upsert_metrics_daily(
    con: duckdb.DuckDBPyConnection,
    day: Optional[dt.date] = None,
    model_id: Optional[str] = None,
    symbol: Optional[str] = None,
    horizon_days: Optional[int] = None,
) -> int:
    """Computes/upserts aggregate metrics for matured predictions on a given day."""
    if day is None:
        day = dt.date.today() - dt.timedelta(days=1)  # Yesterday by default

    where = ["date(p.issued_at) = ?"]
    params = [day]

    if model_id:
        where.append("p.model_id = ?")
        params.append(model_id)
    if symbol:
        where.append("p.symbol = ?")
        params.append(symbol)
    if horizon_days is not None:
        where.append("p.horizon_days = ?")
        params.append(int(horizon_days))

    rows = con.execute(f"""
        SELECT p.run_id, p.model_id, p.symbol, p.horizon_days,
               p.yhat_mean, p.prob_up, p.q05, p.q50, p.q95, p.features_ref,
               o.y
        FROM predictions p
        JOIN outcomes    o ON o.run_id = p.run_id
        WHERE {" AND ".join(where)}
    """, params).fetchall()

    if not rows:
        return 0

    from collections import defaultdict
    agg = defaultdict(lambda: {"se":[], "ae":[], "ape":[], "br":[], "cr":[], "cov80":0, "cov90":0, "n":0})

    for (run_id, mid, sym, hz, yhat, p_up, q05, q50, q95, fref, y) in rows:
        if y is None or yhat is None: 
            continue

        se = (yhat - y)**2
        ae = abs(yhat - y)
        ape = abs((yhat - y) / y) * 100 if y != 0 else 0  # Absolute Percentage Error

        realized_up = None
        if fref:
            try:
                fr = json.loads(fref)
                if isinstance(fr, dict) and "realized_up" in fr:
                    realized_up = int(bool(fr["realized_up"]))
            except Exception:
                pass
        if realized_up is None:
            thresh = q50 if q50 is not None else yhat
            realized_up = int(y >= thresh)

        br = _brier(p_up if p_up is not None else 0.5, realized_up)
        cr = _crps_approx(y, q05, q50, q95)

        cov90_hit = int((q05 is not None and q95 is not None and q05 <= y <= q95))
        cov80_hit = cov90_hit  # until q10/q10 added

        key = (day, mid, sym, hz)
        a = agg[key]
        a["se"].append(se); a["ae"].append(ae); a["ape"].append(ape); a["br"].append(br); a["cr"].append(cr)
        a["cov80"] += cov80_hit; a["cov90"] += cov90_hit; a["n"] += 1

    upserts = []
    for (day_, mid, sym, hz), a in agg.items():
        n = max(a["n"], 1)
        rmse = (sum(a["se"]) / n) ** 0.5
        mae  =  sum(a["ae"]) / n
        mape = sum(a["ape"]) / n  # New: Mean Absolute Percentage Error
        brier = sum(a["br"]) / n
        crps  = sum(a["cr"]) / n
        p80   = a["cov80"] / n
        p90   = a["cov90"] / n
        upserts.append([day_, mid, sym, hz, rmse, mae, brier, crps, mape, p80, p90, n])

    con.executemany("""
        INSERT OR REPLACE INTO metrics_daily
        (date, model_id, symbol, horizon_days, rmse, mae, brier, crps, mape, p80_cov, p90_cov, n)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, upserts)

    return len(upserts)

# New: Function to generate nuanced accuracy statements
def generate_accuracy_statements(
    limit: int = 50,
    symbol: Optional[str] = None,
    min_date: Optional[dt.date] = None
) -> List[str]:
    con = get_con()
    where = ["o.y IS NOT NULL"]
    params = []
    if symbol:
        where.append("p.symbol = ?")
        params.append(symbol)
    if min_date:
        where.append("p.issued_at >= ?")
        params.append(min_date)

    rows = con.execute(f"""
        SELECT p.symbol, p.horizon_days, p.q50, o.y
        FROM predictions p
        JOIN outcomes o ON o.run_id = p.run_id
        WHERE {" AND ".join(where)}
        ORDER BY p.issued_at DESC
        LIMIT ?
    """, params + [limit]).fetchall()

    statements = []
    for (sym, horizon, q50, actual) in rows:
        if actual == 0:
            continue
        pct_error = abs((q50 - actual) / actual) * 100
        statements.append(f"Simetrix predicted {sym}'s price within {pct_error:.1f}% of the actual closing value over a {horizon}-day horizon.")

    return statements

def get_recent_mdape(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    horizon_days: int,
    lookback_days: int = 30
) -> float:
    """
    Returns recent MDAPE across that symbol/horizon. If missing, returns NaN.
    """
    row = con.execute("""
        SELECT median(mape) AS mdape
        FROM metrics_daily
        WHERE symbol = ? AND horizon_days = ? AND date >= date 'now' - INTERVAL ? DAY
    """, [symbol.upper(), int(horizon_days), int(lookback_days)]).fetchone()
    if not row or row[0] is None:
        return float("nan")
    return float(row[0])

def get_recent_coverage(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    horizon_days: int,
    lookback_days: int = 21
) -> Tuple[float, int]:
    """
    Returns (avg_p90_cov, n) over the last N days for symbol + horizon.
    If no rows, returns (float("nan"), 0).
    """
    rows = con.execute("""
        SELECT p90_cov, n
        FROM metrics_daily
        WHERE symbol = ? AND horizon_days = ? AND date >= date 'now' - INTERVAL ? DAY
    """, [symbol.upper(), int(horizon_days), int(lookback_days)]).fetchall()
    if not rows:
        return float("nan"), 0
    # Weighted average by n
    num, den = 0.0, 0
    for p90, nn in rows:
        if p90 is None or nn is None: 
            continue
        num += float(p90) * int(nn)
        den += int(nn)
    return ((num / den) if den else float("nan"), den)

# ---------- Maintenance ----------

def checkpoint(con): con.execute("CHECKPOINT")
def vacuum(con):     con.execute("CHECKPOINT")
