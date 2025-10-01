# backend/src/feature_store.py
"""
PathPanda DuckDB Feature Store
- Tables: predictions, outcomes, metrics_daily (+ indexes)
- Write adapters: single + batch upserts
- Redis→DuckDB migration helper
- Daily metrics rollup (RMSE/MAE/Brier/CRPS, P80/P90 coverage)
"""

from __future__ import annotations
import os, json, datetime as dt
from typing import Iterable, Optional, Mapping, Any

import duckdb

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
  p80_cov       DOUBLE,
  p90_cov       DOUBLE,
  n             INTEGER,
  PRIMARY KEY (date, model_id, symbol, horizon_days)
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_pred_symbol_time ON predictions(symbol, issued_at DESC);
CREATE INDEX IF NOT EXISTS idx_pred_model_time  ON predictions(model_id, issued_at DESC);
CREATE INDEX IF NOT EXISTS idx_out_symbol_time  ON outcomes(symbol, realized_at DESC);
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
_OUT_COLS = ("run_id","symbol","realized_at","y")

def _norm_pred_row(row: Mapping[str, Any]) -> list[Any]:
    vals = [row.get(k) for k in _PRED_COLS]
    # features_ref can be dict/list; store as JSON
    jf = vals[-1]
    if jf is not None and not isinstance(jf, str):
        vals[-1] = json.dumps(jf)
    # normalize timestamp
    if isinstance(vals[3], str):
        vals[3] = dt.datetime.fromisoformat(vals[3].replace("Z","+00:00"))
    return vals

def _norm_out_row(row: Mapping[str, Any]) -> list[Any]:
    vals = [row.get(k) for k in _OUT_COLS]
    if isinstance(vals[2], str):
        vals[2] = dt.datetime.fromisoformat(vals[2].replace("Z","+00:00"))
    return vals

# ---------- Single upserts ----------

def insert_prediction(con: duckdb.DuckDBPyConnection, row: Mapping[str, Any]) -> None:
    vals = _norm_pred_row(row)
    con.execute(
        f"INSERT OR REPLACE INTO predictions VALUES ({','.join('?'*len(_PRED_COLS))})",
        vals
    )

def insert_outcome(con: duckdb.DuckDBPyConnection, row: Mapping[str, Any]) -> None:
    vals = _norm_out_row(row)
    con.execute("INSERT OR REPLACE INTO outcomes VALUES (?,?,?,?)", vals)

# ---------- Batch upserts ----------

def insert_predictions(con: duckdb.DuckDBPyConnection, rows: Iterable[Mapping[str, Any]]) -> int:
    rows = list(rows)
    if not rows: return 0
    data = [_norm_pred_row(r) for r in rows]
    con.executemany(
        f"INSERT OR REPLACE INTO predictions VALUES ({','.join('?'*len(_PRED_COLS))})",
        data
    )
    return len(data)

def insert_outcomes(con: duckdb.DuckDBPyConnection, rows: Iterable[Mapping[str, Any]]) -> int:
    rows = list(rows)
    if not rows: return 0
    data = [_norm_out_row(r) for r in rows]
    con.executemany("INSERT OR REPLACE INTO outcomes VALUES (?,?,?,?)", data)
    return len(data)

# ---------- Simple reads ----------

def latest_predictions(con, limit: int = 50):
    return con.execute("""
        SELECT run_id, model_id, symbol, issued_at, horizon_days, prob_up, q50
        FROM predictions
        ORDER BY issued_at DESC
        LIMIT ?
    """, [limit]).fetchall()

def predictions_for_symbol(con, symbol: str, limit: int = 500, since: Optional[dt.datetime] = None):
    if since:
        return con.execute("""
            SELECT *
            FROM predictions
            WHERE symbol = ? AND issued_at >= ?
            ORDER BY issued_at DESC
            LIMIT ?
        """, [symbol, since, limit]).fetchall()
    return con.execute("""
        SELECT *
        FROM predictions
        WHERE symbol = ?
        ORDER BY issued_at DESC
        LIMIT ?
    """, [symbol, limit]).fetchall()

# ---------- Redis → DuckDB adapter (optional) ----------

def migrate_redis_predictions(con, redis_client, list_key: str = "pt:predictions", max_items: int = 10_000) -> int:
    """
    Pull JSON objects from a Redis list and persist to DuckDB predictions.
    Each entry should be a JSON object compatible with _PRED_COLS.
    """
    items = redis_client.lrange(list_key, 0, max_items-1) or []
    batch = []
    for raw in items:
        try:
            obj = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
            batch.append(obj)
        except Exception:
            continue
    if not batch: return 0
    return insert_predictions(con, batch)

# ---------- Metrics rollup ----------

def _brier(prob_up: float, realized_up: int) -> float:
    return (float(prob_up) - float(realized_up))**2

def _crps_approx(y: float, q05: Optional[float], q50: Optional[float], q95: Optional[float]) -> float:
    qs = [q for q in (q05, q50, q95) if q is not None]
    if not qs: return float("nan")
    errs = [abs(y - q) for q in qs]
    w    = [0.2, 0.6, 0.2][:len(errs)]
    s    = sum(w)
    return sum(e*wi for e,wi in zip(errs,w)) / (s if s else 1.0)

def compute_and_upsert_metrics_daily(
    con: duckdb.DuckDBPyConnection,
    day: Optional[dt.date] = None,
    model_id: Optional[str] = None,
    horizon_days: Optional[int] = None
) -> int:
    """
    Join predictions with outcomes on run_id for a given issued_at day,
    then aggregate metrics by (date, model_id, symbol, horizon_days).
    """
    if day is None:
        day = dt.date.today()

    where = ["CAST(issued_at AS DATE) = ?"]
    params: list[Any] = [day]
    if model_id:
        where.append("model_id = ?"); params.append(model_id)
    if horizon_days is not None:
        where.append("horizon_days = ?"); params.append(horizon_days)

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
    agg = defaultdict(lambda: {"se":[], "ae":[], "br":[], "cr":[], "cov80":0, "cov90":0, "n":0})

    for (run_id, mid, sym, hz, yhat, p_up, q05, q50, q95, fref, y) in rows:
        if y is None or yhat is None: 
            continue

        se = (yhat - y)**2
        ae = abs(yhat - y)

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
        cov80_hit = cov90_hit  # until q10/q90 are added

        key = (day, mid, sym, hz)
        a = agg[key]
        a["se"].append(se); a["ae"].append(ae); a["br"].append(br); a["cr"].append(cr)
        a["cov80"] += cov80_hit; a["cov90"] += cov90_hit; a["n"] += 1

    upserts = []
    for (day_, mid, sym, hz), a in agg.items():
        n = max(a["n"], 1)
        rmse = (sum(a["se"]) / n) ** 0.5
        mae  =  sum(a["ae"]) / n
        brier = sum(a["br"]) / n
        crps  = sum(a["cr"]) / n
        p80   = a["cov80"] / n
        p90   = a["cov90"] / n
        upserts.append([day_, mid, sym, hz, rmse, mae, brier, crps, p80, p90, n])

    con.executemany("""
        INSERT OR REPLACE INTO metrics_daily
        (date, model_id, symbol, horizon_days, rmse, mae, brier, crps, p80_cov, p90_cov, n)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, upserts)

    return len(upserts)

# ---------- Maintenance ----------

def checkpoint(con): con.execute("CHECKPOINT")
def vacuum(con):     con.execute("CHECKPOINT")
