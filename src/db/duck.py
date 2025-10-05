import os
import pathlib
import json
import duckdb
from typing import Optional, Dict, Any, List, Tuple

# Default: ../data/pt.duckdb relative to this file
DB_PATH = os.getenv(
    "PT_DUCKDB_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "pt.duckdb"),
)

DDL = """
CREATE TABLE IF NOT EXISTS predictions (
  pred_id UUID PRIMARY KEY,
  ts TIMESTAMP NOT NULL,
  symbol TEXT NOT NULL,
  horizon_d INTEGER NOT NULL,
  model_id TEXT NOT NULL,
  prob_up_next DOUBLE,
  p05 DOUBLE, p50 DOUBLE, p95 DOUBLE,
  spot0 DOUBLE,
  user_ctx JSON,
  run_id TEXT
);

CREATE TABLE IF NOT EXISTS outcomes (
  pred_id UUID REFERENCES predictions(pred_id),
  realized_ts TIMESTAMP NOT NULL,
  realized_price DOUBLE NOT NULL,
  ret DOUBLE NOT NULL,
  label_up BOOLEAN NOT NULL,
  PRIMARY KEY (pred_id)
);

CREATE TABLE IF NOT EXISTS metrics_daily (
  date DATE,
  model_id TEXT,
  symbol TEXT,
  crps DOUBLE, brier DOUBLE, ece DOUBLE,
  cov_p80 DOUBLE, cov_p95 DOUBLE,
  drift_psi DOUBLE,
  n_preds INTEGER,
  PRIMARY KEY (date, model_id, symbol)
);

-- Helpful indexes for fast windows & lookups
CREATE INDEX IF NOT EXISTS idx_predictions_symbol_ts ON predictions(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_outcomes_predid ON outcomes(pred_id);
"""

_conn: Optional[duckdb.DuckDBPyConnection] = None


def _ensure_dir() -> str:
    path = pathlib.Path(DB_PATH).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_conn() -> duckdb.DuckDBPyConnection:
    """Singleton connection with a sane thread setting."""
    global _conn
    if _conn is None:
        db_file = _ensure_dir()
        _conn = duckdb.connect(db_file)
        _conn.execute("PRAGMA threads=4;")
    return _conn


def init_schema() -> None:
    con = get_conn()
    con.execute(DDL)


def insert_prediction(row: Dict[str, Any]) -> None:
    """
    row keys:
      pred_id (uuid str), ts (datetime), symbol (str), horizon_d (int),
      model_id (str), prob_up_next (float), p05 (float or None),
      p50 (float or None), p95 (float or None), spot0 (float or None),
      user_ctx (dict|json str|None), run_id (str|None)
    """
    con = get_conn()
    ctx = row.get("user_ctx")
    if isinstance(ctx, dict):
        row["user_ctx"] = json.dumps(ctx)  # store JSON text
    placeholders = ",".join(["?"] * 12)
    con.execute(
        f"""
        INSERT OR REPLACE INTO predictions
        (pred_id, ts, symbol, horizon_d, model_id, prob_up_next, p05, p50, p95, spot0, user_ctx, run_id)
        VALUES ({placeholders})
        """,
        [
            row.get("pred_id"),
            row.get("ts"),
            row.get("symbol"),
            row.get("horizon_d"),
            row.get("model_id"),
            row.get("prob_up_next"),
            row.get("p05"),
            row.get("p50"),
            row.get("p95"),
            row.get("spot0"),
            row.get("user_ctx"),
            row.get("run_id"),
        ],
    )


def insert_outcome(pred_id, realized_ts, realized_price, ret, label_up: bool) -> None:
    con = get_conn()
    con.execute(
        """
        INSERT OR REPLACE INTO outcomes (pred_id, realized_ts, realized_price, ret, label_up)
        VALUES (?, ?, ?, ?, ?)
        """,
        [pred_id, realized_ts, realized_price, ret, label_up],
    )


def matured_predictions_now(limit: int = 5000) -> List[Tuple]:
    """
    Rows that have matured (ts + horizon_d days <= now) and are not yet labeled.
    Returns: [(pred_id, ts, symbol, horizon_d, spot0), ...]
    """
    con = get_conn()
    return con.execute(
        """
        SELECT p.pred_id, p.ts, p.symbol, p.horizon_d, p.spot0
        FROM predictions p
        LEFT JOIN outcomes o ON o.pred_id = p.pred_id
        WHERE o.pred_id IS NULL
          AND p.ts + (p.horizon_d * INTERVAL 1 DAY) <= now()
        ORDER BY p.ts
        LIMIT ?
        """,
        [int(limit)],
    ).fetchall()


def recent_predictions(symbol: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    con = get_conn()
    q = """
      SELECT pred_id, ts, symbol, horizon_d, model_id,
             prob_up_next, p05, p50, p95, spot0, run_id
      FROM predictions
    """
    args: list = []
    if symbol:
        q += " WHERE symbol = ?"
        args.append(symbol.upper())
    q += " ORDER BY ts DESC LIMIT ?"
    args.append(int(limit))
    rows = con.execute(q, args).fetchall()
    cols = [d[0] for d in con.description]
    items = [dict(zip(cols, r)) for r in rows]
    return items
