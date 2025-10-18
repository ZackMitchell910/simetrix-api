import os
import pathlib
import json
import logging
import duckdb
from typing import Optional, Dict, Any, List, Tuple


def _default_data_root() -> pathlib.Path:
    env_root = os.getenv("PT_DATA_ROOT")
    if env_root:
        return pathlib.Path(env_root).expanduser().resolve()
    container_root = pathlib.Path("/data")
    if container_root.exists() and os.access(container_root, os.W_OK):
        return container_root
    return pathlib.Path(os.path.join(os.path.dirname(__file__), "..", "..", "data")).resolve()


DATA_ROOT = _default_data_root()
os.environ.setdefault("PT_DATA_ROOT", str(DATA_ROOT))
logger = logging.getLogger(__name__)

# Default: ../data/pt.duckdb relative to this file (or /data/pt.duckdb in containers)
DB_PATH = os.getenv(
    "PT_DUCKDB_PATH",
    str((DATA_ROOT / "pt.duckdb").resolve()),
)

_SCHEMA_READY = False
_PREDICTIONS_HAS_ISSUED_AT: Optional[bool] = None

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
  run_id TEXT,
  issued_at TIMESTAMP,
  features_ref TEXT
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
CREATE TABLE IF NOT EXISTS mc_params (
  symbol         TEXT PRIMARY KEY,
  mu             DOUBLE,
  sigma          DOUBLE,
  lookback_mu    INTEGER,
  lookback_sigma INTEGER,
  updated_at     TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mc_metrics (
  as_of         DATE,
  symbol        TEXT,
  horizon_days  INTEGER,
  mape          DOUBLE,
  mdape         DOUBLE,
  n             INTEGER,
  mu            DOUBLE,
  sigma         DOUBLE,
  n_paths       INTEGER,
  lookback_mu   INTEGER,
  lookback_sigma INTEGER,
  seeded_by     TEXT,
  PRIMARY KEY (as_of, symbol, horizon_days)
);
-- Helpful indexes for fast windows & lookups

CREATE INDEX IF NOT EXISTS idx_mc_metrics_sym_time
  ON mc_metrics(symbol, as_of DESC, horizon_days);
CREATE INDEX IF NOT EXISTS idx_predictions_symbol_ts ON predictions(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_outcomes_predid ON outcomes(pred_id);
"""

_conn: Optional[duckdb.DuckDBPyConnection] = None


def _ensure_dir() -> str:
    global DB_PATH
    path = pathlib.Path(DB_PATH).expanduser().resolve()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        DB_PATH = str(path)
        return DB_PATH
    except PermissionError:
        fallback = (DATA_ROOT / path.name).resolve()
        fallback.parent.mkdir(parents=True, exist_ok=True)
        logger.warning("Falling back to DuckDB path %s (original %s not writable)", fallback, path)
        DB_PATH = str(fallback)
        return DB_PATH


def get_conn() -> duckdb.DuckDBPyConnection:
    """Singleton connection with a sane thread setting and self-healing schema."""
    global _conn
    if _conn is None:
        db_file = _ensure_dir()

        # Debug log to confirm the actual DB path used (ASCII only for Windows consoles)
        logger.info("DuckDB core DB path: %s", db_file)

        _conn = duckdb.connect(db_file)
        _conn.execute("PRAGMA threads=4;")

        # ---- Self-healing patch ----
        try:
            cols = {r[1] for r in _conn.execute("PRAGMA table_info('predictions')").fetchall()}
            if "issued_at" not in cols:
                logger.info("Adding missing 'issued_at' column on predictions")
                _conn.execute("ALTER TABLE predictions ADD COLUMN issued_at TIMESTAMP")
                if "ts" in cols:
                    _conn.execute("UPDATE predictions SET issued_at = ts WHERE issued_at IS NULL")
            if "features_ref" not in cols:
                logger.info("Adding missing 'features_ref' column on predictions")
                _conn.execute("ALTER TABLE predictions ADD COLUMN features_ref VARCHAR")
            _conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_issued_at ON predictions(issued_at)"
            )
        except Exception as e:
            logger.warning("Schema self-heal failed: %s", e)
        # ----------------------------

    return _conn

def init_schema() -> None:
    global _SCHEMA_READY, _PREDICTIONS_HAS_ISSUED_AT
    con = get_conn()
    con.execute(DDL)
    try:
        con.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS issued_at TIMESTAMP")
        con.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS features_ref TEXT")
        con.execute("UPDATE predictions SET issued_at = COALESCE(issued_at, ts) WHERE issued_at IS NULL")
        con.execute("CREATE INDEX IF NOT EXISTS idx_predictions_issued_at ON predictions(issued_at)")
    except Exception as exc:
        logger.warning("DuckDB compatibility adjustments failed: %s", exc)
    try:
        cols = {row[1] for row in con.execute("PRAGMA table_info('predictions')").fetchall()}
        _PREDICTIONS_HAS_ISSUED_AT = "issued_at" in cols
    except Exception:
        _PREDICTIONS_HAS_ISSUED_AT = False
    _SCHEMA_READY = True


def _ensure_schema() -> None:
    if not _SCHEMA_READY:
        try:
            init_schema()
        except Exception as exc:
            logger.warning("init_schema deferred due to error: %s", exc)


def insert_prediction(row: Dict[str, Any]) -> None:
    """
    row keys:
      pred_id (uuid str), ts (datetime), symbol (str), horizon_d (int),
      model_id (str), prob_up_next (float), p05 (float or None),
      p50 (float or None), p95 (float or None), spot0 (float or None),
      user_ctx (dict|json str|None), run_id (str|None)
    """
    _ensure_schema()
    con = get_conn()
    ctx = row.get("user_ctx")
    if isinstance(ctx, dict):
        row["user_ctx"] = json.dumps(ctx)  # store JSON text
    features_ref = row.get("features_ref")
    if isinstance(features_ref, (dict, list)):
        try:
            row["features_ref"] = json.dumps(features_ref)
        except Exception:
            row["features_ref"] = json.dumps({"value": features_ref})
    issued_at = row.get("issued_at") or row.get("ts")
    row["issued_at"] = issued_at
    placeholders = ",".join(["?"] * 14)
    con.execute(
        f"""
        INSERT OR REPLACE INTO predictions
        (pred_id, ts, issued_at, features_ref, symbol, horizon_d, model_id, prob_up_next, p05, p50, p95, spot0, user_ctx, run_id)
        VALUES ({placeholders})
        """,
        [
            row.get("pred_id"),
            row.get("ts"),
            row.get("issued_at"),
            row.get("features_ref"),
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
    _ensure_schema()
    con = get_conn()
    column = "COALESCE(p.issued_at, p.ts)" if _PREDICTIONS_HAS_ISSUED_AT else "p.ts"
    return con.execute(
        f"""
        SELECT p.pred_id, {column} AS ts, p.symbol, p.horizon_d, p.spot0
        FROM predictions p
        LEFT JOIN outcomes o ON o.pred_id = p.pred_id
        WHERE o.pred_id IS NULL
          AND {column} + (p.horizon_d * INTERVAL 1 DAY) <= now()
        ORDER BY ts
        LIMIT ?
        """,
        [int(limit)],
    ).fetchall()


def recent_predictions(symbol: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
    _ensure_schema()
    con = get_conn()
    column = "COALESCE(issued_at, ts)" if _PREDICTIONS_HAS_ISSUED_AT else "ts"
    q = f"""
      SELECT pred_id, {column} AS issued_at, symbol, horizon_d, model_id,
             prob_up_next, p05, p50, p95, spot0, run_id
      FROM predictions
    """
    args: list = []
    if symbol:
        q += " WHERE symbol = ?"
        args.append(symbol.upper())
    q += " ORDER BY issued_at DESC LIMIT ?"
    args.append(int(limit))
    rows = con.execute(q, args).fetchall()
    cols = [d[0] for d in con.description]
    items = [dict(zip(cols, r)) for r in rows]
    return items
