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
import logging
from pathlib import Path
from typing import Iterable, Optional, Mapping, Any, List, Tuple

import duckdb
import numpy as np
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional import fallback
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]


def _default_data_root() -> Path:
    env_root = os.getenv("PT_DATA_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    container_root = Path("/data")
    if container_root.exists() and os.access(container_root, os.W_OK):
        return container_root
    fallback = Path(__file__).resolve().parents[2] / "data"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback.resolve()


DATA_ROOT = _default_data_root()
os.environ.setdefault("PT_DATA_ROOT", str(DATA_ROOT))
logger = logging.getLogger(__name__)

ARROW_ROOT = DATA_ROOT / "arrow_store"
ARROW_PREDICTIONS_DIR = ARROW_ROOT / "predictions"
ARROW_OUTCOMES_DIR = ARROW_ROOT / "outcomes"
ARROW_ROOT.mkdir(parents=True, exist_ok=True)
ARROW_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
ARROW_OUTCOMES_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_db_path(path_str: str) -> str:
    p = Path(path_str).expanduser().resolve()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)
    except PermissionError:
        fallback = (DATA_ROOT / p.name).resolve()
        fallback.parent.mkdir(parents=True, exist_ok=True)
        logger.warning("Falling back to feature store path %s (original %s not writable)", fallback, p)
        return str(fallback)


DB_PATH = _ensure_db_path(os.getenv("PT_DB_PATH", str((DATA_ROOT / "pathpanda.duckdb").resolve())))

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

CREATE TABLE IF NOT EXISTS calibration_params (
  symbol        TEXT,
  horizon_days  INTEGER,
  df            DOUBLE,
  loc           DOUBLE,
  scale         DOUBLE,
  q05           DOUBLE,
  q50           DOUBLE,
  q95           DOUBLE,
  sample_n      INTEGER,
  sigma_ann     DOUBLE,
  updated_at    TIMESTAMP DEFAULT now(),
  PRIMARY KEY (symbol, horizon_days)
);

CREATE TABLE IF NOT EXISTS news_articles (
  symbol    TEXT,
  ts        TIMESTAMP,
  source    TEXT,
  title     TEXT,
  url       TEXT,
  summary   TEXT,
  sentiment DOUBLE,
  PRIMARY KEY (symbol, ts, source, url)
);

CREATE INDEX IF NOT EXISTS idx_news_sym_ts
  ON news_articles(symbol, ts);

CREATE TABLE IF NOT EXISTS earnings (
  symbol         TEXT,
  report_date    DATE,
  eps            DOUBLE,
  surprise       DOUBLE,
  revenue        DOUBLE,
  guidance_delta DOUBLE,
  PRIMARY KEY (symbol, report_date)
);

CREATE INDEX IF NOT EXISTS idx_earn_sym
  ON earnings(symbol, report_date);

CREATE TABLE IF NOT EXISTS macro_daily (
  as_of    DATE PRIMARY KEY,
  rff      DOUBLE,
  cpi_yoy  DOUBLE,
  u_rate   DOUBLE
);

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
    """
    Open the Feature Store DB and ensure legacy tables get the columns
    newer code expects (issued_at, features_ref, horizon_days).
    """
    folder = os.path.dirname(DB_PATH) or "."
    os.makedirs(folder, exist_ok=True)
    con = duckdb.connect(DB_PATH)

    # Create tables if first run
    con.execute(DDL)

    # --- Compatibility patch for old files ---
    try:
        cols = {row[1] for row in con.execute("PRAGMA table_info('predictions')").fetchall()}

        if "issued_at" not in cols:
            con.execute("ALTER TABLE predictions ADD COLUMN issued_at TIMESTAMP")
            # backfill based on what's available
            cols2 = {row[1] for row in con.execute("PRAGMA table_info('predictions')").fetchall()}
            if "ts" in cols2:  # some very old schemas had ts
                con.execute("UPDATE predictions SET issued_at = ts WHERE issued_at IS NULL")
            else:
                con.execute("UPDATE predictions SET issued_at = now() WHERE issued_at IS NULL")

        if "features_ref" not in cols:
            con.execute("ALTER TABLE predictions ADD COLUMN features_ref TEXT")

        # Some older builds used horizon_d; keep both but ensure horizon_days is present
        cols = {row[1] for row in con.execute("PRAGMA table_info('predictions')").fetchall()}
        if "horizon_days" not in cols and "horizon_d" in cols:
            con.execute("ALTER TABLE predictions ADD COLUMN horizon_days INTEGER")
            con.execute("UPDATE predictions SET horizon_days = horizon_d WHERE horizon_days IS NULL")

        # Helpful index for common reads
        con.execute("CREATE INDEX IF NOT EXISTS idx_fs_pred_symbol_issued ON predictions(symbol, issued_at DESC)")
    except Exception as exc:
        logger.warning("Feature Store compat patch failed: %s", exc)

    return con

def get_con() -> duckdb.DuckDBPyConnection:
    """Return a live DuckDB connection to the Feature Store and ensure schema compatibility."""
    con = connect()

    try:
        logger.info('Feature Store DB path: %s', DB_PATH)

        cols = {r[1] for r in con.execute("PRAGMA table_info('predictions')").fetchall()}

        if 'issued_at' not in cols:
            logger.info("Adding missing 'issued_at' column to Feature Store predictions table")
            con.execute("ALTER TABLE predictions ADD COLUMN issued_at TIMESTAMP")
            if 'ts' in cols:
                con.execute("UPDATE predictions SET issued_at = ts WHERE issued_at IS NULL")
            else:
                con.execute("UPDATE predictions SET issued_at = now() WHERE issued_at IS NULL")

        if 'features_ref' not in cols:
            logger.info("Adding missing 'features_ref' column to Feature Store predictions table")
            con.execute("ALTER TABLE predictions ADD COLUMN features_ref VARCHAR")

        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_fs_pred_symbol_issued ON predictions(symbol, issued_at DESC)"
        )

    except Exception as exc:
        logger.warning('Feature Store schema check failed: %s', exc)

    return con

def _ensure_predictions_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Backfill legacy predictions table with issued_at and features_ref columns."""
    try:
        columns = {
            row[1]
            for row in con.execute("PRAGMA table_info('predictions')").fetchall()
        }
        if "issued_at" not in columns:
            con.execute("ALTER TABLE predictions ADD COLUMN issued_at TIMESTAMP")
            con.execute(
                "UPDATE predictions SET issued_at = COALESCE(issued_at, now())"
            )
        if "features_ref" not in columns:
            con.execute("ALTER TABLE predictions ADD COLUMN features_ref TEXT")
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_pred_symbol_time ON predictions(symbol, issued_at DESC)"
        )
        con.execute(
            "CREATE INDEX IF NOT EXISTS idx_pred_model_time ON predictions(model_id, issued_at DESC)"
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("ensure_predictions_schema_failed: %s", exc)

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

def upsert_calibration_params(con, rows: list[tuple]) -> int:
    """
    rows: [(symbol, horizon_days, df, loc, scale, q05, q50, q95, sample_n, sigma_ann, updated_at), ...]
    """
    if not rows:
        return 0
    con.executemany(
        """
        INSERT OR REPLACE INTO calibration_params
        (symbol, horizon_days, df, loc, scale, q05, q50, q95, sample_n, sigma_ann, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return con.rowcount

def get_calibration_params(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    horizon_days: int,
) -> dict | None:
    row = con.execute(
        """
        SELECT df, loc, scale, q05, q50, q95, sample_n, sigma_ann, updated_at
        FROM calibration_params
        WHERE symbol = ? AND horizon_days = ?
        LIMIT 1
        """,
        [symbol.upper(), int(horizon_days)],
    ).fetchone()
    if not row:
        return None
    cols = ["df", "loc", "scale", "q05", "q50", "q95", "sample_n", "sigma_ann", "updated_at"]
    return dict(zip(cols, row))

def insert_news(con, rows: list[tuple]) -> int:
    """
    rows: [(symbol, ts, source, title, url, summary, sentiment), ...]
    """
    if not rows:
        return 0
    con.executemany(
        """
        INSERT OR REPLACE INTO news_articles
        (symbol, ts, source, title, url, summary, sentiment)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return con.rowcount

def insert_earnings(con, rows: list[tuple]) -> int:
    """
    rows: [(symbol, report_date, eps, surprise, revenue, guidance_delta), ...]
    """
    if not rows:
        return 0
    con.executemany(
        """
        INSERT OR REPLACE INTO earnings
        (symbol, report_date, eps, surprise, revenue, guidance_delta)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return con.rowcount

def upsert_macro(con, rows: list[tuple]) -> int:
    """
    rows: [(as_of, rff, cpi_yoy, u_rate), ...]
    """
    if not rows:
        return 0
    con.executemany(
        """
        INSERT OR REPLACE INTO macro_daily (as_of, rff, cpi_yoy, u_rate)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
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
    log_predictions([pred])

def log_predictions(preds: Iterable[Mapping[str, Any]]) -> int:
    """Batch upsert for prediction artifacts."""
    preds_list = [dict(p) for p in preds]
    if not preds_list:
        return 0
    for row in preds_list:
        val = row.get("features_ref")
        if isinstance(val, (dict, list)):
            row["features_ref"] = json.dumps(val)
    tuples = [_pred_tuple(row) for row in preds_list]
    con = get_con()
    con.executemany(f"""
        INSERT OR REPLACE INTO predictions ({','.join(_PRED_COLS)})
        VALUES ({','.join(['?']*len(_PRED_COLS))})
    """, tuples)
    count = con.rowcount
    con.close()
    try:
        _append_predictions_arrow(preds_list)
    except Exception as exc:
        logger.debug("Arrow append (predictions) failed: %s", exc)
    return count
def insert_outcome(outcome: Mapping[str, Any]) -> None:
    """Single-row upsert for realized outcome (y)."""
    con = get_con()
    con.execute("""
        INSERT OR REPLACE INTO outcomes (run_id, symbol, realized_at, y)
        VALUES (?, ?, ?, ?)
    """, (outcome["run_id"], outcome["symbol"], outcome["realized_at"], outcome["y"]))
    con.close()
    try:
        _append_outcomes_arrow([outcome])
    except Exception as exc:
        logger.debug("Arrow append (outcomes) failed: %s", exc)

# ---------- Retrieval ----------

def matured_predictions(limit: int = 5000) -> list[tuple]:
    """Matured but unlabeled predictions: (run_id, issued_at, symbol, horizon_days)."""
    con = get_con()
    _ensure_predictions_schema(con)
    has_issued = _predictions_has_issued_at(con)
    issued_expr = "p.issued_at" if has_issued else "p.ts"
    select_issued = f"{issued_expr} AS issued_at"

    query = f"""
        SELECT p.run_id, {select_issued}, p.symbol, p.horizon_days
        FROM predictions p
        LEFT JOIN outcomes o USING (run_id)
        WHERE o.run_id IS NULL
          AND {issued_expr} + (p.horizon_days * INTERVAL 1 DAY) <= now()
        ORDER BY {issued_expr} DESC
        LIMIT ?
    """
    return con.execute(query, [limit]).fetchall()

# ---------- Arrow / Parquet helpers ----------

def _require_pyarrow() -> tuple[Any, Any]:
    if pa is None or pq is None:
        raise RuntimeError("pyarrow is not available; install the pyarrow extra to use this helper")
    return pa, pq


def _predictions_has_issued_at(con: duckdb.DuckDBPyConnection) -> bool:
    """Check whether the predictions table currently exposes issued_at."""
    try:
        cols_now = {row[1] for row in con.execute("PRAGMA table_info('predictions')").fetchall()}
    except Exception:
        return False
    return "issued_at" in cols_now


def _pred_cols_and_order(
    con: duckdb.DuckDBPyConnection,
    base_cols: Sequence[str],
) -> tuple[list[str], str]:
    """
    Build a safe SELECT column list and ORDER BY column for reading predictions.
    Falls back to ts if issued_at is missing (e.g., on transient tables).
    """
    select_cols = list(base_cols)
    order_col = "issued_at"

    if not _predictions_has_issued_at(con):
        select_cols = [c for c in select_cols if c != "issued_at"] + ["ts AS issued_at"]
        order_col = "ts"

    return select_cols, order_col

def _normalize_datetime(value: Any) -> tuple[str, str]:
    if isinstance(value, dt.datetime):
        dt_obj = value.astimezone(dt.timezone.utc)
    else:
        try:
            dt_obj = dt.datetime.fromisoformat(str(value))
            if dt_obj.tzinfo is None:
                dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)
            else:
                dt_obj = dt_obj.astimezone(dt.timezone.utc)
        except Exception:
            dt_obj = dt.datetime.now(dt.timezone.utc)
    return dt_obj.isoformat(), dt_obj.date().isoformat()

def _append_predictions_arrow(rows: Iterable[Mapping[str, Any]]) -> int:
    if pa is None or pq is None:
        return 0
    records: list[dict[str, Any]] = []
    for row in rows:
        rec = {}
        for col in _PRED_COLS:
            if col == "features_ref":
                val = row.get(col)
                if isinstance(val, (dict, list)):
                    rec[col] = json.dumps(val)
                else:
                    rec[col] = val
            else:
                rec[col] = row.get(col)
        issued_iso, issued_day = _normalize_datetime(rec.get("issued_at"))
        rec["issued_at"] = issued_iso
        rec["issued_date"] = issued_day
        records.append(rec)
    if not records:
        return 0
    keys = records[0].keys()
    columns = {k: [rec.get(k) for rec in records] for k in keys}
    table = pa.Table.from_pydict(columns)  # type: ignore[arg-type]
    try:
        pq.write_to_dataset(
            table,
            root_path=ARROW_PREDICTIONS_DIR.as_posix(),
            partition_cols=["issued_date"],
            existing_data_behavior="overwrite_or_ignore",
        )
    except TypeError:
        pq.write_to_dataset(
            table,
            root_path=ARROW_PREDICTIONS_DIR.as_posix(),
            partition_cols=["issued_date"],
        )
    return len(records)

def _append_outcomes_arrow(rows: Iterable[Mapping[str, Any]]) -> int:
    if pa is None or pq is None:
        return 0
    records: list[dict[str, Any]] = []
    for row in rows:
        rec = dict(row)
        realized_iso, realized_day = _normalize_datetime(rec.get("realized_at"))
        rec["realized_at"] = realized_iso
        rec["realized_date"] = realized_day
        records.append(rec)
    if not records:
        return 0
    keys = records[0].keys()
    columns = {k: [rec.get(k) for rec in records] for k in keys}
    table = pa.Table.from_pydict(columns)  # type: ignore[arg-type]
    try:
        pq.write_to_dataset(
            table,
            root_path=ARROW_OUTCOMES_DIR.as_posix(),
            partition_cols=["realized_date"],
            existing_data_behavior="overwrite_or_ignore",
        )
    except TypeError:
        pq.write_to_dataset(
            table,
            root_path=ARROW_OUTCOMES_DIR.as_posix(),
            partition_cols=["realized_date"],
        )
    return len(records)

def fetch_predictions_arrow(
    con: duckdb.DuckDBPyConnection,
    *,
    symbol: str | None = None,
    min_issued_at: dt.datetime | None = None,
    day: dt.date | None = None,
    limit: int = 5000,
) -> "pa.Table":
    """
    Return recent predictions as a PyArrow table directly from DuckDB.
    """
    _require_pyarrow()
    _ensure_predictions_schema(con)
    has_issued = _predictions_has_issued_at(con)
    issued_expr = "issued_at" if has_issued else "ts"

    filters: list[str] = []
    params: list[Any] = []
    if symbol:
        filters.append("symbol = ?")
        params.append(symbol.upper())
    if min_issued_at:
        filters.append(f"{issued_expr} >= ?")
        params.append(min_issued_at)
    if day:
        filters.append(f"date({issued_expr}) = ?")
        params.append(day)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""

    select_cols, order_col = _pred_cols_and_order(con, _PRED_COLS)
    sql = f"""
        SELECT {','.join(select_cols)}
        FROM predictions
        {where}
        ORDER BY {order_col} DESC
        LIMIT {int(max(1, limit))}
    """
    tbl = con.execute(sql, params).fetch_arrow_table()
    if hasattr(tbl, "combine_chunks"):
        tbl = tbl.combine_chunks()
    return tbl

def fetch_outcomes_arrow(
    con: duckdb.DuckDBPyConnection,
    *,
    symbol: str | None = None,
    day: dt.date | None = None,
    limit: int = 5000,
) -> "pa.Table":
    """
    Return recent outcomes as a PyArrow table directly from DuckDB.
    """
    _require_pyarrow()
    filters: list[str] = []
    params: list[Any] = []
    if symbol:
        filters.append("symbol = ?")
        params.append(symbol.upper())
    if day:
        filters.append("date(realized_at) = ?")
        params.append(day)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    sql = f"""
        SELECT run_id, symbol, realized_at, y
        FROM outcomes
        {where}
        ORDER BY realized_at DESC
        LIMIT {int(max(1, limit))}
    """
    tbl = con.execute(sql, params).fetch_arrow_table()
    if hasattr(tbl, "combine_chunks"):
        tbl = tbl.combine_chunks()
    return tbl

def export_predictions_parquet(
    dest: str | Path,
    *,
    symbol: str | None = None,
    min_issued_at: dt.datetime | None = None,
    day: dt.date | None = None,
    limit: int = 5000,
) -> Path:
    """
    Persist recent predictions to a Parquet file using PyArrow.
    """
    _, pq_mod = _require_pyarrow()
    con = get_con()
    table = fetch_predictions_arrow(
        con,
        symbol=symbol,
        min_issued_at=min_issued_at,
        day=day,
        limit=limit,
    )
    path = Path(dest).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    pq_mod.write_table(table, path.as_posix())
    return path


def export_outcomes_parquet(
    dest: str | Path,
    *,
    symbol: str | None = None,
    day: dt.date | None = None,
    limit: int = 5000,
) -> Path:
    """
    Persist outcomes to a Parquet file using PyArrow.
    """
    _, pq_mod = _require_pyarrow()
    con = get_con()
    table = fetch_outcomes_arrow(
        con,
        symbol=symbol,
        day=day,
        limit=limit,
    )
    path = Path(dest).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    pq_mod.write_table(table, path.as_posix())
    return path


def fetch_metrics_daily_arrow(
    con: duckdb.DuckDBPyConnection,
    *,
    day: dt.date | None = None,
    symbol: str | None = None,
    model_id: str | None = None,
    horizon_days: int | None = None,
    limit: int | None = None,
) -> "pa.Table":
    """
    Return metrics_daily rows as a PyArrow table.
    """
    _require_pyarrow()
    filters: list[str] = []
    params: list[Any] = []
    if day:
        filters.append("date = ?")
        params.append(day)
    if symbol:
        filters.append("symbol = ?")
        params.append(symbol.upper())
    if model_id:
        filters.append("model_id = ?")
        params.append(model_id)
    if horizon_days is not None:
        filters.append("horizon_days = ?")
        params.append(int(horizon_days))
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    limit_clause = f"LIMIT {int(limit)}" if limit else ""
    sql = f"""
        SELECT date, model_id, symbol, horizon_days,
               rmse, mae, brier, crps, mape, p80_cov, p90_cov, n
        FROM metrics_daily
        {where}
        ORDER BY date DESC, symbol, horizon_days
        {limit_clause}
    """
    tbl = con.execute(sql, params).fetch_arrow_table()
    if hasattr(tbl, "combine_chunks"):
        tbl = tbl.combine_chunks()
    return tbl


def export_metrics_parquet(
    dest: str | Path,
    *,
    day: dt.date | None = None,
    symbol: str | None = None,
    model_id: str | None = None,
    horizon_days: int | None = None,
    limit: int | None = None,
) -> Path:
    """
    Persist metrics_daily rows to Parquet.
    """
    _, pq_mod = _require_pyarrow()
    con = get_con()
    table = fetch_metrics_daily_arrow(
        con,
        day=day,
        symbol=symbol,
        model_id=model_id,
        horizon_days=horizon_days,
        limit=limit,
    )
    path = Path(dest).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    pq_mod.write_table(table, path.as_posix())
    return path

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
    _ensure_predictions_schema(con)
    has_issued = _predictions_has_issued_at(con)
    issued_expr = "p.issued_at" if has_issued else "p.ts"

    where = [f"CAST({issued_expr} AS DATE) = ?"]
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
    _ensure_predictions_schema(con)
    has_issued = _predictions_has_issued_at(con)
    issued_expr = "p.issued_at" if has_issued else "p.ts"
    where = ["o.y IS NOT NULL"]
    params = []
    if symbol:
        where.append("p.symbol = ?")
        params.append(symbol)
    if min_date:
        where.append(f"{issued_expr} >= ?")
        params.append(min_date)

    rows = con.execute(f"""
        SELECT p.symbol, p.horizon_days, p.q50, o.y
        FROM predictions p
        JOIN outcomes o ON o.run_id = p.run_id
        WHERE {" AND ".join(where)}
        ORDER BY {issued_expr} DESC
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
