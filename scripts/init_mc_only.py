import os, sys, importlib
sys.path.insert(0, r".\src")
os.environ["PT_DUCKDB_PATH"] = os.environ.get("PT_DUCKDB_PATH") or r".\data\simetrix.duckdb"
duck = importlib.import_module("db.duck")
con  = duck.get_conn()

# --- MC tables (standalone) ---
con.execute("""
CREATE TABLE IF NOT EXISTS mc_params (
  symbol         TEXT PRIMARY KEY,
  mu             DOUBLE,
  sigma          DOUBLE,
  lookback_mu    INTEGER,
  lookback_sigma INTEGER,
  updated_at     TIMESTAMP
)
""")

con.execute("""
CREATE TABLE IF NOT EXISTS mc_metrics (
  as_of          DATE,
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
)
""")

con.execute("""
CREATE INDEX IF NOT EXISTS idx_mc_metrics_sym_time
ON mc_metrics(symbol, as_of DESC, horizon_days)
""")

print("mc tables ready")
