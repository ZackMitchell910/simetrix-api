-- feature_store: core metrics + ingest log
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

CREATE INDEX IF NOT EXISTS idx_mc_metrics_symbol_time
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
