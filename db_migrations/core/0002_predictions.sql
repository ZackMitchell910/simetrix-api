-- core: predictions + outcomes tables
CREATE TABLE IF NOT EXISTS predictions (
  pred_id UUID PRIMARY KEY,
  ts TIMESTAMP NOT NULL,
  symbol TEXT NOT NULL,
  horizon_d INTEGER NOT NULL,
  model_id TEXT NOT NULL,
  prob_up_next DOUBLE,
  p05 DOUBLE,
  p50 DOUBLE,
  p95 DOUBLE,
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
  crps DOUBLE,
  brier DOUBLE,
  ece DOUBLE,
  cov_p80 DOUBLE,
  cov_p95 DOUBLE,
  drift_psi DOUBLE,
  n_preds INTEGER,
  PRIMARY KEY (date, model_id, symbol)
);

CREATE INDEX IF NOT EXISTS idx_predictions_symbol_ts ON predictions(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_outcomes_predid ON outcomes(pred_id);
