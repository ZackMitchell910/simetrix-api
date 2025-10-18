-- Ensure feature store predictions table includes new metadata columns.
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS issued_at TIMESTAMP;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS features_ref TEXT;
UPDATE predictions SET issued_at = COALESCE(issued_at, now()) WHERE issued_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_pred_symbol_time ON predictions(symbol, issued_at DESC);
CREATE INDEX IF NOT EXISTS idx_pred_model_time ON predictions(model_id, issued_at DESC);
