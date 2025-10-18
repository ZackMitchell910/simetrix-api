ALTER TABLE predictions ADD COLUMN IF NOT EXISTS issued_at TIMESTAMP;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS features_ref VARCHAR;
CREATE INDEX IF NOT EXISTS idx_predictions_issued_at ON predictions(issued_at);
