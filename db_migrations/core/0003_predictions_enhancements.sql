-- Ensure predictions table retains extended metadata used by new fan chart + caching.
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS issued_at TIMESTAMP;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS features_ref TEXT;

-- Backfill issued_at with existing timestamps so downstream queries continue to work.
UPDATE predictions SET issued_at = COALESCE(issued_at, ts) WHERE issued_at IS NULL;

-- Helpful index for lookups by issued_at (used by recent_predictions, etc.).
CREATE INDEX IF NOT EXISTS idx_predictions_issued_at ON predictions(issued_at);
