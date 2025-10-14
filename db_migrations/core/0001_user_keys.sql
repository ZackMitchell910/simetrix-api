-- core: auth user keys
CREATE SEQUENCE IF NOT EXISTS user_keys_id_seq START 1;

CREATE TABLE IF NOT EXISTS user_keys (
  id             BIGINT PRIMARY KEY DEFAULT nextval('user_keys_id_seq'),
  email          TEXT NOT NULL UNIQUE,
  hashed_secret  TEXT NOT NULL,
  plan           TEXT NOT NULL DEFAULT 'free',
  scopes         TEXT NOT NULL DEFAULT '',
  is_active      BOOLEAN NOT NULL DEFAULT TRUE,
  created_at     TIMESTAMP NOT NULL DEFAULT now(),
  updated_at     TIMESTAMP NOT NULL DEFAULT now(),
  last_login     TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_user_keys_plan ON user_keys(plan);
