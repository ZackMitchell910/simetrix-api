import os, sys, importlib
sys.path.insert(0, r".\src")
os.environ["PT_DUCKDB_PATH"] = os.environ.get("PT_DUCKDB_PATH") or r".\data\simetrix.duckdb"
os.environ["PT_DB_PATH"]     = os.environ["PT_DUCKDB_PATH"]
duck = importlib.import_module("db.duck")
con  = duck.get_conn()

def table_exists(name: str) -> bool:
    try:
        con.execute(f"SELECT 1 FROM {name} LIMIT 1")
        return True
    except Exception:
        return False

if not table_exists("predictions"):
    print("predictions table does not exist yet — nothing to rebuild.")
else:
    print("Before:", con.execute("PRAGMA table_info('predictions')").fetchall())
    con.execute("BEGIN")
    # Canonical schema expected by duck.py
    con.execute("""
        CREATE TABLE predictions__canon (
          pred_id       UUID PRIMARY KEY,
          ts            TIMESTAMP NOT NULL,
          symbol        TEXT NOT NULL,
          horizon_d     INTEGER NOT NULL,
          model_id      TEXT NOT NULL,
          prob_up_next  DOUBLE,
          p05           DOUBLE,
          p50           DOUBLE,
          p95           DOUBLE,
          spot0         DOUBLE,
          user_ctx      JSON,
          run_id        TEXT
        )
    """)
    # Map old columns -> canonical columns (fill missing with NULLs/defaults)
    con.execute("""
        INSERT INTO predictions__canon (
          pred_id, ts, symbol, horizon_d, model_id,
          prob_up_next, p05, p50, p95, spot0, user_ctx, run_id
        )
        SELECT
          COALESCE(pred_id, uuid())         AS pred_id,
          COALESCE(issued_at, NOW())        AS ts,
          symbol,
          COALESCE(horizon_days, 0)         AS horizon_d,
          COALESCE(model_id, 'unknown')     AS model_id,
          prob_up                            AS prob_up_next,
          q05                                AS p05,
          q50                                AS p50,
          q95                                AS p95,
          NULL                               AS spot0,
          NULL                               AS user_ctx,
          run_id
        FROM predictions
    """)
    con.execute("DROP TABLE predictions")
    con.execute("ALTER TABLE predictions__canon RENAME TO predictions")
    con.execute("COMMIT")
    print("After:", con.execute("PRAGMA table_info('predictions')").fetchall())

print("Rebuild predictions to canonical schema: OK")
