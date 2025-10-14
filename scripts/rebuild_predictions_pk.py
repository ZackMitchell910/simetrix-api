import os, sys, importlib

# Use your src\db\duck.py
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
    cols = [r[1] for r in con.execute("PRAGMA table_info('predictions')").fetchall()]
    print("Before:", cols)

    # Build new table with pred_id as PRIMARY KEY
    con.execute("BEGIN")
    con.execute("""
        CREATE TABLE predictions__new (
          pred_id      UUID PRIMARY KEY,
          run_id       VARCHAR,
          model_id     VARCHAR,
          symbol       TEXT,
          issued_at    TIMESTAMP,
          horizon_days INTEGER,
          yhat_mean    DOUBLE,
          prob_up      DOUBLE,
          q05          DOUBLE,
          q50          DOUBLE,
          q95          DOUBLE,
          uncertainty  DOUBLE,
          features_ref VARCHAR
        )
    """)
    # Copy data, generating pred_id where missing
    # Map columns that exist; absent ones are filled with NULLs
    sel_cols = ", ".join([
        "COALESCE(pred_id, uuid()) AS pred_id",
        "run_id", "model_id", "symbol", "issued_at", "horizon_days",
        "yhat_mean", "prob_up", "q05", "q50", "q95", "uncertainty", "features_ref"
    ])
    con.execute(f"INSERT INTO predictions__new SELECT {sel_cols} FROM predictions")
    con.execute("DROP TABLE predictions")
    con.execute("ALTER TABLE predictions__new RENAME TO predictions")
    con.execute("COMMIT")

    print("After:", con.execute("PRAGMA table_info('predictions')").fetchall())

print("Rebuild predictions with PK: OK")
