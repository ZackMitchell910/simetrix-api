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
    print("predictions table does not exist yet - nothing to rebuild.")
else:
    info = con.execute("PRAGMA table_info('predictions')").fetchall()
    print("Before:", info)
    have = {row[1] for row in info}

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
          run_id        TEXT,
          issued_at     TIMESTAMP,
          features_ref  TEXT
        )
    """)
    # Map old columns -> canonical columns (fill missing with NULLs/defaults)
    pred_id_expr = "COALESCE(pred_id, uuid())" if "pred_id" in have else "uuid()"
    ts_expr = (
        "ts"
        if "ts" in have
        else "COALESCE(issued_at, NOW())" if "issued_at" in have else "NOW()"
    )
    horizon_expr = (
        "COALESCE(horizon_d, 0)"
        if "horizon_d" in have
        else "COALESCE(horizon_days, 0)" if "horizon_days" in have else "0"
    )
    model_expr = (
        "COALESCE(model_id, 'unknown')"
        if "model_id" in have
        else "'unknown'"
    )
    prob_expr = (
        "prob_up_next"
        if "prob_up_next" in have
        else "prob_up" if "prob_up" in have else "NULL"
    )
    q05_expr = "q05" if "q05" in have else "p05" if "p05" in have else "NULL"
    q50_expr = "q50" if "q50" in have else "p50" if "p50" in have else "NULL"
    q95_expr = "q95" if "q95" in have else "p95" if "p95" in have else "NULL"
    issued_expr = (
        "issued_at"
        if "issued_at" in have
        else "ts" if "ts" in have else "NOW()"
    )
    features_expr = "features_ref" if "features_ref" in have else "NULL"
    symbol_expr = "symbol" if "symbol" in have else "'UNKNOWN'"
    run_id_expr = "run_id" if "run_id" in have else "NULL"

    insert_sql = f"""
        INSERT INTO predictions__canon (
          pred_id, ts, symbol, horizon_d, model_id,
          prob_up_next, p05, p50, p95, spot0, user_ctx, run_id,
          issued_at, features_ref
        )
        SELECT
          {pred_id_expr}              AS pred_id,
          {ts_expr}                   AS ts,
          {symbol_expr}               AS symbol,
          {horizon_expr}              AS horizon_d,
          {model_expr}                AS model_id,
          {prob_expr}                 AS prob_up_next,
          {q05_expr}                  AS p05,
          {q50_expr}                  AS p50,
          {q95_expr}                  AS p95,
          NULL                        AS spot0,
          NULL                        AS user_ctx,
          {run_id_expr}               AS run_id,
          {issued_expr}               AS issued_at,
          {features_expr}             AS features_ref
        FROM predictions
    """
    con.execute(insert_sql)
    con.execute("DROP TABLE predictions")
    con.execute("ALTER TABLE predictions__canon RENAME TO predictions")
    con.execute("COMMIT")
    print("After:", con.execute("PRAGMA table_info('predictions')").fetchall())

print("Rebuild predictions to canonical schema: OK")
