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

def cols(name: str):
    return [r[1] for r in con.execute(f"PRAGMA table_info('{name}')").fetchall()]

con.execute("BEGIN")

# 0) If outcomes exists, back it up and drop the original (to release FK)
if table_exists("outcomes"):
    print("Backing up outcomes -> outcomes__bak")
    con.execute("DROP TABLE IF EXISTS outcomes__bak")
    con.execute("CREATE TABLE outcomes__bak AS SELECT * FROM outcomes")
    con.execute("DROP TABLE outcomes")

# 1) Build canonical predictions__canon
if table_exists("predictions"):
    print("Rebuilding predictions -> canonical schema")
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
    # Map old columns -> canonical
    have = set(cols("predictions"))
    # Defaults for missing columns
    sel = """
        SELECT
          COALESCE({pred_id}, uuid())      AS pred_id,
          COALESCE({ts}, NOW())            AS ts,
          {symbol}                         AS symbol,
          COALESCE({horizon}, 0)           AS horizon_d,
          COALESCE(model_id, 'unknown')    AS model_id,
          {prob_up_next}                   AS prob_up_next,
          {p05}                            AS p05,
          {p50}                            AS p50,
          {p95}                            AS p95,
          NULL                             AS spot0,
          NULL                             AS user_ctx,
          {run_id}                         AS run_id,
          COALESCE({issued}, NOW())        AS issued_at,
          {features}                       AS features_ref
        FROM predictions
    """.format(
        pred_id   = "pred_id"      if "pred_id"      in have else "NULL",
        ts        = "issued_at"    if "issued_at"    in have else "ts" if "ts" in have else "NULL",
        symbol    = "symbol"       if "symbol"       in have else "'UNKNOWN'",
        horizon   = "horizon_days" if "horizon_days" in have else "horizon_d" if "horizon_d" in have else "NULL",
        prob_up_next = "prob_up"   if "prob_up"      in have else "prob_up_next" if "prob_up_next" in have else "NULL",
        p05       = "q05"          if "q05"          in have else "p05" if "p05" in have else "NULL",
        p50       = "q50"          if "q50"          in have else "p50" if "p50" in have else "NULL",
        p95       = "q95"          if "q95"          in have else "p95" if "p95" in have else "NULL",
        run_id    = "run_id"       if "run_id"       in have else "NULL",
        issued    = "issued_at"    if "issued_at"    in have else "ts" if "ts" in have else "NULL",
        features  = "features_ref" if "features_ref" in have else "NULL",
    )
    con.execute("INSERT INTO predictions__canon " + sel)
    con.execute("DROP TABLE predictions")
    con.execute("ALTER TABLE predictions__canon RENAME TO predictions")
else:
    print("predictions did not exist; creating empty canonical table")
    con.execute("""
        CREATE TABLE predictions (
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

# 2) Recreate outcomes with FK → predictions(pred_id)
print("Recreating outcomes with FK")
con.execute("""
    CREATE TABLE outcomes (
      pred_id UUID REFERENCES predictions(pred_id),
      realized_ts TIMESTAMP NOT NULL,
      realized_price DOUBLE NOT NULL,
      label_up BOOLEAN NOT NULL,
      PRIMARY KEY (pred_id)
    )
""")

# 3) Restore rows from backup if present (map columns flexibly)
if table_exists("outcomes__bak"):
    have_o = set(cols("outcomes__bak"))
    sel_o = """
        SELECT
          pred_id,
          {rt}  AS realized_ts,
          {rp}  AS realized_price,
          {lu}  AS label_up
        FROM outcomes__bak
    """.format(
        rt = "realized_ts" if "realized_ts" in have_o else "ts" if "ts" in have_o else "NOW()",
        rp = "realized_price" if "realized_price" in have_o else "price" if "price" in have_o else "NULL",
        lu = "label_up" if "label_up" in have_o else "CASE WHEN ret IS NULL THEN NULL WHEN ret>=0 THEN TRUE ELSE FALSE END" if "ret" in have_o else "NULL",
    )
    print("Restoring rows into outcomes")
    con.execute("INSERT INTO outcomes " + sel_o)
    con.execute("DROP TABLE outcomes__bak")

con.execute("COMMIT")
print("Migration complete.")
