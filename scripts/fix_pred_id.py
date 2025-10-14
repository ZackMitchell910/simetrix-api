import os, sys, importlib

# Point to canonical DB
os.environ["PT_DUCKDB_PATH"] = os.environ.get("PT_DUCKDB_PATH") or r".\data\simetrix.duckdb"
os.environ["PT_DB_PATH"]     = os.environ["PT_DUCKDB_PATH"]

# Import duck from src\db\duck.py
sys.path.insert(0, r".\src")
duck = importlib.import_module("db.duck")

con = duck.get_conn()

def table_exists(name: str) -> bool:
    try:
        con.execute(f"SELECT 1 FROM {name} LIMIT 1")
        return True
    except Exception:
        return False

if not table_exists("predictions"):
    print("predictions table does not exist yet — nothing to migrate.")
else:
    print("Before:", con.execute("PRAGMA table_info('predictions')").fetchall())
    # Add pred_id if missing
    cols = {r[1] for r in con.execute("PRAGMA table_info('predictions')").fetchall()}
    if "pred_id" not in cols:
        con.execute("ALTER TABLE predictions ADD COLUMN pred_id UUID")
        print("Added pred_id column")

    # Populate pred_id where NULL
    con.execute("UPDATE predictions SET pred_id = COALESCE(pred_id, uuid())")
    # Ensure uniqueness for FK
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_predictions_pred_id ON predictions(pred_id)")
    print("After:", con.execute("PRAGMA table_info('predictions')").fetchall())

print("Migration: OK")
