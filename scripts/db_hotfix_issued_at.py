# scripts/db_hotfix_issued_at.py
import os, duckdb, sys

def patch(db_path: str) -> None:
    con = duckdb.connect(db_path)
    cols = {r[1] for r in con.execute("PRAGMA table_info('predictions')").fetchall()}

    # Add issued_at if missing and backfill sanely per DB
    if "issued_at" not in cols:
        con.execute("ALTER TABLE predictions ADD COLUMN issued_at TIMESTAMP")
        cols = {r[1] for r in con.execute("PRAGMA table_info('predictions')").fetchall()}
        if "ts" in cols:  # core DB has ts
            con.execute("UPDATE predictions SET issued_at = ts WHERE issued_at IS NULL")
        else:             # FS DB
            con.execute("UPDATE predictions SET issued_at = now() WHERE issued_at IS NULL")
        con.execute("CREATE INDEX IF NOT EXISTS idx_predictions_issued_at ON predictions(issued_at)")

    # Add features_ref if missing
    if "features_ref" not in cols:
        con.execute("ALTER TABLE predictions ADD COLUMN features_ref TEXT")

    # FS-only compatibility for horizon
    if "horizon_days" not in cols and "horizon_d" in cols:
        con.execute("ALTER TABLE predictions ADD COLUMN horizon_days INTEGER")
        con.execute("UPDATE predictions SET horizon_days = horizon_d WHERE horizon_days IS NULL")

    con.close()
    print(f"Patched: {db_path}")

if __name__ == "__main__":
    core_db = os.getenv("PT_DUCKDB_PATH", "/data/pt.duckdb")
    fs_db   = os.getenv("PT_DB_PATH", "/data/pathpanda.duckdb")
    for db in [core_db, fs_db]:
        try:
            patch(db)
        except Exception as e:
            print(f"Failed to patch {db}: {e}", file=sys.stderr)
            sys.exit(1)
