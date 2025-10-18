from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb


def _prune_old_backups(folder: Path, stem_prefix: str, keep: int) -> None:
    files = sorted(
        (p for p in folder.glob(f"{stem_prefix}-*.duckdb") if p.is_file()),
        key=lambda p: (p.stat().st_mtime, p.name),
        reverse=True,
    )
    for old in files[keep:]:
        try:
            old.unlink(missing_ok=True)
        except Exception:
            # Best-effort prune; ignore failures so backup succeeds.
            pass


def create_duckdb_backup(db_path: str | Path, target_dir: str | Path, *, keep: int = 7) -> Path:
    """
    Create an online DuckDB backup at `db_path` into `target_dir` with a UTC timestamp suffix.

    Parameters
    ----------
    db_path:
        Path to the DuckDB database file.
    target_dir:
        Directory where backups are written. Will be created if missing.
    keep:
        Number of most recent backups to keep (per database stem). Older backups are trimmed.

    Returns
    -------
    Path to the created backup file.
    """
    src = Path(db_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"DuckDB file not found: {src}")

    target = Path(target_dir).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    stem = src.stem or "duckdb"
    dest = target / f"{stem}-{timestamp}.duckdb"

    # Export the live database to a temporary directory, then import into the
    # destination file. This keeps the primary writer connection online.
    with tempfile.TemporaryDirectory(prefix=f"{stem}-export-") as tmp_dir:
        export_path = Path(tmp_dir) / "export"
        export_path.mkdir(parents=True, exist_ok=True)

        export_sql_path = export_path.as_posix().replace("'", "''")
    with duckdb.connect(str(src)) as source_con:
        try:
            source_con.execute("""
                ALTER TABLE IF EXISTS predictions
                ADD COLUMN IF NOT EXISTS issued_at TIMESTAMP;
            """)
            source_con.execute("""
                ALTER TABLE IF EXISTS predictions
                ADD COLUMN IF NOT EXISTS features_ref VARCHAR;
            """)
        except Exception as e:
            print("⚠️  source_con schema patch skipped:", e)

        # Proceed with export
        source_con.execute(f"EXPORT DATABASE '{export_sql_path}'")

    with duckdb.connect(str(dest)) as dest_con:
        try:
            dest_con.execute("""
                ALTER TABLE IF EXISTS predictions
                ADD COLUMN IF NOT EXISTS issued_at TIMESTAMP;
            """)
            dest_con.execute("""
                ALTER TABLE IF EXISTS predictions
                ADD COLUMN IF NOT EXISTS features_ref VARCHAR;
            """)
        except Exception as e:
            print("⚠️  dest_con schema patch skipped:", e)

        # Proceed with import
        dest_con.execute(f"IMPORT DATABASE '{export_sql_path}'")


    keep = max(1, int(keep))
    _prune_old_backups(target, stem, keep)
    return dest

