from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


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
    Copy the DuckDB file at `db_path` into `target_dir` with a UTC timestamp suffix.

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
    shutil.copy2(src, dest)

    keep = max(1, int(keep))
    _prune_old_backups(target, stem, keep)
    return dest

