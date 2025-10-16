from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List

import duckdb


logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_MIGRATIONS_ROOT = _REPO_ROOT / "db_migrations"


def _default_data_root() -> Path:
    env_root = (os.getenv("PT_DATA_ROOT") or "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    container_root = Path("/data")
    if container_root.exists() and os.access(container_root, os.W_OK):
        return container_root
    fallback = (_REPO_ROOT / "data").resolve()
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _normalized_db_path(db_path: str) -> str:
    """
    Expand environment variables and ensure the parent directory exists.
    Supports relative paths by resolving them against the repository root.
    """
    raw = os.path.expandvars(db_path)
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        fallback_root = _default_data_root()
        fallback_root.mkdir(parents=True, exist_ok=True)
        fallback = (fallback_root / p.name).resolve()
        fallback.parent.mkdir(parents=True, exist_ok=True)
        logger.warning("Path %s not writable; using %s instead", p, fallback)
        return str(fallback)
    return str(p)


def _ensure_migrations_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
          target     TEXT,
          version    TEXT,
          applied_at TIMESTAMP DEFAULT now(),
          PRIMARY KEY (target, version)
        )
        """
    )


def _load_sql(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def run_migrations(target: str, db_path: str) -> List[str]:
    """
    Execute pending migrations for a given target (subdirectory under db_migrations).

    Returns the list of migration versions that were applied.
    """
    mig_dir = _MIGRATIONS_ROOT / target
    if not mig_dir.exists():
        logger.debug("Migration directory %s missing; skipping", mig_dir)
        return []

    files = sorted(p for p in mig_dir.glob("*.sql") if p.is_file())
    if not files:
        return []

    db_file = _normalized_db_path(db_path)
    con = duckdb.connect(db_file)
    try:
        _ensure_migrations_table(con)
        applied = {
            row[0]
            for row in con.execute(
                "SELECT version FROM schema_migrations WHERE target = ?",
                [target],
            ).fetchall()
        }

        executed: List[str] = []
        for path in files:
            version = path.stem
            if version in applied:
                continue
            sql = _load_sql(path)
            con.execute("BEGIN")
            try:
                con.execute(sql)
                con.execute(
                    "INSERT INTO schema_migrations (target, version) VALUES (?, ?)",
                    [target, version],
                )
                con.execute("COMMIT")
                executed.append(version)
                logger.info("Applied %s migration %s", target, version)
            except Exception:
                con.execute("ROLLBACK")
                logger.exception("Migration %s/%s failed; rolled back", target, version)
                raise
        return executed
    finally:
        con.close()


def run_all() -> Dict[str, List[str]]:
    """
    Convenience helper to execute migrations for both the core analytics
    DuckDB and the feature store (if available).
    """
    applied: Dict[str, List[str]] = {}

    try:
        from .duck import DB_PATH as core_db_path  # local import to avoid cycles

        executed = run_migrations("core", core_db_path)
        if executed:
            applied["core"] = executed
    except Exception:
        logger.exception("Failed to run core migrations")

    try:
        from ..feature_store import DB_PATH as fs_db_path  # type: ignore

        executed = run_migrations("feature_store", fs_db_path)
        if executed:
            applied["feature_store"] = executed
    except Exception:
        logger.exception("Failed to run feature_store migrations")

    return applied
