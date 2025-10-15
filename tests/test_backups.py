import sys
import time
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infra.backups import create_duckdb_backup


def _create_sample_db(db_path: Path, values: tuple[int, ...]) -> None:
    with duckdb.connect(str(db_path)) as con:
        con.execute("CREATE OR REPLACE TABLE sample(value INTEGER)")
        con.executemany("INSERT INTO sample VALUES (?)", [(v,) for v in values])


def _read_values(db_path: Path) -> list[int]:
    with duckdb.connect(str(db_path), read_only=True) as con:
        return [row[0] for row in con.execute("SELECT value FROM sample ORDER BY value").fetchall()]


def test_create_duckdb_backup_limits_retention(tmp_path: Path):
    db_file = tmp_path / "test.duckdb"
    backup_dir = tmp_path / "backups"

    _create_sample_db(db_file, (1,))

    first = create_duckdb_backup(db_file, backup_dir, keep=1)
    assert first.exists()
    assert _read_values(first) == [1]

    _create_sample_db(db_file, (1, 2))
    time.sleep(0.05)

    second = create_duckdb_backup(db_file, backup_dir, keep=1)
    assert second.exists()
    assert _read_values(second) == [1, 2]

    backups = sorted(backup_dir.glob("test-*.duckdb"))
    assert backups == [second]
