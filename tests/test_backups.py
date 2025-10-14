import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infra.backups import create_duckdb_backup


def test_create_duckdb_backup_limits_retention(tmp_path: Path):
    db_file = tmp_path / "test.duckdb"
    db_file.write_bytes(b"duckdb-test")
    backup_dir = tmp_path / "backups"

    first = create_duckdb_backup(db_file, backup_dir, keep=1)
    assert first.exists()
    time.sleep(1)
    second = create_duckdb_backup(db_file, backup_dir, keep=1)
    assert second.exists()

    backups = list(backup_dir.glob("test-*.duckdb"))
    assert len(backups) == 1
    assert backups[0] == second
