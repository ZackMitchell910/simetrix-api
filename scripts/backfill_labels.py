from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("backfill-labels")


def _ensure_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill matured predictions by logging realized outcomes.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Maximum number of matured predictions to process (default: 5000).",
    )
    return parser.parse_args()


def main() -> int:
    _ensure_repo_root()
    from src.core import settings
    from src.services.labeling import labeler_pass

    args = _parse_args()
    limit_cfg = int(getattr(settings, "labeling_batch_limit", 5000))
    limit = max(100, min(int(args.limit), limit_cfg))

    logger.info("Starting backfill run at %s (UTC)", datetime.now(timezone.utc).isoformat())
    stats = labeler_pass(limit)
    processed = int(stats.get("processed", 0))
    labeled = int(stats.get("labeled", 0))
    logger.info("Backfill completed: processed=%s labeled=%s limit=%s", processed, labeled, limit)
    print({"processed": processed, "labeled": labeled, "limit": limit})
    return 0


if __name__ == "__main__":
    sys.exit(main())
