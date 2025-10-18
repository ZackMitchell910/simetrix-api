from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from src.core import EXPORT_ROOT

logger = logging.getLogger("simetrix.services.export")

def maybe_upload_to_s3(path: Path) -> Optional[str]:
    bucket = (os.getenv("PT_EXPORT_S3_BUCKET") or "").strip()
    if not bucket:
        return None
    try:
        import boto3  # type: ignore
    except Exception as exc:
        logger.warning("S3 upload skipped (boto3 unavailable): %s", exc)
        return None

    prefix = (os.getenv("PT_EXPORT_S3_PREFIX") or "").strip("/")
    try:
        relative = path.relative_to(EXPORT_ROOT)
        rel_key = relative.as_posix()
    except Exception:
        rel_key = path.name
    key = f"{prefix}/{rel_key}".strip("/") if prefix else rel_key
    try:
        s3 = boto3.client("s3")  # type: ignore[name-defined]
        s3.upload_file(path.as_posix(), bucket, key)
        return f"s3://{bucket}/{key}"
    except Exception as exc:
        logger.warning("S3 upload failed for %s: %s", path, exc)
        return None


__all__ = ["maybe_upload_to_s3"]
