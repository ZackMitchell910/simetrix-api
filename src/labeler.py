# backend/src/labeler.py
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Callable

from feature_store import connect, insert_outcome

logger = logging.getLogger(__name__)

def _to_utc_aware(dt_obj: datetime) -> datetime:
    """DuckDB TIMESTAMPs are tz-naive; treat them as UTC and return tz-aware."""
    if dt_obj.tzinfo is None:
        return dt_obj.replace(tzinfo=timezone.utc)
    return dt_obj.astimezone(timezone.utc)

def label_mature_predictions(
    fetch_realized_fn: Callable[[str, datetime], float],
    *,
    max_per_pass: int = 50,
) -> int:
    """
    Label matured predictions:
      - A prediction is mature if (issued_at + horizon_days) <= now (UTC).
      - Writes outcomes.y as the *return* from issue->cutoff: (p1 - p0) / p0.
      - Skips rows on provider errors (e.g., 429) instead of crashing.
      - Returns the number of rows labeled this pass.
    """
    # Read candidates
    con = connect()
    rows = con.execute(
        """
        SELECT run_id, symbol, issued_at, horizon_days
        FROM predictions
        WHERE run_id NOT IN (SELECT run_id FROM outcomes)
        ORDER BY issued_at ASC
        """
    ).fetchall()
    con.close()

    now_utc = datetime.now(timezone.utc)
    labeled = 0
    processed = 0

    for run_id, symbol, issued_at, horizon_days in rows:
        if processed >= max_per_pass:
            break
        processed += 1

        issued_utc = _to_utc_aware(issued_at)
        cutoff = issued_utc + timedelta(days=int(horizon_days or 0))
        if cutoff > now_utc:
            continue

        # Fetch prices (best-effort; skip on failure)
        try:
            p0 = float(fetch_realized_fn(symbol, issued_utc))
            p1 = float(fetch_realized_fn(symbol, cutoff))
            if p0 <= 0.0:
                raise ValueError("issue price <= 0")
            ret = (p1 - p0) / p0
        except Exception as e:
            logger.warning(f"Label skip [{symbol} {run_id}] at {cutoff.date()}: fetch failed: {e}")
            continue

        # Write outcome
        con = connect()
        insert_outcome(
            con,
            {
                "run_id": run_id,
                "symbol": symbol,
                "realized_at": cutoff,  # tz-aware; DuckDB stores as naive-UTC
                "y": float(ret),
            },
        )
        con.close()
        labeled += 1

    logger.info(f"Label pass complete: processed={processed}, labeled={labeled}")
    return labeled
