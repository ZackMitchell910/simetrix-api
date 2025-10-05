# backend/src/labeler.py
from __future__ import annotations

import logging
from typing import Callable, Optional
from datetime import datetime, timedelta, timezone

import numpy as np

# Use the Predictions/Outcomes store (pred_id-keyed)
from .db.duck import matured_predictions_now, insert_outcome

# Optional: mirror into PathPanda Feature Store (run_id-keyed)
try:
    from .feature_store import connect as _pfs_connect, insert_outcome as _pfs_insert_out
except Exception:  # feature_store not required at import time
    _pfs_connect = None
    _pfs_insert_out = None

logger = logging.getLogger(__name__)


def label_mature_predictions(
    fetch_close_fn: Callable[[str, datetime], Optional[float]],
    limit: int = 5000,
) -> int:
    """
    Labels any matured predictions (ts + horizon_d <= now) that are missing outcomes.

    - Reads matured rows from src.db.duck.predictions/outcomes
    - Uses `fetch_close_fn(symbol, target_ts)` to get realized close
    - Writes labels to src.db.duck.outcomes
    - Optionally mirrors each label into PathPanda Feature Store (if available)

    Returns:
        int: count of outcomes labeled.
    """
    try:
        rows = matured_predictions_now(limit=limit)
    except Exception as e:
        logger.exception(f"matured_predictions_now failed: {e}")
        return 0

    labeled = 0

    for pred_id, ts, symbol, horizon_d, spot0 in rows:
        try:
            if spot0 in (None, 0):
                continue

            # Compute target timestamp (UTC)
            ts_utc = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
            ts_utc = ts_utc.replace(tzinfo=timezone.utc)
            target_ts = ts_utc + timedelta(days=int(horizon_d))

            # Fetch realized close (caller handles weekends/holidays)
            realized = fetch_close_fn(symbol, target_ts)
            if realized is None or not np.isfinite(realized):
                continue

            realized = float(realized)
            ret = (realized / float(spot0)) - 1.0
            label_up = bool(ret > 0.0)

            # Write to Outcomes table (DuckDB)
            try:
                insert_outcome(pred_id, target_ts, realized, float(ret), label_up)
                labeled += 1
            except Exception as e:
                logger.warning(f"insert_outcome failed for {pred_id}: {e}")
                continue

            # Optional mirror to PathPanda FS (so metrics/rollups can see realized prices)
            if _pfs_connect and _pfs_insert_out:
                try:
                    con_fs = _pfs_connect()
                    _pfs_insert_out(
                        con_fs,
                        {
                            "run_id": pred_id,          # reuse pred_id as run_id for joinability
                            "symbol": symbol,
                            "realized_at": target_ts,   # datetime
                            "y": realized,              # realized PRICE level
                        },
                    )
                    con_fs.close()
                except Exception as e:
                    logger.warning(f"feature_store outcome mirror failed for {pred_id}: {e}")

        except Exception as e:
            logger.warning(f"labeling failed for pred_id={pred_id}: {e}")

    return labeled
