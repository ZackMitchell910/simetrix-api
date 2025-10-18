"""Quantitative routes for the modular FastAPI application."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from src.core import settings
from src.services.quant_daily import (
    fetch_daily_snapshot,
    fetch_minimal_summary,
)

router = APIRouter(prefix="/quant", tags=["quant"])


@router.get(
    "/daily/today",
    summary="Daily snapshot or minimal summary when symbol is provided",
)
async def quant_daily_today(
    symbol: str | None = Query(None, description="Ticker, e.g., NVDA or BTC-USD"),
    horizon_days: int = Query(30, ge=5, le=90),
) -> dict[str, Any]:
    """Return the cached quant snapshot or a minimal summary for a specific symbol."""
    if symbol:
        sym = symbol.strip().upper()
        if not sym:
            raise HTTPException(status_code=400, detail="Symbol cannot be blank")
        try:
            return await fetch_minimal_summary(sym, horizon_days)
        except HTTPException:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=f"compute failed for {sym}: {exc}") from exc

    # Default horizon mirrors the scheduler configuration when none is supplied.
    hz = int(horizon_days or settings.daily_quant_horizon_days)
    snapshot = await fetch_daily_snapshot(hz)
    if snapshot.get("as_of") is None:
        snapshot["as_of"] = datetime.now(timezone.utc).date().isoformat()
    return snapshot
