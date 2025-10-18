from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query, Request

from src.api.auth import require_key
from src.services import analytics as analytics_service
from src.services import usage as usage_service


router = APIRouter(tags=["core"])

@router.get("/accuracy-statements")
async def get_accuracy_statements(
    limit: int = Query(50, ge=1, le=200),
    _ok: bool = Depends(require_key),
):
    return analytics_service.accuracy_statements(limit=limit)


@router.get("/me/limits", summary="Return caller plan and usage/limits for key scopes")
async def me_limits(request: Request, _ok: bool = Depends(require_key)):
    return await usage_service.me_limits(request)


@router.post("/analytics/export/predictions")
async def analytics_export_predictions(
    day: Optional[str] = Query(None, description="Date YYYY-MM-DD; defaults to today"),
    symbol: Optional[str] = Query(None),
    limit: int = Query(50_000, ge=100, le=500_000),
    push_to_s3: bool = Query(False),
    _ok: bool = Depends(require_key),
):
    return analytics_service.export_predictions(
        day=day,
        symbol=symbol,
        limit=limit,
        push_to_s3=push_to_s3,
    )


@router.post("/analytics/export/outcomes")
async def analytics_export_outcomes(
    day: Optional[str] = Query(None, description="Date YYYY-MM-DD; defaults to today"),
    symbol: Optional[str] = Query(None),
    limit: int = Query(50_000, ge=100, le=500_000),
    push_to_s3: bool = Query(False),
    _ok: bool = Depends(require_key),
):
    return analytics_service.export_outcomes(
        day=day,
        symbol=symbol,
        limit=limit,
        push_to_s3=push_to_s3,
    )


@router.get("/analytics/metrics/summary")
async def analytics_metrics_summary(
    symbol: Optional[str] = Query(None),
    horizon_days: Optional[int] = Query(None),
    limit: int = Query(90, ge=1, le=365),
    _ok: bool = Depends(require_key),
):
    return analytics_service.metrics_summary(
        symbol=symbol,
        horizon_days=horizon_days,
        limit=limit,
    )
