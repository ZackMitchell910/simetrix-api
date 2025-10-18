from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, Security
from fastapi.responses import StreamingResponse

from src.api.auth import require_key
from src.api.deps import get_redis, get_settings
from src.services import admin as admin_service


router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/backup", summary="Trigger an immediate DuckDB backup")
async def backup(_ok: bool = Security(require_key, scopes=["admin"])):
    results = await admin_service.backup_once()
    return {"ok": True, "results": results}


@router.post("/cron/daily", summary="Run daily label + online learn")
async def cron_daily(
    request: Request,
    n: int = 20,
    steps: int = 50,
    batch: int = 32,
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    return await admin_service.cron_daily(request, n=n, steps=steps, batch=batch)


@router.post("/validate/mc", summary="Run roll-forward validation and persist tuning")
async def validate_mc(
    days: int = Query(30, ge=1, le=365),
    symbols: Optional[str] = Query(None, description="Comma-separated, defaults to watchlist"),
    n_paths: int = Query(4000, ge=500, le=200_000),
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    syms = [s.strip() for s in (symbols.split(",") if symbols else ["SPY", "QQQ", "BTC-USD", "ETH-USD"]) if s.strip()]
    out = await admin_service.validate_mc_paths(syms, days=days, n_paths=n_paths)
    return {"ok": True, "data": out}


@router.post("/fetch/news", summary="Fetch recent news and upsert into feature store")
async def fetch_news(
    request: Request,
    symbol: str = Query(..., min_length=1),
    days: int = Query(7, ge=1, le=30),
    limit: int = Query(60, ge=1, le=200),
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    return await admin_service.fetch_news(request, symbol=symbol, days=days, limit=limit)


@router.post("/score/news", summary="LLM-score recent news sentiment")
async def score_news(
    request: Request,
    symbol: str = Query(..., min_length=1),
    days: int = Query(7, ge=1, le=30),
    batch: int = Query(16, ge=1, le=64),
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    return await admin_service.score_news(request, symbol=symbol, days=days, batch=batch)


@router.post("/fetch/earnings", summary="Fetch earnings data and upsert into feature store")
async def fetch_earnings(
    request: Request,
    symbol: str = Query(..., min_length=1),
    lookback_days: int = Query(370, ge=30, le=1825),
    limit: int = Query(12, ge=1, le=40),
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    return await admin_service.fetch_earnings(
        request,
        symbol=symbol,
        lookback_days=lookback_days,
        limit=limit,
    )


@router.post("/fetch/macro", summary="Fetch macro snapshot and upsert into feature store")
async def fetch_macro(
    request: Request,
    provider: Optional[str] = Query(None),
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    return await admin_service.fetch_macro(request, provider=provider)


@router.post("/ingest/backfill")
async def ingest_backfill(days: int = 7, _ok: bool = Security(require_key, scopes=["admin"])):
    return await admin_service.ingest_backfill(days=days)


@router.post("/plan/set", summary="Set subscription plan for an API key (free|pro|inst)")
async def plan_set(
    api_key: str = Body(..., embed=True),
    plan: str = Body(..., embed=True),
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    return await admin_service.plan_set(api_key, plan)


@router.get("/plan/get", summary="Get subscription plan for an API key")
async def plan_get(api_key: str = Query(...), _ok: bool = Security(require_key, scopes=["admin"])):
    return await admin_service.plan_get(api_key)


@router.get("/logs/latest", summary="Fetch recent service logs (in-memory buffer)")
async def logs_latest(n: int = Query(200, ge=1, le=2000), _ok: bool = Security(require_key, scopes=["admin"])):
    return await admin_service.logs_latest(n=n)


@router.post("/ingest/daily")
async def ingest_daily(
    d: Optional[str] = Query(None, description="YYYY-MM-DD (UTC)"),
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    try:
        as_of = date.fromisoformat(d) if d else datetime.now(timezone.utc).date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid 'd' format; expected YYYY-MM-DD")
    return await admin_service.ingest_daily(as_of)


@router.get("/healthz", summary="Liveness")
def healthz():
    return {"ok": True}


@router.get("/health", summary="Readiness/diagnostics")
async def health(request: Request, redis=Depends(get_redis)):
    try:
        alive = await redis.ping()
    except Exception:
        alive = False
    settings = get_settings()
    polygon_key_present = bool((settings.polygon_key or "").strip())
    base = await admin_service.system_health()
    base.update(
        {
            "redis_ok": bool(alive),
            "redis_url": settings.redis_url,
            "polygon_key_present": polygon_key_present,
            "as_of": datetime.now(timezone.utc).isoformat(),
        }
    )
    return base


@router.get("/reports/system-health")
async def report_system_health(request: Request, _ok: bool = Security(require_key, scopes=["admin"])):
    return await admin_service.report_system_health(request)


@router.get("/reports/simulations")
async def report_simulations(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    symbol: Optional[str] = Query(None),
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    return await admin_service.report_simulations(request, limit=limit, symbol=symbol)


@router.get("/reports/model-training")
async def report_model_training(
    request: Request,
    limit: int = Query(200, ge=10, le=1000),
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    return await admin_service.report_model_training(request, limit=limit)


@router.get("/reports/usage")
async def report_usage(
    request: Request,
    scope: str = Query("simulate"),
    sample: int = Query(20, ge=1, le=200),
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    return await admin_service.report_usage(request, scope=scope, sample=sample)


@router.get("/reports/telemetry", response_class=StreamingResponse)
async def report_telemetry(
    request: Request,
    limit: int = Query(200, ge=10, le=1000),
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    return await admin_service.report_telemetry(request, limit=limit)
