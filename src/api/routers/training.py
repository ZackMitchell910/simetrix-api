# src/api/routers/training.py
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Security

from src.api.auth import require_key
from src.api.deps import get_redis
from src.services.inference import get_ensemble_prob, get_ensemble_prob_light
from src.services.training import _ensure_trained_models

router = APIRouter(tags=["training"])


@router.post("/train", summary="Warm/Train cached models for a symbol")
async def train(
    payload: dict[str, Any],
    _ok: bool = Security(require_key, scopes=["simulate"]),
) -> dict[str, Any]:
    symbol = str(payload.get("symbol") or "").strip().upper()
    lookback_days = int(payload.get("lookback_days") or 365)
    if not symbol:
        raise HTTPException(status_code=422, detail="symbol is required")

    await _ensure_trained_models(symbol, required_lookback=lookback_days)
    return {"ok": True, "symbol": symbol, "lookback_days": lookback_days}


@router.post("/predict", summary="Fast probability 'up next bar'")
async def predict(
    payload: dict[str, Any],
    _ok: bool = Security(require_key, scopes=["simulate"]),
    redis=Depends(get_redis),
) -> dict[str, Any]:
    symbol = str(payload.get("symbol") or "").strip().upper()
    horizon_days = int(payload.get("horizon_days") or 1)
    if not symbol:
        raise HTTPException(status_code=422, detail="symbol is required")

    try:
        prob = await get_ensemble_prob(symbol, redis, horizon_days)
    except Exception:
        prob = await get_ensemble_prob_light(symbol, redis, horizon_days)

    return {"symbol": symbol, "horizon_days": horizon_days, "prob_up_next": float(prob)}
