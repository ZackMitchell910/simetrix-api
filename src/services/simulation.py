from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple
from uuid import uuid4

import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, Field, ConfigDict, model_validator
from redis.asyncio import Redis

from src.core import (
    REDIS,
    settings,
    SIM_DISPATCHER,
    _MODEL_META_CACHE,
)
try:
    from src.db.duck import insert_prediction
except Exception:  # pragma: no cover - optional import
    insert_prediction = None  # type: ignore[assignment]
from src.feature_store import connect as fs_connect, log_prediction as fs_log_prediction
from src.model_registry import get_active_model_version
from src.services.inference import (
    get_ensemble_prob,
    get_ensemble_prob_light,
    winsorize,
    sigma_from_calibration,
    compute_fallback_quantiles,
    get_calibration_params,
)
from src.services import ingestion as ingestion_service

logger = logging.getLogger("simetrix.services.simulation")


class SimRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    symbol: str
    mode: Literal["quick", "deep"] = Field(default="quick")
    horizon_days: int = Field(default=30, ge=1, le=3650)
    paths: int = Field(2000, alias="n_paths", ge=100, le=200_000)
    timespan: Literal["day", "hour", "minute"] = "day"
    include_news: bool = False
    include_options: bool = False
    include_futures: bool = False
    x_handles: list[str] = Field(default_factory=list, description="Optional X/Twitter handles to bias sentiment fetch")
    seed: Optional[int] = None

    @model_validator(mode="before")
    def _coerce_handles(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        raw = (
            values.get("x_handles")
            or values.get("handles")
            or values.get("xHandles")
            or values.get("x_handles")
        )
        if raw is None:
            return values
        handles: list[str] = []
        if isinstance(raw, str):
            handles = [h.strip() for h in raw.split(",") if h and h.strip()]
        elif isinstance(raw, (list, tuple, set)):
            handles = [str(h).strip() for h in raw if str(h).strip()]
        else:
            txt = str(raw).strip()
            if txt:
                handles = [txt]
        values["x_handles"] = handles
        return values

    @property
    def n_paths(self) -> int:
        return self.paths

    def lookback_days(self) -> int:
        return 180 if self.mode == "quick" else 3650

    @property
    def handles(self) -> List[str]:
        return list(self.x_handles)

    def bars_per_day(self) -> int:
        if self.timespan == "day":
            return 1
        if self.timespan == "hour":
            return 24
        return 390

    def young_threshold_bars(self) -> int:
        return 126 * self.bars_per_day()


class RunState(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    status: Literal["queued", "running", "done", "error"] = "queued"
    progress: float = 0.0
    symbol: str | None = None
    horizon_days: int | None = None
    paths: int | None = None
    startedAt: str | None = None
    finishedAt: str | None = None
    error: str | None = None
    status_detail: str | None = None
    owner: str | None = None


async def _ensure_run(run_id: str) -> RunState:
    if not REDIS:
        raise HTTPException(status_code=503, detail="redis_unavailable")
    raw = await REDIS.get(f"run:{run_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="run_not_found")
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    try:
        data = json.loads(raw)
        return RunState.model_validate(data)
    except Exception as exc:
        logger.debug("Failed to parse run state for %s: %s", run_id, exc)
        raise HTTPException(status_code=500, detail="run_state_corrupt") from exc


async def _update_run_state(
    redis: Redis,
    run_id: str,
    rs: RunState,
    *,
    status: str | None = None,
    progress: float | None = None,
    detail: str | None = None,
    error: str | None = None,
) -> None:
    if status is not None:
        rs.status = status
    if progress is not None:
        rs.progress = float(progress)
    if detail is not None:
        rs.status_detail = detail
    if error is not None:
        rs.error = error
    await redis.setex(f"run:{run_id}", settings.run_ttl_seconds, rs.model_dump_json())


async def run_simulation(run_id: str, req: SimRequest, redis: Redis) -> None:
    from src import predictive_api
    predictive_api.REDIS = redis
    if not hasattr(predictive_api, '_ensure_run'):
        predictive_api._ensure_run = _ensure_run
    if not hasattr(predictive_api, '_update_run_state'):
        predictive_api._update_run_state = _update_run_state
    await predictive_api.run_simulation(run_id, req, redis)


async def list_models(redis: Redis | None = None) -> List[str]:
    redis = redis or REDIS
    if not redis:
        return []
    out: List[str] = []
    try:
        async for key in redis.scan_iter(match="model:*", count=500):
            out.append(key.split(":", 1)[1] if ":" in key else key)
    except Exception as exc:
        logger.info("_list_models scan failed: %s", exc)
    return out


async def log_prediction_to_stores(
    symbol: str,
    horizon_days: int,
    prob_up: float,
    feature_payload: dict[str, Any],
    calibration_hint: dict[str, Any] | None,
) -> dict[str, Any]:
    pred_id = str(uuid4())
    spot0 = float(feature_payload.get("spot0", 0.0))
    q05 = calibration_hint.get("q05") if calibration_hint else None
    q50 = calibration_hint.get("q50") if calibration_hint else None
    q95 = calibration_hint.get("q95") if calibration_hint else None

    if callable(insert_prediction):
        try:
            insert_prediction(
                {
                    "pred_id": pred_id,
                    "ts": datetime.utcnow(),
                    "symbol": symbol,
                    "horizon_d": horizon_days,
                    "model_id": "ensemble-v1",
                    "prob_up_next": float(prob_up),
                    "p05": q05,
                    "p50": q50,
                    "p95": q95,
                    "spot0": spot0,
                    "user_ctx": {"ui": "pathpanda"},
                }
            )
        except Exception as exc:
            logger.exception("DuckDB insert_prediction failed: %s", exc)

    try:
        con_fs = fs_connect()
        features_payload = {
            "mom_20": float(feature_payload.get("mom_20", 0.0)),
            "rvol_20": float(feature_payload.get("rvol_20", 0.0)),
            "autocorr_5": float(feature_payload.get("autocorr_5", 0.0)),
            "spot0": spot0,
        }
        if calibration_hint:
            features_payload["calibration"] = {
                "sigma_ann": float(calibration_hint.get("sigma_ann")) if calibration_hint.get("sigma_ann") is not None else None,
                "sample_n": int(calibration_hint.get("sample_n", 0)),
            }
        fs_log_prediction(
            con_fs,
            {
                "run_id": pred_id,
                "model_id": "ensemble-v1",
                "symbol": symbol,
                "issued_at": datetime.now(timezone.utc).isoformat(),
                "horizon_days": horizon_days,
                "yhat_mean": None,
                "prob_up": float(prob_up),
                "q05": q05,
                "q50": q50,
                "q95": q95,
                "uncertainty": None,
                "features_ref": json.dumps(features_payload),
            },
        )
        con_fs.close()
    except Exception as exc:
        logger.warning("Feature store mirror failed: %s", exc)

    return {
        "pred_id": pred_id,
        "symbol": symbol,
        "prob_up_next": float(prob_up),
        "p05": q05,
        "p50": q50,
        "p95": q95,
        "spot0": spot0,
    }


__all__ = [
    "SimRequest",
    "RunState",
    "run_simulation",
    "list_models",
    "log_prediction_to_stores",
    "_update_run_state",
    "_ensure_run",
]
