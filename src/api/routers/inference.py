# src/api/routers/inference.py
from __future__ import annotations
import asyncio, json
from uuid import uuid4
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request, Security
from fastapi.responses import StreamingResponse, JSONResponse

from src.api.deps import get_dispatcher, get_redis, get_settings
from src.api.auth import require_key
from src.quotas import enforce_limits, SIM_LIMIT_PER_MIN  # same enforcement used today
from src.feature_store import connect as fs_connect, get_mc_metrics  # read-only for metrics
from src.services import simulation as simulation_service

router = APIRouter(tags=["inference"])

@router.post("/simulate", summary="Start a simulation run")
async def simulate(
    req_body: dict[str, Any],
    request: Request,
    _ok: bool = Security(require_key, scopes=["simulate"]),
    redis = Depends(get_redis),
    dispatcher = Depends(get_dispatcher),
):
    SimRequest, RunState = simulation_service.SimRequest, simulation_service.RunState
    try:
        req = SimRequest.model_validate(req_body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid request: {e}")

    # Enforce quotas (identical behavior)
    enforce_limits(redis, request, scope="simulate",
                   per_min=SIM_LIMIT_PER_MIN,
                   cost_units=(3 if (req.mode or "quick").lower() == "deep" else 1))

    # Persist initial RunState
    run_id = uuid4().hex
    rs = RunState(run_id=run_id,
                  status="queued",
                  progress=0.0,
                  symbol=req.symbol,
                  horizon_days=int(req.horizon_days),
                  paths=int(req.paths),
                  startedAt=None,
                  status_detail="Queued")
    await redis.setex(f"run:{run_id}", get_settings().run_ttl_seconds, rs.model_dump_json())

    async def _runner():
        await simulation_service.run_simulation(run_id, req, redis)

    # Queue vs. fire-and-forget
    if dispatcher:
        dispatcher.submit(_runner)
    else:
        asyncio.create_task(_runner())

    return {"run_id": run_id}

@router.get("/simulate/{run_id}/state", summary="Get full run state")
async def sim_state(run_id: str, _ok: bool = Security(require_key, scopes=["simulate"])):
    rs = await simulation_service._ensure_run(run_id)
    return rs.model_dump()

@router.get("/simulate/{run_id}/artifact", summary="Get final artifact")
async def sim_artifact(run_id: str, _ok: bool = Security(require_key, scopes=["simulate"]), redis = Depends(get_redis)):
    raw = await redis.get(f"artifact:{run_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="Artifact not found")
    txt = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
    try:
        return json.loads(txt)
    except Exception:
        return JSONResponse(content={"raw": txt})

@router.get("/simulate/{run_id}/stream", summary="SSE: progress updates")
async def sim_stream(run_id: str, _ok: bool = Security(require_key, scopes=["simulate"])) -> StreamingResponse:
    async def gen() -> AsyncIterator[bytes]:
        last = None
        while True:
            try:
                rs = await simulation_service._ensure_run(run_id)
            except HTTPException as e:
                yield f"data: {json.dumps({'error': e.detail})}\n\n".encode("utf-8")
                break
            payload = rs.model_dump()
            if payload != last:
                last = payload
                yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")
            if rs.status in {"done","error"}:
                break
            await asyncio.sleep(0.35)
    return StreamingResponse(gen(), media_type="text/event-stream")

@router.get("/v1/metrics/mc", summary="Latest Monte‑Carlo metrics (feature store)")
def mc_metrics(symbol: str | None = None, limit: int = 200,
               _ok: bool = Security(require_key, scopes=["simulate"])):
    con = fs_connect()
    try:
        rows = get_mc_metrics(con, symbol.strip().upper() if symbol else None, limit=limit)
    finally:
        con.close()
    return {"ok": True, "data": rows}
