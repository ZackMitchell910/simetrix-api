# src/api/routers/llm.py
from __future__ import annotations
from typing import Any, Optional
from fastapi import APIRouter, HTTPException, Security
from src.api.auth import require_key
from src.services import llm as llm_service

router = APIRouter(prefix="/llm", tags=["llm"])

@router.post("/summarize", summary="Summarize a run or pick list with an LLM")
async def summarize(payload: dict[str, Any],
                    prefer_xai: bool = True,
                    _ok: bool = Security(require_key, scopes=["simulate"])):
    prompt_user: dict[str, Any] = payload.get("prompt_user") or {}
    schema: Optional[dict[str, Any]] = payload.get("json_schema")

    try:
        xai_key = payload.get("xai_key")
        oai_key = payload.get("oai_key")
        return await llm_service.summarize(
            prompt_user,
            prefer_xai=prefer_xai,
            xai_key=xai_key,
            oai_key=oai_key,
            json_schema=schema,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM summarize failed: {e}")
