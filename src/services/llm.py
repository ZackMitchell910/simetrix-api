from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import httpx

from src.core import settings

logger = logging.getLogger("simetrix.services.llm")


def _fallback() -> Dict[str, Any]:
    """Return a minimal default payload when LLM calls fail."""
    return {"list": []}


def _coerce_json_obj(content: Any) -> Dict[str, Any]:
    if isinstance(content, dict):
        return content

    if not isinstance(content, str):
        raise ValueError("LLM response was not JSON or string encodable")

    text = content.strip()
    if not text:
        raise ValueError("LLM response was empty")

    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise


def _xai_payload(prompt_user: dict[str, Any], json_schema: Optional[dict[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": os.getenv("XAI_MODEL", "grok-2-mini"),
        "messages": [
            {"role": "system", "content": "Be factual, concise, compliance-safe."},
            prompt_user,
        ],
        "temperature": 0.2,
    }
    if json_schema:
        payload["response_format"] = {"type": "json_schema", "json_schema": json_schema}
    return payload


def _openai_payload(prompt_user: dict[str, Any], json_schema: Optional[dict[str, Any]]) -> dict[str, Any]:
    response_format = {"type": "json_schema", "json_schema": json_schema} if json_schema else {"type": "json_object"}
    return {
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": [
            {"role": "system", "content": "Be factual, concise, compliance-safe."},
            prompt_user,
        ],
        "response_format": response_format,
        "temperature": 0.2,
    }


async def _post(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


def _log_fallback(exc: Exception) -> None:
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("llm_summarize_async: failed to parse LLM content; using fallback", exc_info=exc)
    logger.info("LLM summary failed; fallback used: %s", exc)


async def summarize(
    prompt_user: dict[str, Any],
    *,
    prefer_xai: bool,
    xai_key: Optional[str],
    oai_key: Optional[str],
    json_schema: Optional[dict[str, Any]] = None,
    timeout: float = 20.0,
) -> Dict[str, Any]:
    try:
        if prefer_xai and xai_key:
            payload = _xai_payload(prompt_user, json_schema)
            data = await _post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {xai_key}"},
                payload=payload,
                timeout=timeout,
            )
            raw_content = data.get("choices", [{}])[0].get("message", {}).get("content")
            content = raw_content.strip() if isinstance(raw_content, str) else raw_content
            return _coerce_json_obj(content)

        if oai_key:
            payload = _openai_payload(prompt_user, json_schema)
            data = await _post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {oai_key}"},
                payload=payload,
                timeout=timeout,
            )
            raw_content = data.get("choices", [{}])[0].get("message", {}).get("content")
            content = raw_content.strip() if isinstance(raw_content, str) else raw_content
            return _coerce_json_obj(content)

        return _fallback()

    except Exception as exc:
        _log_fallback(exc)
        return _fallback()
