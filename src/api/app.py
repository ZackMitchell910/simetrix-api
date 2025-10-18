# src/api/app.py
from __future__ import annotations

import logging
import os
import inspect
from importlib import import_module
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.deps import get_settings, get_redis
from src.quotas import BASE_LIMIT_PER_MIN, rate_limit
from src.api.routers import admin, auth as auth_router, core, inference, llm, training

logger = logging.getLogger("simetrix.api")


class BaselineRateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        p = request.url.path
        if p.startswith(("/metrics", "/health", "/healthz", "/status", "/api-docs")):
            return await call_next(request)
        try:
            redis = await get_redis()
            rate_limit(redis, request, scope="base", limit_per_min=BASE_LIMIT_PER_MIN)
        except HTTPException as e:
            return JSONResponse(status_code=e.status_code, content={"ok": False, "error": e.detail})
        return await call_next(request)

def _cors_origins():
    s = get_settings()
    default = [
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:3000", "http://127.0.0.1:3000",
        "https://simetrix.io", "https://www.simetrix.io",
        "https://simetrix.vercel.app",
    ]
    try:
        raw = (s.cors_origins_raw or "").strip()
        if not raw:
            return default
        if raw.startswith("["):
            import json
            return json.loads(raw)
        return [x.strip() for x in raw.split(",") if x.strip()]
    except Exception:
        return default

app = FastAPI(title="Simetrix API (modular)", version="1.3.x", docs_url="/api-docs")
app.add_middleware(BaselineRateLimitMiddleware)

# CORS (matches your defaults)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_origin_regex=r"https://.*\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "X-API-Key", "x-pt-key", "Authorization"],
    max_age=86400,
)

# Routers
app.include_router(auth_router.router)
app.include_router(admin.router)
app.include_router(training.router)
app.include_router(inference.router)
app.include_router(llm.router)
app.include_router(core.router)


@app.get("/")
def root():
    return {"ok": True, "service": "simetrix-api", "modular": True}


# Static assets / SPA fallback (mirrors legacy behaviour)
FRONTEND_DIR = os.getenv("PT_FRONTEND_DIR", "").strip()
if FRONTEND_DIR:
    try:
        app.mount(
            "/assets",
            StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")),
            name="assets",
        )

        @app.get("/{path:path}", response_class=HTMLResponse)
        async def spa_fallback(path: str):  # pragma: no cover - filesystem dependent
            file_path = os.path.join(FRONTEND_DIR, path)
            if os.path.isfile(file_path):
                return FileResponse(file_path)
            index = os.path.join(FRONTEND_DIR, "index.html")
            if os.path.isfile(index):
                return FileResponse(index)
            raise HTTPException(status_code=404, detail="index.html not found")

        logger.info("Static frontend mounted from %s", FRONTEND_DIR)
    except Exception as exc:  # pragma: no cover - optional
        logger.warning("Static frontend disabled: %s", exc)
else:

    @app.get("/{path:path}")
    async def catch_all(path: str):
        return {"status": "ok", "message": "SIMETRIX.IO API Is Running."}


_legacy_module: Any | None = None
_legacy_attempted = False


def _load_legacy() -> Any | None:
    global _legacy_module, _legacy_attempted
    if not _legacy_attempted:
        _legacy_attempted = True
        try:
            _legacy_module = import_module("src.predictive_api")
            logger.info("Legacy predictive_api module loaded for lifecycle hooks")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to import legacy predictive_api: %s", exc)
            _legacy_module = None
    return _legacy_module


@app.on_event("startup")
async def _startup() -> None:
    legacy = _load_legacy()
    if not legacy:
        return
    hook = getattr(legacy, "_on_startup", None)
    if hook:
        result = hook()
        if inspect.isawaitable(result):
            await result


@app.on_event("shutdown")
async def _shutdown() -> None:
    legacy = _load_legacy()
    if not legacy:
        return
    hook = getattr(legacy, "_on_shutdown", None)
    if hook:
        result = hook()
        if inspect.isawaitable(result):
            await result
