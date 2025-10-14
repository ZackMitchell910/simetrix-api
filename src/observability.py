from __future__ import annotations
import os, json, time, uuid, socket
from typing import Callable, Awaitable, Optional, Any
from contextvars import ContextVar
from collections import deque


from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse, JSONResponse
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

LOG_BUFFER_MAX = int(os.getenv("LOG_BUFFER_MAX", "1000"))
LOG_BUFFER = deque(maxlen=LOG_BUFFER_MAX)
# -------- Context --------
_request_id: ContextVar[str] = ContextVar("_request_id", default="-")
SERVICE = os.getenv("SERVICE_NAME", "pathpanda-api")
HOST = socket.gethostname()

# -------- Prometheus Registry + Metrics --------
REGISTRY = CollectorRegistry()
REQ_COUNT = Counter(
    "pp_http_requests_total",
    "HTTP requests (by method, path, status)",
    ["service", "method", "path", "status"],
    registry=REGISTRY,
)
REQ_LATENCY = Histogram(
    "pp_http_request_latency_seconds",
    "HTTP request latency in seconds (by path)",
    ["service", "path"],
    registry=REGISTRY,
)
JOB_SUCCESS = Counter(
    "pp_job_success_total",
    "Background/cron job successes (by job)",
    ["service", "job"],
    registry=REGISTRY,
)
JOB_FAIL = Counter(
    "pp_job_fail_total",
    "Background/cron job failures (by job)",
    ["service", "job"],
    registry=REGISTRY,
)
DUCKDB_HEALTH = Gauge(
    "pp_duckdb_ok",
    "DuckDB connection health (1=ok, 0=bad)",
    ["service"],
    registry=REGISTRY,
)
REDIS_HEALTH = Gauge(
    "pp_redis_ok",
    "Redis connection health (1=ok, 0=bad)",
    ["service"],
    registry=REGISTRY,
)

def log_json(level: str, **fields: Any) -> None:
    base = {
        "level": level,
        "service": SERVICE,
        "host": HOST,
        "request_id": _request_id.get(),
        "ts": time.time(),
    }
    base.update(fields)
    line = json.dumps(base, separators=(",", ":"))
    # write to stdout
    print(line, flush=True)
    # ---- NEW: also keep a rolling in-memory buffer for quick admin peek ----
    try:
        LOG_BUFFER.append(base)
    except Exception:
        pass

# ---- helper the API can call ----
def get_recent_logs(n: int = 200) -> list[dict]:
    n = max(1, min(n, LOG_BUFFER_MAX))
    # Return newest-last for natural reading
    return list(LOG_BUFFER)[-n:]

# -------- Middleware --------
async def _request_logger(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex
    _request_id.set(rid)

    path = request.url.path
    method = request.method
    start = time.perf_counter()
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    except Exception as e:
        status = 500
        log_json("error", msg="unhandled_error", path=path, method=method, error=str(e))
        raise
    finally:
        dur = time.perf_counter() - start
        REQ_COUNT.labels(SERVICE, method, path, str(status)).inc()
        REQ_LATENCY.labels(SERVICE, path).observe(dur)
        log_json("info", msg="http_access", method=method, path=path, status=status, latency_ms=int(dur*1000))

def install_observability(app: FastAPI) -> None:
    app.middleware("http")(_request_logger)

    @app.get("/health", summary="Liveness probe")
    async def health():
        # Simple liveness (process up)
        return {"ok": True, "service": SERVICE}

    @app.get("/status", summary="Readiness & quick status")
    async def status():
        # Quick dependency checks (best-effort; soft-fail)
        duck_ok = int(check_duckdb_ok())
        redis_ok = int(check_redis_ok())
        DUCKDB_HEALTH.labels(SERVICE).set(duck_ok)
        REDIS_HEALTH.labels(SERVICE).set(redis_ok)
        return {
            "ok": bool(duck_ok and redis_ok),
            "service": SERVICE,
            "duckdb_ok": bool(duck_ok),
            "redis_ok": bool(redis_ok),
            "host": HOST,
        }

    @app.get("/metrics")
    async def metrics():
        data = generate_latest(REGISTRY)
        return PlainTextResponse(content=data, media_type=CONTENT_TYPE_LATEST)

# -------- Dependency checks (stubs that you can wire to your actual clients) --------
def check_duckdb_ok() -> bool:
    # Import lazily to avoid hard deps if not needed at import time
    try:
        import duckdb  # noqa
        return True
    except Exception:
        return False

def check_redis_ok() -> bool:
    try:
        import redis
        url = os.getenv("REDIS_URL", "")
        if not url:
            return True  # treat as ok if not configured
        r = redis.from_url(url, decode_responses=True)
        r.ping()
        return True
    except Exception:
        return False

# -------- Helpers for jobs --------
def job_ok(job: str, **extra):
    JOB_SUCCESS.labels(SERVICE, job).inc()
    log_json("info", msg="job_success", job=job, **extra)

def job_fail(job: str, err: str, **extra):
    JOB_FAIL.labels(SERVICE, job).inc()
    log_json("error", msg="job_fail", job=job, error=err, **extra)
    _notify(err=f"{job} failed: {err}", **extra)

# Optional: Discord/Slack webhook
def _notify(**payload):
    webhook = os.getenv("ALERT_WEBHOOK_URL", "").strip()
    if not webhook:
        return
    try:
        import httpx  # lightweight
        # best-effort, fire-and-forget
        with httpx.Client(timeout=3.0) as cli:
            cli.post(webhook, json={"content": f"[{SERVICE}] {json.dumps(payload)}"})
    except Exception:
        pass
