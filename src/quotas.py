# src/quotas.py
from __future__ import annotations
import os, time, ipaddress
from typing import Optional, Tuple
from fastapi import Request, HTTPException
from redis import Redis

# Defaults (override via env)
BASE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "120"))   # baseline RPS/min per caller
SIM_LIMIT_PER_MIN  = int(os.getenv("SIM_LIMIT_PER_MIN",  "30"))    # simulate endpoints
CRON_LIMIT_PER_MIN = int(os.getenv("CRON_LIMIT_PER_MIN", "10"))    # admin cron endpoints

# Daily quotas per plan (override via env)
Q_FREE  = int(os.getenv("QUOTA_FREE_DAILY",  "50"))
Q_PRO   = int(os.getenv("QUOTA_PRO_DAILY",   "500"))
Q_INST  = int(os.getenv("QUOTA_INST_DAILY",  "5000"))
PLAN_DEFAULT = os.getenv("PLAN_DEFAULT", "free")  # free|pro|inst

# Redis key helpers
def _k_rate(caller: str, scope: str, minute_bucket: int) -> str:
    return f"rl:{scope}:{caller}:{minute_bucket}"

def _k_quota(caller: str, scope: str, ymd: str) -> str:
    return f"qt:{scope}:{caller}:{ymd}"

def _state_caller(request: Request) -> Optional[str]:
    return getattr(request.state, "caller_id", None)

def _state_plan(request: Request) -> Optional[str]:
    return getattr(request.state, "plan", None)

def _caller_id(request: Request) -> str:
    state_caller = _state_caller(request)
    if state_caller:
        return state_caller
    # Prefer API key; otherwise bucket by IP
    key = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
    if key:
        return f"key:{key.strip()}"
    # Fallback to client host
    ip = request.client.host if request.client else "0.0.0.0"
    # normalize IPv6/IPv4
    try:
        ip = str(ipaddress.ip_address(ip))
    except Exception:
        pass
    return f"ip:{ip}"

def _now_minute_bucket() -> int:
    return int(time.time() // 60)

def _today_ymd() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())

def _plan_for(caller: str, r: Optional[Redis]) -> str:
    # Optional: store/override user plans in Redis: HGET plan:{caller} field "plan"
    if not r:
        return PLAN_DEFAULT
    try:
        p = r.hget(f"plan:{caller}", "plan")
        if p:
            return str(p)
    except Exception:
        pass
    return PLAN_DEFAULT

def _daily_quota_for(plan: str) -> int:
    if plan == "inst":
        return Q_INST
    if plan == "pro":
        return Q_PRO
    return Q_FREE

def rate_limit(r: Optional[Redis], request: Request, scope: str, limit_per_min: int) -> None:
    """
    Fixed-window per-minute limiter. Raises 429 on exceed.
    """
    if not r:
        return  # no Redis configured => no enforcement
    caller = _caller_id(request)
    bucket = _now_minute_bucket()
    key = _k_rate(caller, scope, bucket)
    try:
        cur = r.incr(key)
        if cur == 1:
            # first hit this minute; set TTL ~90s to let bucket expire
            r.expire(key, 90)
        if cur > limit_per_min:
            raise HTTPException(status_code=429, detail=f"Rate limit exceeded for scope={scope}")
    except HTTPException:
        raise
    except Exception:
        # fail-open on Redis issues
        return

def quota_consume(r: Optional[Redis], request: Request, scope: str, units: int = 1) -> Tuple[int,int]:
    """
    Per-day quota by plan (free/pro/inst). Returns (used, limit). Raises 429 on exceed.
    """
    if not r:
        return (0, 0)  # no Redis => skip
    caller = _caller_id(request)
    plan = _state_plan(request) or _plan_for(caller, r)
    limit = _daily_quota_for(plan)
    ymd = _today_ymd()
    key = _k_quota(caller, scope, ymd)
    try:
        used = r.incrby(key, units)
        if used == units:
            # first touch today; expire after ~27h to straddle TZ safely
            r.expire(key, 27 * 3600)
        if used > limit:
            # roll back a bit (best-effort)
            try: r.decrby(key, units)
            except Exception: pass
            raise HTTPException(status_code=429, detail=f"Daily quota exceeded (plan={plan}, scope={scope})")
        return (used, limit)
    except HTTPException:
        raise
    except Exception:
        return (0, 0)  # fail-open

VALID_PLANS = {"free", "pro", "inst"}

def set_plan_for_key(r: Optional[Redis], api_key: str, plan: str) -> str:
    if not r:
        raise RuntimeError("Redis not configured")
    plan = plan.strip().lower()
    if plan not in VALID_PLANS:
        raise ValueError(f"invalid plan '{plan}' (valid: {sorted(VALID_PLANS)})")
    caller = f"key:{api_key.strip()}"
    r.hset(f"plan:{caller}", mapping={"plan": plan, "updated_at": str(int(time.time()))})
    return plan

def get_plan_for_key(r: Optional[Redis], api_key: str) -> str:
    if not r:
        return PLAN_DEFAULT
    caller = f"key:{api_key.strip()}"
    try:
        p = r.hget(f"plan:{caller}", "plan")
        return (p or PLAN_DEFAULT)
    except Exception:
        return PLAN_DEFAULT

# Usage helpers (today) for a given caller or current request
def usage_today_for_caller(r: Optional[Redis], caller: str, scope: str) -> Tuple[int, int, str]:
    """
    Returns (used, limit, plan) for today's quota in 'scope' for a given caller id.
    """
    if not r:
        return (0, 0, PLAN_DEFAULT)
    plan = _plan_for(caller, r)  # reuses Redis lookup if present
    limit = _daily_quota_for(plan)
    ymd = _today_ymd()
    try:
        used = int(r.get(_k_quota(caller, scope, ymd)) or 0)
    except Exception:
        used = 0
    return (used, limit, plan)

def usage_today(r: Optional[Redis], request: Request, scope: str) -> Tuple[int, int, str, str]:
    """
    Returns (used, limit, plan, caller) for today's quota in 'scope' for the current caller.
    """
    caller = _caller_id(request)
    used, limit, plan = usage_today_for_caller(r, caller, scope)
    return (used, limit, plan, caller)
# A tiny helper you can call at the top of “heavy” endpoints:
def enforce_limits(r: Optional[Redis], request: Request, scope: str, per_min: int, cost_units: int = 1) -> None:
    rate_limit(r, request, scope, per_min)
    quota_consume(r, request, scope, cost_units)
