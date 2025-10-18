from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import HTTPException, Request

from src.core import REDIS
from src.observability import log_json
from src.quotas import (
    BASE_LIMIT_PER_MIN,
    CRON_LIMIT_PER_MIN,
    SIM_LIMIT_PER_MIN,
    usage_today,
    usage_today_for_caller,
)


async def me_limits(request: Request) -> dict[str, Any]:
    try:
        used_sim, limit_sim, plan_sim, caller = await usage_today(REDIS, request, scope="simulate")
        used_cron, limit_cron, plan_cron = await usage_today_for_caller(REDIS, caller, scope="cron")

        plan = plan_sim or plan_cron

        now = datetime.now(timezone.utc)
        tomorrow = (now + timedelta(days=1)).date()
        reset_at = datetime.combine(tomorrow, datetime.min.time(), tzinfo=timezone.utc)
        seconds_to_reset = int((reset_at - now).total_seconds())

        payload = {
            "ok": True,
            "plan": plan,
            "caller": caller,
            "reset_secs": seconds_to_reset,
            "per_min_caps": {
                "base": BASE_LIMIT_PER_MIN,
                "simulate": SIM_LIMIT_PER_MIN,
                "cron": CRON_LIMIT_PER_MIN,
            },
            "daily": {
                "simulate": {
                    "used": used_sim,
                    "limit": limit_sim,
                    "remaining": max(0, limit_sim - used_sim),
                },
                "cron": {
                    "used": used_cron,
                    "limit": limit_cron,
                    "remaining": max(0, limit_cron - used_cron),
                },
            },
        }
        log_json("info", msg="me_limits", plan=plan, caller_tail=caller[-8:])
        return payload
    except Exception as exc:
        log_json("error", msg="me_limits_fail", error=str(exc))
        raise HTTPException(status_code=500, detail="failed_to_get_limits") from exc
