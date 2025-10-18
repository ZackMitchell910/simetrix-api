from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import hmac
import inspect
import io
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from jose import JWTError, jwt
from pydantic import BaseModel

from src.core import ARROW_ROOT, BACKUP_DIR, REDIS, START_TIME, settings
from src.infra.backups import create_duckdb_backup
from src.observability import job_fail, job_ok, log_json, get_recent_logs
from src.quotas import (
    BASE_LIMIT_PER_MIN,
    CRON_LIMIT_PER_MIN,
    PLAN_DEFAULT,
    SIM_LIMIT_PER_MIN,
    enforce_limits,
    rate_limit,
    set_plan_for_key,
    get_plan_for_key,
)
from src.services import common as service_common
from src.services import ingestion as ingestion_service
from src.services.watchlists import get_crypto_watchlist, get_equity_watchlist
from src.services.validation import validate_mc_paths as run_mc_validation
from src.services.labeling import label_outcomes
from src.services.training import OnlineLearnRequest, learn_online as perform_learn_online
try:
    from src.db.duck import DB_PATH as CORE_DB_PATH, recent_predictions
except Exception:  # pragma: no cover - optional import
    CORE_DB_PATH = None  # type: ignore
    def recent_predictions(*args, **kwargs):  # type: ignore
        raise RuntimeError("recent_predictions unavailable")

try:
    from src.feature_store import connect as feature_store_connect, DB_PATH as FS_DB_PATH
except Exception:  # pragma: no cover - optional import
    feature_store_connect = None  # type: ignore
    FS_DB_PATH = None  # type: ignore

JWT_ALGORITHM = "HS256"


def _resolve_backup_context() -> tuple[list[tuple[str, Path]], Path, int]:
    cfg = settings
    keep = max(1, int(getattr(cfg, "backup_keep", 7)))
    backup_dir = Path(getattr(cfg, "backup_dir", BACKUP_DIR)).expanduser()
    targets: list[tuple[str, Path]] = []
    if CORE_DB_PATH:
        targets.append(("core", Path(CORE_DB_PATH).expanduser()))
    if FS_DB_PATH:
        targets.append(("feature_store", Path(FS_DB_PATH).expanduser()))
    return targets, backup_dir, keep


def _active_settings():
    return settings


def _active_redis():
    return REDIS


class AdminLoginRequest(BaseModel):
    username: str
    password: str


def credentials_configured() -> bool:
    cfg = _active_settings()
    username = str(getattr(cfg, "admin_username", "") or "").strip()
    password_hash = str(getattr(cfg, "admin_password_hash", "") or "").strip()
    return bool(username) and bool(password_hash)


def verify_admin_password(candidate: str) -> bool:
    cfg = _active_settings()
    stored = str(getattr(cfg, "admin_password_hash", "") or "").strip()
    if not stored:
        return False
    try:
        from passlib.hash import bcrypt as bcrypt_hash  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        bcrypt_hash = None  # type: ignore
    if stored.startswith("$2") and bcrypt_hash is not None:
        try:
            return bool(bcrypt_hash.verify(candidate, stored))
        except ValueError:
            return False
    if stored.startswith("sha256:"):
        expected = stored.split(":", 1)[1]
        digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        return hmac.compare_digest(digest, expected)
    if stored.startswith("plain:"):
        expected = stored.split(":", 1)[1]
        return hmac.compare_digest(candidate, expected)
    return False


def _admin_session_claims(username: str) -> dict[str, Any]:
    cfg = _active_settings()
    issued_at = datetime.now(timezone.utc)
    expires_at = issued_at + timedelta(
        minutes=int(getattr(cfg, "admin_session_ttl_minutes", 30))
    )
    return {
        "sub": username,
        "type": "admin",
        "scopes": ["admin"],
        "iat": int(issued_at.timestamp()),
        "exp": int(expires_at.timestamp()),
        "jti": uuid4().hex,
    }


def encode_admin_session(username: str) -> str:
    claims = _admin_session_claims(username)
    cfg = _active_settings()
    secret = getattr(cfg, "admin_session_secret", "") or ""
    return jwt.encode(claims, secret, algorithm=JWT_ALGORITHM)


def set_admin_cookie(response: Response, token: str) -> None:
    cfg = _active_settings()
    max_age = int(getattr(cfg, "admin_session_ttl_minutes", 30)) * 60
    cookie_name = getattr(cfg, "admin_session_cookie", "pt_admin_session")
    response.set_cookie(
        key=cookie_name,
        value=token,
        max_age=max_age,
        expires=max_age,
        path="/",
        secure=bool(getattr(cfg, "admin_cookie_secure", False)),
        httponly=True,
        samesite="Strict",
    )


def clear_admin_cookie(response: Response) -> None:
    cfg = _active_settings()
    response.delete_cookie(getattr(cfg, "admin_session_cookie", "pt_admin_session"), path="/")


def decode_admin_session(token: str) -> dict[str, Any]:
    try:
        cfg = _active_settings()
        secret = getattr(cfg, "admin_session_secret", "") or ""
        payload = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="invalid_admin_session") from exc
    if payload.get("type") != "admin":
        raise HTTPException(status_code=403, detail="invalid_admin_session_type")
    return payload


def get_admin_session(request: Request, *, required: bool = False) -> dict[str, Any] | None:
    cfg = _active_settings()
    cookie_name = getattr(cfg, "admin_session_cookie", "pt_admin_session")
    token = request.cookies.get(cookie_name)
    if not token:
        if required:
            raise HTTPException(status_code=401, detail="admin_session_required")
        return None
    try:
        payload = decode_admin_session(token)
    except HTTPException as exc:
        if required:
            raise
        log_json(
            "warning",
            msg="admin_session_decode_failed",
            error=str(exc.detail),
            remote=request.client.host if request.client else None,
        )
        return None
    request.state.admin_session = payload
    return payload


def admin_actor(request: Request) -> str:
    session = getattr(request.state, "admin_session", {}) or {}
    cfg = _active_settings()
    default_actor = getattr(cfg, "admin_username", "admin") or "admin"
    return str(session.get("sub") or default_actor)


async def login(request: Request, response: Response, payload: Any) -> dict[str, Any]:
    try:
        data = AdminLoginRequest.model_validate(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid_payload") from exc

    username = (data.username or "").strip()
    password = data.password or ""

    if not credentials_configured():
        raise HTTPException(status_code=503, detail="admin_login_disabled")

    cfg = _active_settings()
    redis = _active_redis()

    await rate_limit(redis, request, "admin_login", int(getattr(cfg, "admin_login_rpm", 10)))

    configured = str(getattr(cfg, "admin_username", "") or "").strip()
    if (username.lower() != configured.lower()) or (not verify_admin_password(password)):
        log_json(
            "warning",
            msg="admin_login_failed",
            user=username or "<empty>",
            reason="invalid_credentials",
            remote=request.client.host if request.client else None,
        )
        raise HTTPException(status_code=401, detail="invalid_credentials")

    token = encode_admin_session(configured)
    set_admin_cookie(response, token)
    response.headers["Cache-Control"] = "no-store"
    log_json(
        "info",
        msg="admin_login_success",
        user=configured,
        remote=request.client.host if request.client else None,
    )
    return {"status": "ok"}


async def logout(request: Request, response: Response) -> dict[str, Any]:
    actor = None
    try:
        session = get_admin_session(request, required=False)
        if session:
            actor = session.get("sub")
    except HTTPException:
        actor = None
    clear_admin_cookie(response)
    response.headers["Cache-Control"] = "no-store"
    log_json(
        "info",
        msg="admin_logout",
        user=actor or "unknown",
        remote=request.client.host if request.client else None,
    )
    return {"status": "ok"}


async def refresh(request: Request, response: Response) -> dict[str, Any]:
    session = get_admin_session(request, required=True) or {}
    cfg = _active_settings()
    actor = str(session.get("sub") or (getattr(cfg, "admin_username", "admin") or "admin"))
    token = encode_admin_session(actor)
    set_admin_cookie(response, token)
    response.headers["Cache-Control"] = "no-store"
    log_json(
        "info",
        msg="admin_session_refreshed",
        user=actor,
        remote=request.client.host if request.client else None,
    )
    return {"status": "ok"}


async def warm() -> dict[str, Any]:
    await asyncio.gather(
        service_common.ensure_tf(),
        service_common.ensure_arima(),
    )
    try:
        await service_common.ensure_sb3()
    except Exception:
        pass
    return {
        "ok": True,
        "tf": bool(service_common.TF_AVAILABLE),
        "arima": bool(service_common.ARIMA_AVAILABLE),
        "sb3": bool(service_common.SB3_AVAILABLE),
    }


async def system_health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "simetrix-api",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def backup_once() -> list[dict[str, Any]]:
    targets, backup_dir, backup_keep = _resolve_backup_context()
    results: list[dict[str, Any]] = []
    if not targets:
        return results
    for name, db_path in targets:
        try:
            dest = await asyncio.to_thread(
                create_duckdb_backup,
                db_path,
                backup_dir,
                keep=backup_keep,
            )
            log_json("info", msg="duckdb_backup_ok", db=name, dest=str(dest))
            results.append({"db": name, "dest": str(dest), "ok": True})
        except FileNotFoundError:
            log_json("warning", msg="duckdb_backup_missing", db=name, path=str(db_path))
            results.append({"db": name, "ok": False, "error": "missing"})
        except Exception as exc:
            log_json("error", msg="duckdb_backup_fail", db=name, error=str(exc))
            results.append({"db": name, "ok": False, "error": str(exc)})
    return results


async def validate_mc_paths(symbols: list[str], days: int, n_paths: int) -> dict[str, Any]:
    return await run_mc_validation(symbols, days=days, n_paths=n_paths)


async def cron_daily(request: Request, *, n: int, steps: int, batch: int) -> dict[str, Any]:
    redis = _active_redis()
    await enforce_limits(redis, request, scope="cron", per_min=CRON_LIMIT_PER_MIN, cost_units=1)

    WL_EQ = list(get_equity_watchlist())
    WL_CR = list(get_crypto_watchlist())

    job = "cron_daily"
    t0 = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()

    try:
        log_json("info", msg="cron_start", job=job, n=n, steps=steps, batch=batch)

        label_t0 = time.perf_counter()
        lab = await label_outcomes(limit=20000)  # type: ignore[misc]
        label_sec = round(time.perf_counter() - label_t0, 3)
        log_json("info", msg="cron_labeled", job=job, labeled=lab, duration_s=label_sec)

        syms = list(dict.fromkeys(list(WL_EQ)[:n] + list(WL_CR)[:n]))
        log_json("info", msg="cron_symbols_prepared", job=job, n_symbols=len(syms))

        learned: list[dict[str, Any]] = []
        ok_count = 0
        err_count = 0

        async def learn_one(sym: str) -> dict[str, Any]:
            sym_t0 = time.perf_counter()
            last_err: Optional[str] = None
            for attempt in (1, 2):
                try:
                    req = OnlineLearnRequest(symbol=sym, steps=steps, batch=batch)  # type: ignore[call-arg]
                    res = await perform_learn_online(req)  # type: ignore[misc]
                    dur = round(time.perf_counter() - sym_t0, 3)
                    log_json("info", msg="learn_ok", job=job, symbol=sym, attempt=attempt, duration_s=dur)
                    return {
                        "symbol": sym,
                        "status": res.get("status", "ok") if isinstance(res, dict) else "ok",
                        "attempt": attempt,
                        "duration_s": dur,
                    }
                except Exception as exc:
                    last_err = str(exc)
                    log_json("error", msg="learn_err", job=job, symbol=sym, attempt=attempt, error=last_err)
                    if attempt == 1:
                        await asyncio.sleep(0.5)
            dur = round(time.perf_counter() - sym_t0, 3)
            return {"symbol": sym, "status": "error", "error": last_err or "unknown", "attempt": 2, "duration_s": dur}

        for sym in syms:
            item = await learn_one(sym)
            learned.append(item)
            if item.get("status") == "ok":
                ok_count += 1
            else:
                err_count += 1

        total_sec = round(time.perf_counter() - t0, 3)
        summary = {
            "ok": True,
            "job": job,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_s": total_sec,
            "labeled": lab,
            "n_symbols": len(syms),
            "learn_ok": ok_count,
            "learn_err": err_count,
            "learned": learned,
            "params": {"n": n, "steps": steps, "batch": batch},
        }

        job_ok(job, n=n, steps=steps, batch=batch, duration_s=total_sec, learn_ok=ok_count, learn_err=err_count)
        log_json("info", msg="cron_done", **summary)
        return summary
    except Exception as exc:
        err_msg = str(exc)
        job_fail(job, err=err_msg, n=n, steps=steps, batch=batch)
        log_json("error", msg="cron_failed", job=job, error=err_msg)
        raise HTTPException(status_code=500, detail=f"{job} failed: {err_msg}") from exc


async def fetch_news(
    request: Request,
    *,
    symbol: str,
    days: int,
    limit: int,
) -> dict[str, Any]:
    redis = _active_redis()
    await enforce_limits(redis, request, scope="cron", per_min=CRON_LIMIT_PER_MIN, cost_units=1)
    result = await ingestion_service.ingest_news(
        symbol,
        days=int(days),
        limit=int(limit),
        provider=None,
        log_tag="admin",
    )
    return {"ok": True, **(result or {})}


async def score_news(
    request: Request,
    *,
    symbol: str,
    days: int,
    batch: int,
) -> dict[str, Any]:
    redis = _active_redis()
    await enforce_limits(redis, request, scope="cron", per_min=CRON_LIMIT_PER_MIN, cost_units=1)
    result = await ingestion_service.score_news(
        symbol,
        days=int(days),
        batch=int(batch),
        log_tag="admin",
    )
    return {"ok": True, **(result or {})}


async def fetch_earnings(
    request: Request,
    *,
    symbol: str,
    lookback_days: int,
    limit: int,
) -> dict[str, Any]:
    redis = _active_redis()
    await enforce_limits(redis, request, scope="cron", per_min=CRON_LIMIT_PER_MIN, cost_units=1)
    result = await ingestion_service.ingest_earnings(
        symbol,
        lookback_days=int(lookback_days),
        limit=int(limit),
        provider=None,
        log_tag="admin",
    )
    return {"ok": True, **(result or {})}


async def fetch_macro(
    request: Request,
    *,
    provider: Optional[str],
) -> dict[str, Any]:
    redis = _active_redis()
    await enforce_limits(redis, request, scope="cron", per_min=CRON_LIMIT_PER_MIN, cost_units=1)
    result = await ingestion_service.ingest_macro(provider, log_tag="admin")
    return {"ok": True, **(result or {})}


async def ingest_backfill(days: int) -> dict[str, Any]:
    base = datetime.now(timezone.utc).date()
    total = 0
    pause_s = float(os.getenv("PT_BACKFILL_PAUSE_S", "2.5") or 2.5)
    for offset in range(int(max(1, days))):
        target = base - timedelta(days=offset)
        result = await ingestion_service.ingest_grouped_daily(target, log_tag="admin_backfill")
        total += int(result.get("upserted") or 0)
        await asyncio.sleep(pause_s)
    return {"ok": True, "days": int(days), "rows": total}


async def logs_latest(n: int) -> dict[str, Any]:
    try:
        items = get_recent_logs(n=n)
        return {"ok": True, "count": len(items), "logs": items}
    except Exception as exc:
        log_json("error", msg="admin_logs_latest_fail", error=str(exc))
        raise HTTPException(status_code=500, detail="failed_to_fetch_logs") from exc


async def plan_set(api_key: str, plan: str) -> dict[str, Any]:
    try:
        redis = _active_redis()
        new_plan = await set_plan_for_key(redis, api_key, plan)
        log_json("info", msg="admin_plan_set", target_key=f"...{api_key[-6:]}", plan=new_plan)
        return {"ok": True, "api_key_tail": api_key[-6:], "plan": new_plan}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log_json("error", msg="admin_plan_set_fail", error=str(exc))
        raise HTTPException(status_code=500, detail="failed_to_set_plan") from exc


async def plan_get(api_key: str) -> dict[str, Any]:
    try:
        redis = _active_redis()
        plan = await get_plan_for_key(redis, api_key)
        return {"ok": True, "api_key_tail": api_key[-6:], "plan": plan}
    except Exception as exc:
        log_json("error", msg="admin_plan_get_fail", error=str(exc))
        raise HTTPException(status_code=500, detail="failed_to_get_plan") from exc


async def ingest_daily(as_of: date) -> dict[str, Any]:
    result = await ingestion_service.ingest_grouped_daily(as_of, log_tag="admin")
    return {"ok": True, "date": as_of.isoformat(), "rows": int(result.get("upserted") or 0)}


async def report_system_health(request: Request) -> dict[str, Any]:
    start_time = START_TIME if isinstance(START_TIME, datetime) else datetime.now(timezone.utc)
    uptime = datetime.now(timezone.utc) - start_time
    arrow_root = str(ARROW_ROOT)
    fs_path = FS_DB_PATH
    wl_eq = get_equity_watchlist(update_settings=False)
    wl_cr = get_crypto_watchlist(update_settings=False)

    result = {
        "service": "simetrix-api",
        "version": os.getenv("PT_RELEASE", "dev"),
        "uptime_seconds": int(uptime.total_seconds()),
        "start_time": start_time.isoformat(),
        "redis": bool(_active_redis()),
        "duckdb_path": str(fs_path) if fs_path else None,
        "arrow_root": arrow_root,
        "watch_equities": len(wl_eq),
        "watch_cryptos": len(wl_cr),
    }
    log_json(
        "info",
        msg="admin_report_system_health",
        actor=admin_actor(request),
        uptime_seconds=result["uptime_seconds"],
    )
    return result


async def report_simulations(request: Request, *, limit: int, symbol: Optional[str]) -> dict[str, Any]:
    try:
        rows = recent_predictions(symbol=symbol, limit=limit)
    except Exception as exc:
        log_json(
            "error",
            msg="admin_report_simulations_fail",
            actor=admin_actor(request),
            error=str(exc),
            symbol=symbol,
        )
        raise HTTPException(status_code=500, detail=f"failed to fetch simulations: {exc}") from exc

    payload = {"count": len(rows), "items": rows}
    log_json(
        "info",
        msg="admin_report_simulations",
        actor=admin_actor(request),
        count=len(rows),
        symbol=symbol or "*",
    )
    return payload


async def _scan_model_meta(pattern: str = "model_meta:*", limit: int = 500) -> list[dict[str, Any]]:
    redis = _active_redis()
    if not redis:
        return []
    results: list[dict[str, Any]] = []
    try:
        async for key in redis.scan_iter(match=pattern, count=200):
            try:
                raw = await redis.get(key)
                if not raw:
                    continue
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8", errors="replace")
                doc = json.loads(raw)
                doc["key"] = key
                results.append(doc)
            except Exception:
                continue
            if len(results) >= limit:
                break
    except Exception:
        return []
    return results


async def report_model_training(request: Request, *, limit: int) -> dict[str, Any]:
    meta = await _scan_model_meta(limit=limit)
    meta.sort(key=lambda x: x.get("trained_at", ""), reverse=True)
    payload = {"count": len(meta), "items": meta[:limit]}
    log_json(
        "info",
        msg="admin_report_model_training",
        actor=admin_actor(request),
        count=payload["count"],
    )
    return payload


async def report_usage(request: Request, *, scope: str, sample: int) -> dict[str, Any]:
    redis = _active_redis()
    records: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    if not redis:
        payload = {"records": records, "note": "Redis unavailable; usage data not captured."}
        log_json(
            "warning",
            msg="admin_report_usage_unavailable",
            actor=admin_actor(request),
            scope=scope,
        )
        return payload
    try:
        async for key in redis.scan_iter(match=f"qt:{scope}:*", count=500):
            raw = await redis.get(key)
            if not raw:
                continue
            parts = key.split(":")
            if len(parts) < 4:
                continue
            _, _, caller, ymd = parts
            used = int(raw)
            records.append({"caller": caller, "date": ymd, "used": used})
            if len(records) >= sample:
                break
    except Exception as exc:
        log_json(
            "error",
            msg="admin_report_usage_fail",
            actor=admin_actor(request),
            scope=scope,
            error=str(exc),
        )
        return {"records": [], "note": f"Usage fetch failed: {exc}"}
    payload = {"generated_at": now.isoformat(), "records": records}
    log_json(
        "info",
        msg="admin_report_usage",
        actor=admin_actor(request),
        scope=scope,
        count=len(records),
    )
    return payload


async def report_telemetry(request: Request, *, limit: int) -> StreamingResponse:
    headers = ["timestamp", "level", "message"]
    timestamp = datetime.now(timezone.utc).isoformat()
    rows = [
        [timestamp, "INFO", "Telemetry export not yet implemented; hook into real store to enrich."],
    ]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(headers)
    writer.writerows(rows[:limit])
    buffer.seek(0)
    log_json(
        "info",
        msg="admin_report_telemetry",
        actor=admin_actor(request),
        rows=min(len(rows), limit),
    )
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=telemetry_{timestamp[:10]}.csv"},
    )


def ls_valid_signature(raw: bytes, sig_b64: str) -> bool:
    cfg = _active_settings()
    secret = (getattr(cfg, "admin_session_secret", "") or "").encode("utf-8")
    if not secret or not sig_b64:
        return False
    mac = hmac.new(secret, msg=raw, digestmod=hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expected, sig_b64)


async def grant_plan(email: str, plan: str, ttl_days: int | None = 365) -> None:
    redis = _active_redis()
    if not redis or not email:
        return
    key = f"user:{email.lower()}:plan"
    if ttl_days:
        await redis.set(key, plan, ex=ttl_days * 24 * 3600)
    else:
        await redis.set(key, plan)


async def revoke_or_downgrade(email: str) -> None:
    redis = _active_redis()
    if not redis or not email:
        return
    key = f"user:{email.lower()}:plan"
    await redis.set(key, PLAN_DEFAULT, ex=30 * 24 * 3600)


def _norm_plan_name(raw: str | None) -> str:
    name = (raw or "").strip().lower()
    if "enterprise" in name:
        return "enterprise"
    if "pro" in name:
        return "pro"
    return PLAN_DEFAULT


def _payload_email(attrs: dict[str, Any]) -> str:
    return str(
        attrs.get("user_email")
        or attrs.get("customer_email")
        or attrs.get("email")
        or ""
    )


async def billing_webhook(request: Request) -> dict[str, Any]:
    raw = await request.body()
    sig = request.headers.get("X-Signature", "")
    if not ls_valid_signature(raw, sig or ""):
        raise HTTPException(status_code=401, detail="invalid_signature")

    payload = json.loads(raw.decode("utf-8") or "{}")
    meta = payload.get("meta") or {}
    event = (meta.get("event_name") or payload.get("event_name") or "").strip().lower()
    data = payload.get("data") or {}
    attrs = (data.get("attributes") or {}) if isinstance(data, dict) else {}

    evt_id = meta.get("event_id") or data.get("id") if isinstance(data, dict) else None
    redis = _active_redis()
    if redis and evt_id:
        if await redis.sismember("ls:events:seen", evt_id):
            return {"ok": True, "idempotent": True}
        await redis.sadd("ls:events:seen", evt_id)
        await redis.expire("ls:events:seen", 14 * 24 * 3600)

    email = (_payload_email(attrs) or "").lower()
    variant_name = _norm_plan_name(attrs.get("variant_name") or attrs.get("name"))

    grant_events = {
        "order_created",
        "subscription_created",
        "subscription_updated",
    }
    revoke_events = {
        "subscription_cancelled",
        "subscription_expired",
        "subscription_paused",
        "order_refunded",
    }

    if event in grant_events:
        await grant_plan(email, variant_name, ttl_days=365)
    elif event in revoke_events:
        await revoke_or_downgrade(email)

    return {"ok": True, "event": event, "email": email, "plan": variant_name}


__all__ = [
    "AdminLoginRequest",
    "credentials_configured",
    "verify_admin_password",
    "encode_admin_session",
    "set_admin_cookie",
    "clear_admin_cookie",
    "decode_admin_session",
    "get_admin_session",
    "admin_actor",
    "login",
    "logout",
    "refresh",
    "warm",
    "system_health",
    "backup_once",
    "validate_mc_paths",
    "cron_daily",
    "fetch_news",
    "score_news",
    "fetch_earnings",
    "fetch_macro",
    "ingest_backfill",
    "logs_latest",
    "plan_set",
    "plan_get",
    "ingest_daily",
    "report_system_health",
    "report_simulations",
    "report_model_training",
    "report_usage",
    "report_telemetry",
    "ls_valid_signature",
    "grant_plan",
    "revoke_or_downgrade",
    "billing_webhook",
]
