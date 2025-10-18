from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Sequence

from fastapi import HTTPException, Request
from jose import JWTError, jwt

from src.auth_service import authenticate, all_scopes_for, get_user, UserKey
from src.core import JWT_ALGORITHM, REDIS, settings
from src.quotas import PLAN_DEFAULT, get_plan_for_key


def create_access_token(user: UserKey, scopes: Sequence[str]) -> str:
    """Mint a signed JWT for the given user and scope list."""
    if user.id is None:
        raise ValueError("User missing primary key")
    issued_at = datetime.now(timezone.utc)
    expires_at = issued_at + timedelta(minutes=int(settings.jwt_exp_minutes))
    scope_list = sorted({scope.strip() for scope in scopes if scope}) or ["simulate"]
    payload = {
        "sub": str(user.id),
        "email": user.email,
        "plan": user.plan,
        "scopes": scope_list,
        "iat": int(issued_at.timestamp()),
        "exp": int(expires_at.timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Mapping[str, Any]:
    """Validate and decode an access token."""
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[JWT_ALGORITHM])
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid access token") from exc


async def authenticate_bearer(
    request: Request,
    required_scopes: Sequence[str],
    token: str | None,
) -> UserKey:
    """Authenticate a bearer token and populate request.state."""
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    payload = decode_token(token)
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Token missing subject")

    try:
        user_id = int(sub)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=401, detail="Invalid subject in token") from exc

    user = get_user(user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User inactive or missing")

    token_scopes = payload.get("scopes") or []
    missing = [scope for scope in required_scopes if scope not in token_scopes]
    if missing:
        raise HTTPException(status_code=403, detail=f"Missing scopes: {', '.join(missing)}")

    request.state.caller_id = f"user:{user.id}"
    request.state.plan = payload.get("plan") or user.plan or PLAN_DEFAULT
    request.state.scopes = list(token_scopes)
    request.state.user = user
    request.state.auth_source = "oauth"
    return user


async def authorize_api_key(
    request: Request,
    api_key: str,
) -> bool:
    """
    Authenticate a caller via API key and annotate request.state.

    Returns True when the key is accepted, otherwise raises HTTPException.
    """
    provided = api_key.strip()
    expected = (settings.pt_api_key or os.getenv("PT_API_KEY", "") or "").strip()
    if expected and provided != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    plan = PLAN_DEFAULT
    if REDIS:
        try:
            plan = await get_plan_for_key(REDIS, provided) or PLAN_DEFAULT
        except Exception:  # pragma: no cover - defensive fail-open
            plan = PLAN_DEFAULT

    request.state.caller_id = f"key:{provided}"
    request.state.plan = plan
    request.state.scopes = ["simulate", "admin", "cron"]
    request.state.user = None
    request.state.auth_source = "api_key"
    return True


def open_access_allowed(required_scopes: Sequence[str]) -> bool:
    """Return True if open-access traffic is allowed for the requested scopes."""
    if not bool(getattr(settings, "open_access", True)):
        return False
    return not any(scope in {"admin", "cron"} for scope in required_scopes)


def apply_open_access(request: Request) -> None:
    """Populate request.state for anonymous/open-access usage."""
    host = request.client.host if request.client else "0.0.0.0"
    request.state.caller_id = f"anon:{host}"
    request.state.plan = PLAN_DEFAULT
    request.state.scopes = ["simulate"]
    request.state.user = None
    request.state.auth_source = "open_access"


def issue_token(username: str, password: str, requested_scopes: Sequence[str]) -> dict[str, Any]:
    """Validate credentials and return a signed bearer token payload."""
    user = authenticate(username, password)
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    allowed_scopes = set(all_scopes_for(user))
    requested = {scope for scope in requested_scopes if scope}

    if requested and not requested.issubset(allowed_scopes):
        raise HTTPException(status_code=403, detail="Requested scope not permitted for this user")

    scopes = list(requested or (allowed_scopes or {"simulate"}))
    token = create_access_token(user, scopes)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": int(settings.jwt_exp_minutes) * 60,
        "scope": " ".join(scopes),
        "user": {
            "id": user.id,
            "email": user.email,
            "plan": user.plan or PLAN_DEFAULT,
        },
    }


__all__ = [
    "apply_open_access",
    "authenticate_bearer",
    "authorize_api_key",
    "create_access_token",
    "decode_token",
    "issue_token",
    "open_access_allowed",
]
