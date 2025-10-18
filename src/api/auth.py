# src/api/auth.py
from __future__ import annotations
from typing import Sequence

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import OAuth2PasswordBearer, SecurityScopes

from src.api.deps import get_settings
from src.services import auth as auth_service
from src.services.admin import admin_actor, get_admin_session

# Notes:
# - We use the same scope names as the monolith: simulate/admin/cron.
# - Dependencies populate request.state for quota enforcement and logging.
SCOPES = {
    "simulate": "Run and view simulations",
    "admin": "Administrative operations",
    "cron": "Scheduled automation hooks",
}
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", scopes=SCOPES, auto_error=False)


async def require_user(
    request: Request,
    security_scopes: SecurityScopes,
    token: str | None = Depends(oauth2_scheme),
) -> object:
    required = security_scopes.scopes or ["simulate"]
    return await auth_service.authenticate_bearer(request, required, token)


async def require_key(
    request: Request,
    security_scopes: SecurityScopes,
    token: str | None = Depends(oauth2_scheme),
    x_api_key: str | None = None,
    api_key: str | None = None,
) -> bool:
    """Accept OAuth or API key credentials; annotate request.state accordingly."""
    settings = get_settings()
    required = security_scopes.scopes or ["simulate"]
    requires_admin = any(scope == "admin" for scope in required)

    if requires_admin:
        session_payload = get_admin_session(
            request,
            required=bool(request.cookies.get(settings.admin_session_cookie)),
        )
        if session_payload:
            actor = admin_actor(request)
            request.state.caller_id = f"admin:{actor}"
            request.state.plan = "inst"
            request.state.scopes = list(sorted(set(session_payload.get("scopes", []) or ["admin"])))
            request.state.user = None
            request.state.auth_source = "admin_session"
            return True
        if request.cookies.get(settings.admin_session_cookie):
            raise HTTPException(status_code=401, detail="admin_session_invalid")

    if token:
        await auth_service.authenticate_bearer(request, required, token)
        return True

    provided = (x_api_key or api_key or "").strip()
    if provided:
        return await auth_service.authorize_api_key(request, provided)

    if auth_service.open_access_allowed(required):
        auth_service.apply_open_access(request)
        return True

    raise HTTPException(status_code=401, detail="Invalid or missing credentials")
