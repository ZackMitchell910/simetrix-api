from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.security import OAuth2PasswordRequestForm

from src.services import auth as auth_service

router = APIRouter(tags=["auth"])


@router.post("/auth/token", summary="Obtain an OAuth2 access token")
async def issue_token(form_data: OAuth2PasswordRequestForm = Depends()) -> dict[str, object]:
    return auth_service.issue_token(
        form_data.username,
        form_data.password,
        form_data.scopes or [],
    )
