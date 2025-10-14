from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

from passlib.context import CryptContext
from sqlmodel import Field, Session, SQLModel, create_engine, select

try:
    from .db.duck import DB_PATH as _CORE_DB_PATH
except Exception:
    _CORE_DB_PATH = "data/pt.duckdb"

import os


AUTH_DB_PATH = os.getenv("PT_AUTH_DB_PATH", _CORE_DB_PATH)

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
_ENGINE = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


class UserKey(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True, nullable=False)
    hashed_secret: str = Field(nullable=False)
    plan: str = Field(default="free", nullable=False)
    scopes: str = Field(default="", nullable=False)
    is_active: bool = Field(default=True, nullable=False)
    created_at: datetime = Field(default_factory=_now, nullable=False)
    updated_at: datetime = Field(default_factory=_now, nullable=False)
    last_login: Optional[datetime] = None


def _scopes_to_str(scopes: Sequence[str] | None) -> str:
    if not scopes:
        return ""
    unique = sorted({s.strip() for s in scopes if s and s.strip()})
    return " ".join(unique)


def _ensure_engine():
    global _ENGINE
    if _ENGINE is None:
        path = Path(AUTH_DB_PATH).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        _ENGINE = create_engine(f"duckdb:///{path}", connect_args={"read_only": False})
        SQLModel.metadata.create_all(_ENGINE)
    return _ENGINE


def get_session() -> Session:
    engine = _ensure_engine()
    return Session(engine)


def hash_secret(secret: str) -> str:
    return _pwd_context.hash(secret)


def verify_secret(secret: str, hashed: str) -> bool:
    try:
        return _pwd_context.verify(secret, hashed)
    except Exception:
        return False


def upsert_user(
    *,
    email: str,
    secret: Optional[str] = None,
    plan: str = "free",
    scopes: Sequence[str] | None = None,
    is_active: bool = True,
) -> UserKey:
    email_norm = email.strip().lower()
    scopes_str = _scopes_to_str(scopes)
    now = _now()

    with get_session() as session:
        stmt = select(UserKey).where(UserKey.email == email_norm)
        user = session.exec(stmt).first()
        if user:
            if secret:
                user.hashed_secret = hash_secret(secret)
            user.plan = plan
            user.scopes = scopes_str
            user.is_active = is_active
            user.updated_at = now
        else:
            if not secret:
                raise ValueError("secret required when creating user")
            user = UserKey(
                email=email_norm,
                hashed_secret=hash_secret(secret),
                plan=plan,
                scopes=scopes_str,
                is_active=is_active,
                created_at=now,
                updated_at=now,
            )
            session.add(user)
        session.commit()
        session.refresh(user)
        return user


def authenticate(email: str, secret: str) -> Optional[UserKey]:
    email_norm = (email or "").strip().lower()
    if not email_norm or not secret:
        return None
    with get_session() as session:
        stmt = select(UserKey).where(UserKey.email == email_norm, UserKey.is_active == True)  # noqa: E712
        user = session.exec(stmt).first()
        if not user:
            return None
        if not verify_secret(secret, user.hashed_secret):
            return None
        user.last_login = _now()
        user.updated_at = _now()
        session.add(user)
        session.commit()
        session.refresh(user)
        return user


def get_user(user_id: int) -> Optional[UserKey]:
    with get_session() as session:
        return session.get(UserKey, user_id)


def get_user_by_email(email: str) -> Optional[UserKey]:
    email_norm = email.strip().lower()
    with get_session() as session:
        stmt = select(UserKey).where(UserKey.email == email_norm)
        return session.exec(stmt).first()


def all_scopes_for(user: UserKey) -> Iterable[str]:
    if not user.scopes:
        return []
    return [s for s in user.scopes.split() if s]
