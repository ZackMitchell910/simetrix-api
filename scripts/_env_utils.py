"""Helpers for configuring environment variables in local maintenance scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Mapping

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv_files() -> None:
    """Load standard .env files without overriding existing values."""
    for env_file in (ROOT / ".env", ROOT / ".env.local"):
        if env_file.exists():
            load_dotenv(env_file, override=False)


def ensure_env(
    required: Iterable[str] = (),
    defaults: Mapping[str, str | None] | None = None,
) -> None:
    """Ensure scripts run with the expected environment configuration."""
    _load_dotenv_files()

    if defaults:
        for key, value in defaults.items():
            if value is not None:
                os.environ.setdefault(key, value)

    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(sorted(missing))
        )
