# src/api/deps.py
from __future__ import annotations
from typing import Optional

from fastapi import HTTPException
from redis.asyncio import Redis as AsyncRedis

from src.core import REDIS, SIM_DISPATCHER, settings
from src.task_queue import TaskDispatcher


def get_settings():
    return settings


async def get_redis() -> AsyncRedis:
    if REDIS is None:
        raise HTTPException(status_code=503, detail="Redis unavailable")
    return REDIS


def get_dispatcher() -> Optional[TaskDispatcher]:
    return SIM_DISPATCHER
