import asyncio
import os
from datetime import datetime

from scripts._env_utils import ensure_env
from src.core import REDIS

ensure_env(
    required=("PT_API_KEY",),
    defaults={"PT_REDIS_URL": os.getenv("PT_REDIS_URL") or "redis://localhost:6379/0"},
)

caller = f"key:{os.environ['PT_API_KEY']}"
scope = "simulate"
ymd = datetime.utcnow().strftime("%Y-%m-%d")


async def reset() -> None:
    await REDIS.delete(f"qt:{scope}:{caller}:{ymd}")
    print("cleared daily counter", caller, ymd)


asyncio.run(reset())
