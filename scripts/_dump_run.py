import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from redis.asyncio import Redis
from src.core import settings

RUN_ID = os.environ.get("SIM_RUN_ID", "7840579bc49d4c7ab3ebfa180ce662e4")

async def main():
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    run = await redis.get(f"run:{RUN_ID}")
    print("run state: ", run is not None)
    if run:
        print(run)

asyncio.run(main())
