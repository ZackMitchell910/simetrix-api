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

RUN_ID = os.environ.get("SIM_RUN_ID", "75311dbee6694a5fa61c180328b596c9")

async def main():
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    run = await redis.get(f"run:{RUN_ID}")
    art = await redis.get(f"artifact:{RUN_ID}")
    print("run state:", run)
    if art:
        try:
            parsed = json.loads(art)
            print("artifact keys:", list(parsed.keys())[:10])
        except Exception as exc:
            print("artifact raw length", len(art), "error", exc)
    else:
        print("artifact missing")

asyncio.run(main())
