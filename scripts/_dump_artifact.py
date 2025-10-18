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
    artifact_raw = await redis.get(f"artifact:{RUN_ID}")
    if artifact_raw is None:
        print("artifact missing")
        return
    print("artifact length", len(artifact_raw))
    data = json.loads(artifact_raw)
    keys = list(data.keys())
    print("keys", keys[:20])
    print(json.dumps({k: data.get(k) for k in ("symbol","prob_up_end","median_path","terminal_prices","var_es","model_info")}, indent=2)[:4000])

asyncio.run(main())
