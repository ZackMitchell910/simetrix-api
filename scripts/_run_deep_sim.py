import asyncio
import json
import os
import sys
from pathlib import Path

from scripts._env_utils import ensure_env

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ensure_env(
    required=(
        "PT_API_KEY",
        "PT_FRED_API_KEY",
        "PT_POLYGON_KEY",
    ),
    defaults={
        "PT_DATA_ROOT": os.getenv("PT_DATA_ROOT") or str(ROOT / "data"),
        "PT_DUCKDB_PATH": os.getenv("PT_DUCKDB_PATH") or str(ROOT / "data" / "pt.duckdb"),
        "PT_FS_DUCKDB_PATH": os.getenv("PT_FS_DUCKDB_PATH")
        or str(ROOT / "data" / "feature_store.duckdb"),
        "PT_OPEN_ACCESS": os.getenv("PT_OPEN_ACCESS") or "1",
        "PT_SIM_BIAS": os.getenv("PT_SIM_BIAS") or "1",
        "TF_CPP_MIN_LOG_LEVEL": os.getenv("TF_CPP_MIN_LOG_LEVEL") or "2",
    },
)

from src.services import quant_adapters
from src.services.quant_scheduler import quant_allow, quant_consume
from src.core import REDIS


async def main():
    symbol, summary = await quant_adapters.run_mc_for(
        "RIOT",
        horizon=50,
        paths=1000,
        redis=REDIS,
        quant_allow=quant_allow,
        quant_consume=quant_consume,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

asyncio.run(main())
