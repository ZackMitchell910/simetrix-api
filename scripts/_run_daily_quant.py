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
        "PT_REDIS_URL",
        "XAI_API_KEY",
    ),
    defaults={
        "PT_CORS_ORIGINS_RAW": os.getenv("PT_CORS_ORIGINS_RAW")
        or json.dumps(
            [
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "https://simetrix.io",
                "https://www.simetrix.io",
                "https://simetrix.vercel.app",
            ]
        ),
        "PT_DATA_ROOT": os.getenv("PT_DATA_ROOT") or str(ROOT / "data"),
        "PT_DUCKDB_PATH": os.getenv("PT_DUCKDB_PATH") or str(ROOT / "data" / "pt.duckdb"),
        "PT_FS_DUCKDB_PATH": os.getenv("PT_FS_DUCKDB_PATH")
        or str(ROOT / "data" / "feature_store.duckdb"),
        "PT_REDIS_URL": os.getenv("PT_REDIS_URL") or "redis://localhost:6379/0",
        "PT_N_PATHS_MAX": os.getenv("PT_N_PATHS_MAX") or "200000",
        "PT_HORIZON_DAYS_MAX": os.getenv("PT_HORIZON_DAYS_MAX") or "3650",
        "PT_PATHDAY_BUDGET_MAX": os.getenv("PT_PATHDAY_BUDGET_MAX") or "50000000",
        "PT_MAX_ACTIVE_RUNS": os.getenv("PT_MAX_ACTIVE_RUNS") or "4",
        "PT_RUN_TTL_SECONDS": os.getenv("PT_RUN_TTL_SECONDS") or "7200",
        "PT_PREDICTIVE_DEFAULTS": os.getenv("PT_PREDICTIVE_DEFAULTS")
        or json.dumps(
            {
                "X:BTCUSD": {"horizon_days": 365, "n_paths": 10000},
                "NVDA": {"horizon_days": 30, "n_paths": 5000},
            }
        ),
        "PT_EARNINGS_SOURCE": os.getenv("PT_EARNINGS_SOURCE") or "polygon",
        "PT_MACRO_SOURCE": os.getenv("PT_MACRO_SOURCE") or "fred",
        "PT_OPEN_ACCESS": os.getenv("PT_OPEN_ACCESS") or "1",
        "PT_SIM_BIAS": os.getenv("PT_SIM_BIAS") or "1",
        "PT_WARM_JITTER_MS": os.getenv("PT_WARM_JITTER_MS") or "300",
        "XAI_MODEL": os.getenv("XAI_MODEL") or "grok-4-latest",
        "PT_EQUITY_WATCH": os.getenv("PT_EQUITY_WATCH")
        or json.dumps(["NVDA", "MSFT", "GOOGL", "AAPL"]),
        "PT_CRYPTO_WATCH": os.getenv("PT_CRYPTO_WATCH")
        or json.dumps(["BTC-USD", "ETH-USD", "SOL-USD"]),
        "TF_CPP_MIN_LOG_LEVEL": os.getenv("TF_CPP_MIN_LOG_LEVEL") or "2",
    },
)

from src.services.quant_scheduler import run_daily_quant

async def main():
    res = await run_daily_quant(horizon_days=30)
    payload = {
        "as_of": res.get("as_of"),
        "equity": (res.get("equity") or {}).get("symbol"),
        "crypto": (res.get("crypto") or {}).get("symbol"),
        "equity_prob": (res.get("equity") or {}).get("prob_up_end"),
        "crypto_prob": (res.get("crypto") or {}).get("prob_up_end"),
        "equity_top_count": len(res.get("equity_top", [])) if isinstance(res.get("equity_top"), list) else 0,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

asyncio.run(main())
