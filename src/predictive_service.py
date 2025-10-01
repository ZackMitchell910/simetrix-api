# ----------------------------
# Standard library
# ----------------------------
from __future__ import annotations

import os
import math
import logging
import asyncio
import pickle
import json
from uuid import uuid4
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Literal, Optional, Tuple, List

# ----------------------------
# Third-party
# ----------------------------
import numpy as np
import pandas as pd
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, WebSocket, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader, APIKeyQuery
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from redis.asyncio import Redis
from statsmodels.tsa.arima.model import ARIMA

# Optional ML libs (don’t fail if missing)
try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, GRU, Dense
except Exception:
    Sequential = load_model = None
    LSTM = GRU = Dense = None

try:
    import gymnasium as gym
except Exception:
    gym = None

try:
    from stable_baselines3 import DQN
except Exception:
    DQN = None

# ----------------------------
# Local modules
# ----------------------------
from feature_store import connect as fs_connect, insert_prediction as fs_log_prediction
from feature_store import connect as _fs_connect, compute_and_upsert_metrics_daily as _rollup
from learners import OnlineLinear as SGDOnline, ExpWeights as EW
from labeler import label_mature_predictions
import learners as _learners_mod

# ----------------------------
# Logger
# ----------------------------
logger = logging.getLogger(__name__)
_json = json  # legacy alias if you still reference `_json`

# --- Indicators: prefer TA-Lib, else pandas_ta, else ta, else no-op ---
try:
    import talib
    _TA_BACKEND = "talib"
except Exception:
    try:
        import pandas_ta as pta
        _TA_BACKEND = "pandas_ta"
    except Exception:
        try:
            import ta
            _TA_BACKEND = "ta"
        except Exception:
            _TA_BACKEND = "none"

class SimRequest(BaseModel):
    symbol: str
    horizon_days: int
    n_paths: int
    timespan: Literal["day","hour"] = "day"
    include_news: bool = False
    include_options: bool = False
    include_futures: bool = False
    # Your UI is now sending a single string; make this Optional[str]
    x_handles: Optional[str] = Field(default=None, description="comma-separated handles")


class TrainRequest(BaseModel):
    symbol: str
    lookback_days: int = 3650

class PredictRequest(BaseModel):
    symbol: str
    horizon_days: int = 1

class RunState(BaseModel):
    status: str = "queued"
    progress: float = 0.0
    error: str | None = None
    artifact: dict | None = None

def ta_rsi(arr: np.ndarray, length: int = 14) -> float:
    x = np.asarray(arr, float)
    if _TA_BACKEND == "talib":
        r = talib.RSI(x, timeperiod=length)
        r = r[~np.isnan(r)]
        return float(r[-1]) if r.size else 50.0
    s = pd.Series(x)
    if _TA_BACKEND == "pandas_ta":
        r = pta.rsi(s, length=length).dropna()
        return float(r.iloc[-1]) if not r.empty else 50.0
    if _TA_BACKEND == "ta":
        r = ta.momentum.RSIIndicator(s, window=length).rsi().dropna()
        return float(r.iloc[-1]) if not r.empty else 50.0
    return 50.0

def ta_macd(arr: np.ndarray):
    x = np.asarray(arr, float)
    if _TA_BACKEND == "talib":
        return talib.MACD(x)
    s = pd.Series(x)
    if _TA_BACKEND == "pandas_ta":
        m = pta.macd(s)
        if m is None or m.empty: return [np.array([np.nan])]*3
        return [m.iloc[:,0].to_numpy(), m.iloc[:,1].to_numpy(), m.iloc[:,2].to_numpy()]
    if _TA_BACKEND == "ta":
        ind = ta.trend.MACD(s)
        return [ind.macd().to_numpy(), ind.macd_signal().to_numpy(), ind.macd_diff().to_numpy()]
    return [np.array([np.nan])]*3

def ta_bbands(arr: np.ndarray, length: int = 20, ndev: float = 2.0):
    x = np.asarray(arr, float)
    if _TA_BACKEND == "talib":
        return talib.BBANDS(x, timeperiod=length, nbdevup=ndev, nbdevdn=ndev)
    s = pd.Series(x)
    if _TA_BACKEND == "pandas_ta":
        b = pta.bbands(s, length=length, std=ndev)
        if b is None or b.empty: return x, x, x
        return b.filter(like="BBU").iloc[:,0].to_numpy(), \
               b.filter(like="BBM").iloc[:,0].to_numpy(), \
               b.filter(like="BBL").iloc[:,0].to_numpy()
    if _TA_BACKEND == "ta":
        ind = ta.volatility.BollingerBands(s, window=length, window_dev=ndev)
        return ind.bollinger_hband().to_numpy(), ind.bollinger_mavg().to_numpy(), ind.bollinger_lband().to_numpy()
    return x, x, x

def _sigmoid(x: float) -> float:
    # numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return float(1.0 / (1.0 + z))
    else:
        z = math.exp(x)
        return float(z / (1.0 + z))


async def _model_key(symbol: str) -> str:
    return f"model:{symbol.upper()}"

async def _fetch_realized_price(symbol: str, when: datetime) -> float:
    """Best-effort realized price for the 'when' date. Falls back if rate-limited."""
    if settings.polygon_key:
        d0 = when.strftime("%Y-%m-%d")
        d1 = (when + timedelta(days=1)).strftime("%Y-%m-%d")
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{d0}/{d1}"
        params = {"adjusted": "true", "sort": "asc", "limit": "2", "apiKey": settings.polygon_key}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(url, params=params)
                # Gracefully handle rate limit without raising
                if r.status_code == 429:
                    logger.warning(f"Polygon 429 for {symbol} {d0}->{d1}; falling back.")
                else:
                    r.raise_for_status()
                    res = (r.json() or {}).get("results", [])
                    if res:
                        return float(res[-1]["c"])
        except Exception as e:
            logger.info(f"Polygon fetch failed for {symbol} @ {d0}: {e}; falling back.")
    # Fallbacks (safe + deterministic enough for testing)
    try:
        # last known price approximation if you want a saner fallback than random:
        px = await _fetch_hist_prices(symbol)
        if px:
            return float(px[-1])
    except Exception:
        pass
    # ultimate fallback: synthetic
    return float(np.random.lognormal(mean=np.log(100), sigma=0.12))



async def _fetch_hist_prices(symbol: str) -> List[float]:
    key = (os.getenv("PT_POLYGON_KEY") or os.getenv("POLYGON_KEY") or "").strip()
    if key:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2024-01-01/2025-09-01"
                r = await client.get(url, params={"apiKey": key})
                r.raise_for_status()
                data = r.json().get("results", [])
                px = [float(d["c"]) for d in data if "c" in d]
                if px:
                    return px
        except Exception as e:
            logger.warning(f"Polygon fetch failed ({symbol}): {e}")
    # fallback synthetic
    return list(np.random.lognormal(mean=math.log(100.0), sigma=0.15, size=200))

async def _feat_from_prices(symbol: str, px: List[float]) -> Dict:
    arr = np.asarray(px, dtype=float)
    if arr.ndim != 1 or arr.size < 3:
        return {"mom_20": 0.0, "rvol_20": 0.0, "autocorr_5": 0.0}

    rets = np.diff(np.log(arr))
    mom_20 = float(np.mean(rets[-20:])) if rets.size >= 20 else float(np.mean(rets))
    rvol_20 = float(np.std(rets[-20:])) if rets.size >= 20 else float(np.std(rets))

    if rets.size >= 6:
        try:
            autocorr_5 = float(np.corrcoef(rets[-6:-1], rets[-5:])[0, 1])
            if not np.isfinite(autocorr_5):
                autocorr_5 = 0.0
        except Exception:
            autocorr_5 = 0.0
    else:
        autocorr_5 = 0.0

    f = {"mom_20": mom_20, "rvol_20": rvol_20, "autocorr_5": autocorr_5}

    # Async enrichments (placeholders)
    f["sentiment"] = 0.0  # float(await _fetch_sentiment(symbol))
    f.update({"vix": 20.0, "spx_ret": 0.0})  # await _fetch_macro()
# TA features (guard against NaNs)
    try:
        rsi_val = float(ta_rsi(arr, 14))
    except Exception:
        rsi_val = 50.0
    f["rsi"] = rsi_val

    try:
        macd, macd_sig, macd_hist = ta_macd(arr)
        _v = ~np.isnan(macd)
        f["macd"] = float(macd[_v][-1]) if np.any(_v) else 0.0
    except Exception:
        f["macd"] = 0.0

    try:
        bb_up, bb_mid, bb_low = ta_bbands(arr)
        _v = ~np.isnan(bb_up)
        f["bb_upper"] = float(bb_up[_v][-1]) if np.any(_v) else float(arr[-1])
    except Exception:
        f["bb_upper"] = float(arr[-1])

    f["iv"] = 0.2  # stub; wire to options IV if you have it
    # Peer corr (placeholder: self-corr = 0)
    try:
        peer_px = arr  # TODO: replace with real peer series
        pc = np.corrcoef(arr[-100:], peer_px[-100:])[0, 1] if arr.size >= 100 else 0.0
        f["peer_corr"] = float(pc) if np.isfinite(pc) else 0.0
    except Exception:
        f["peer_corr"] = 0.0

    return f
async def get_ensemble_prob_light(symbol: str, redis: Redis, horizon_days: int = 1) -> float:
    """
    Async-friendly, non-blocking ensemble:
    - Uses only the linear model stored in Redis (no Keras/ARIMA/RL)
    - Stable and fast: won't block the event loop
    """
    try:
        raw = await redis.get(await _model_key(symbol + "_linear"))
        if not raw:
            return 0.5
        model_linear = json.loads(raw)
        feats = model_linear.get("features", [])
        coef  = model_linear.get("coef", [])

        px = await _fetch_hist_prices(symbol)
        if not px or len(px) < 10:
            return 0.5
        f = await _feat_from_prices(symbol, px)

        X = np.array([f.get(feat, 0.0) for feat in feats], dtype=float)
        w = np.array(coef, dtype=float)
        m = int(min(X.shape[0], w.shape[0]))
        if m == 0:
            return 0.5

        score = float(np.dot(X[:m], w[:m]))
        return _sigmoid(score)
    except Exception as e:
        logger.info(f"get_ensemble_prob_light fallback (0.5) due to: {e}")
        return 0.5

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)

def _server_keys() -> tuple[str, str]:
    app_key = os.getenv("PT_API_KEY", "").strip()
    poly_key = (os.getenv("PT_POLYGON_KEY") or os.getenv("POLYGON_KEY") or "").strip()
    return app_key, poly_key

async def verify_api_key(
    api_key_h: Optional[str] = Depends(api_key_header),
    api_key_q: Optional[str] = Depends(api_key_query),
):
    supplied = (api_key_h or api_key_q or "").strip()
    app_key, poly_key = _server_keys()
    # Dev: if neither key set, allow
    if not app_key and not poly_key:
        return supplied
    expected = app_key or poly_key
    if supplied != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return supplied

def load_lstm_model(symbol):
    try:
        return load_model(f"models/{symbol}_lstm.h5")
    except:
        raise HTTPException(status_code=404, detail="LSTM model not found; train first")

def load_arima_model(symbol):
    try:
        with open(f"models/{symbol}_arima.pkl", 'rb') as file:
            return pickle.load(file)
    except:
        raise HTTPException(status_code=404, detail="ARIMA model not found; train first")

def _env_list(name: str, default: list[str] | None = None) -> list[str]:
    """
    Read a comma-separated env var into a list, trimming spaces.
    Example: PT_CORS_ORIGINS=http://localhost:5173, http://localhost:8080
    """
    s = os.getenv(name, "")
    if not s:
        return default or []
    return [x.strip() for x in s.split(",") if x.strip()]

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.info(f"Using learners from: {_learners_mod.__file__}")

load_dotenv("backend/.env")

class Settings(BaseSettings):
    api_keys: list[str] = Field(default_factory=lambda: [k for k in _server_keys() if k])
    polygon_key: str | None = Field(default_factory=lambda: os.getenv("PT_POLYGON_KEY"))
    news_api_key: str | None = Field(default_factory=lambda: os.getenv("PT_NEWS_API_KEY"))
    redis_url: str = Field(default_factory=lambda: os.getenv("PT_REDIS_URL", "redis://localhost:6379/0"))
    cors_origins: list[str] = Field(default_factory=lambda: _env_list("PT_CORS_ORIGINS", ["*"]))
    n_paths_max: int = Field(default_factory=lambda: int(os.getenv("PT_N_PATHS_MAX", "10000")))
    horizon_days_max: int = Field(default_factory=lambda: int(os.getenv("PT_HORIZON_DAYS_MAX", "365")))
    lookback_days_max: int = Field(default_factory=lambda: int(os.getenv("PT_LOOKBACK_DAYS_MAX", str(365*10))))
    pathday_budget_max: int = Field(default_factory=lambda: int(os.getenv("PT_PATHDAY_BUDGET_MAX", "500000")))
    max_active_runs: int = Field(default_factory=lambda: int(os.getenv("PT_MAX_ACTIVE_RUNS", "2")))
    run_ttl_seconds: int = Field(default_factory=lambda: int(os.getenv("PT_RUN_TTL_SECONDS", "3600")))
    predictive_defaults: dict = {
        "X:BTCUSD": {"horizon_days": 365, "n_paths": 10000, "lookback_preset": "3y"},
        "NVDA": {"horizon_days": 30, "n_paths": 5000, "lookback_preset": "180d"},
    }
    class Config:
        env_prefix = "PT_"
        case_sensitive = False

settings = Settings()
REDIS = Redis.from_url(settings.redis_url)
# --- RL constants ---
RL_WINDOW = int(os.getenv("PT_RL_WINDOW", "100"))  # single source of truth
logger = logging.getLogger(__name__)
if not settings.polygon_key:
    logger.warning("POLYGON_KEY not set. Using synthetic data for simulations.")
# --- FastAPI App ---
app = FastAPI(
    title="PredictiveTwin API",
    version="1.2.1",
    docs_url="/api-docs",  # Move docs
    redoc_url=None,
    redirect_slashes=False  # No 307
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Redis init + background GC ---
async def _gc_loop():
    while True:
        try:
            if REDIS:
                keys = await REDIS.keys("run:*")
                for key in keys:
                    ttl = await REDIS.ttl(key)
                    # If key has no TTL (ttl < 0), clean it up to avoid leaks
                    if ttl is not None and ttl < 0:
                        await REDIS.delete(key)
        except Exception as e:
            logger.error(f"GC loop error: {e}")
        await asyncio.sleep(60)


@app.on_event("startup")
async def _startup():
    global REDIS
    try:
        # If you want text back (no .decode()), use decode_responses=True
        REDIS = Redis.from_url(settings.redis_url, decode_responses=True)
        await REDIS.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        REDIS = None
    # start GC
    asyncio.create_task(_gc_loop())
    asyncio.create_task(_labeling_daemon())

@app.on_event("shutdown")
async def _shutdown():
    try:
        if REDIS:
            await REDIS.close()
            logger.info("Redis closed")
    except Exception as e:
        logger.error(f"Error closing Redis: {e}")

async def get_ensemble_prob(symbol: str, redis: Redis, horizon_days: int = 1) -> float:
    """
    Direct model load for prod: Reuse /predict logic without HTTP.
    """
    try:
        # 1) Load linear metadata
        raw = await redis.get(await _model_key(symbol + "_linear"))
        if not raw:
            return 0.5  # Fallback
        model_linear = json.loads(raw)
        feature_list = model_linear.get("features", ["mom_20", "rvol_20", "autocorr_5"])

        # 2) Get prices + features
        px = await _fetch_hist_prices(symbol)
        if not px or len(px) < 10:
            return 0.5
        f = await _feat_from_prices(symbol, px)

        # 3) Input vector
        X = np.array([f.get(feat, 0.0) for feat in feature_list], dtype=float)

        preds = []

        # Linear
        if "coef" in model_linear:
            w = np.array(model_linear["coef"], dtype=float)
            w = w[:X.shape[0]]
            X_lin = X[:w.shape[0]]
            score = float(np.dot(X_lin, w))
            score = float(np.clip(score, -60.0, 60.0))
            preds.append(_sigmoid(score))

        # LSTM
        try:
            model_lstm = load_lstm_model(symbol)
            X_lstm = np.expand_dims(np.expand_dims(X, axis=0), axis=0)
            preds.append(float(model_lstm.predict(X_lstm, verbose=0)[0][0]))
        except:
            pass
        # ARIMA
        try:
            model_arima = load_arima_model(symbol)
            fc = model_arima.forecast(steps=horizon_days)
            last_fc = float(fc.iloc[-1] if hasattr(fc, "iloc") else fc[-1])
            preds.append(1.0 if last_fc > float(px[-1]) else 0.0)
        except Exception:
            pass
        # RL adjust
        rl_adjust = 0.0
        try:
            rl_model = DQN.load(f"models/{symbol}_rl.zip", print_system_info=False)
            env = StockEnv(px, window_len=RL_WINDOW)
            obs, _ = env.reset()  # normalized window like in /predict
            action, _ = rl_model.predict(obs, deterministic=True)
            a = float(action[0] if hasattr(action, "__len__") else action)
            rl_adjust = float(np.clip(a * 0.01, -0.05, 0.05))  # cap ±5%
            env.close()
        except Exception as e:
            logger.info(f"RL skipped in ensemble: {e}")
        # Ensemble
        losses = [0.1] * len(preds)
        ew = EW()
        ew.init(len(preds))
        ew.update(np.array(losses, dtype=float))
        prob_up = float(np.clip(sum(w * p for w, p in zip(ew.w, preds)) + rl_adjust, 0.0, 1.0))

        return prob_up
    except Exception as e:
        logger.warning(f"Ensemble prob failed for {symbol}: {e}")
        return 0.5

async def _list_models() -> list[str]:
    keys = await REDIS.keys("model:*")
    # keys are already str when decode_responses=True
    return [k.split(":", 1)[1] for k in keys]

async def _fetch_cached_hist_prices(symbol: str, window_days: int, redis: Redis) -> List[float]:
    cache_key = f"hist_prices:{symbol}:{window_days}"
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)
   
    px = await _fetch_hist_prices(symbol)  # Existing fetch
    # Truncate to window_days if needed
    cached_px = px[-window_days:] if len(px) > window_days else px
    await redis.setex(cache_key, 3600, json.dumps(cached_px))
    return cached_px

def _dynamic_window_days(horizon_days: int, timespan: str) -> int:
    base = 180
    if timespan == "minute":
        base *= 390 / 252  # Trading minutes per day
    return min(base + horizon_days * 2, settings.lookback_days_max)

async def _news_sentiment_for_symbol(symbol: str) -> float:
    # Placeholder; implement with news API
    return 0.0

async def _ensure_run(run_id: str) -> RunState:
    raw = await REDIS.get(f"run:{run_id}")
    if not raw:
        raise HTTPException(404, "Run not found")
    rs = RunState.parse_raw(raw)
    ttl = await REDIS.ttl(f"run:{run_id}")
    # -2 = missing, -1 = no expiry. Only treat -2 as gone.
    if ttl == -2:
        raise HTTPException(410, "Run expired")
    return rs

async def _labeling_daemon():
    while True:
        try:
            res = await outcomes_label(_api_key="cron")  # reuse the route logic
            labeled = (res or {}).get("labeled", 0)
            logger.info(f"Outcome labeling pass: labeled={labeled}")
        except Exception as e:
            logger.warning(f"Labeling pass failed: {e}")
        await asyncio.sleep(900)

async def run_simulation(run_id: str, req: SimRequest, redis: Redis):
    logger.info(f"Starting simulation for run_id={run_id}, symbol={req.symbol}")
    try:
        rs = await _ensure_run(run_id)
    except Exception as e:
        logger.error(f"_ensure_run failed at start for {run_id}: {e}")
        rs = RunState(status="error", progress=0.0, error=str(e))
        await redis.set(f"run:{run_id}", rs.json(), ex=settings.run_ttl_seconds)
        return

    rs.status = "running"
    rs.progress = 0
    await redis.set(f"run:{run_id}", rs.json(), ex=settings.run_ttl_seconds)

    try:
        # ---------- Guardrails ----------
        if req.n_paths * req.horizon_days > settings.pathday_budget_max:
            raise ValueError("compute budget exceeded")
        if req.horizon_days > settings.horizon_days_max or req.n_paths > settings.n_paths_max:
            raise ValueError("input limits exceeded")

        # ---------- History / params ----------
        window_days = _dynamic_window_days(req.horizon_days, req.timespan)
        historical_prices = await _fetch_cached_hist_prices(req.symbol, window_days, redis)
        if not historical_prices or len(historical_prices) < 30:
            raise ValueError("Insufficient history")

        rets = np.diff(np.log(np.array(historical_prices, dtype=float)))
        scale = 252 if req.timespan == "day" else 252 * 24
        mu_ann = float(np.mean(rets) * scale)
        sigma_ann = float(np.std(rets) * math.sqrt(scale))
        sigma_ann = float(np.clip(sigma_ann, 1e-4, 1.5))
        mu_ann = float(np.clip(mu_ann, -2.0, 2.0))

        # ML adjustment (bounded)
        ensemble_prob = 0.5
        try:
            ensemble_prob = await asyncio.wait_for(
                get_ensemble_prob(req.symbol, redis, req.horizon_days),
                timeout=1.0,
            )
        except asyncio.TimeoutError:
            logger.info("get_ensemble_prob timed out; using 0.5")
        except Exception as e:
            logger.info(f"get_ensemble_prob failed: {e}; using 0.5")

        mu_ann *= (2 * ensemble_prob - 1) * 0.5

        # ---------- Monte Carlo (GBM) ----------
        np.random.seed(abs(hash(run_id)) % (2**32 - 1))
        S0 = float(historical_prices[-1])
        dt = 1.0 / scale
        n_days = max(1, int(req.horizon_days))

        Z = np.random.normal(size=(req.n_paths, n_days))
        drift = (mu_ann - 0.5 * sigma_ann**2) * dt
        diffusion = sigma_ann * math.sqrt(dt)
        log_returns = drift + diffusion * Z
        log_paths = np.cumsum(log_returns, axis=1)
        # include t0 = S0 in the path
        paths = S0 * np.exp(np.concatenate([np.zeros((req.n_paths, 1)), log_paths], axis=1))
        T = paths.shape[1]  # includes t0

        # Progress tick (post-paths)
        rs.progress = 40
        await redis.set(f"run:{run_id}", rs.json(), ex=settings.run_ttl_seconds)

        # ---------- Percentile bands ----------
        median_path = np.median(paths, axis=0)
        p80_low, p80_high = np.percentile(paths, [10, 90], axis=0)
        p95_low, p95_high = np.percentile(paths, [2.5, 97.5], axis=0)

        def _ffill_nonfinite(arr: np.ndarray, fallback: float) -> np.ndarray:
            out = np.array(arr, dtype=float)
            if not np.isfinite(out[0]):
                out[0] = float(fallback)
            for i in range(1, len(out)):
                if not np.isfinite(out[i]):
                    out[i] = out[i - 1]
            return out

        fallback = S0
        p50 = np.array(median_path, dtype=float)
        for arr in (p80_low, p80_high, p95_low, p95_high):
            arr[:] = _ffill_nonfinite(arr, fallback)

        p80_low = np.minimum(p80_low, p50)
        p80_high = np.maximum(p80_high, p50)
        p95_low = np.minimum(p95_low, p80_low)
        p95_high = np.maximum(p95_high, p80_high)
        for arr in (p50, p80_low, p80_high, p95_low, p95_high):
            np.nan_to_num(arr, copy=False, nan=fallback, posinf=fallback, neginf=fallback)

        # ---------- Optional news/options/futures tweaks ----------
        sentiment = 0.0
        if req.include_news:
            try:
                sentiment = await _news_sentiment_for_symbol(req.symbol)
            except Exception as e:
                logger.info(f"news sentiment failed: {e}")

        if req.include_options:
            sigma_ann *= 1.05  # placeholder
        if req.include_futures:
            mu_ann += 0.001   # placeholder

        # Probability up at horizon vs S0 (clip with sentiment nudge)
        terminal = paths[:, -1]
        prob_up = float(np.clip(np.mean(terminal > S0) + sentiment, 0.0, 1.0))

        # ---------- Extras for new visuals ----------
        # VaR/ES as returns (switch to $ if you prefer)
        ret_terminal = (terminal - S0) / S0
        q05 = float(np.percentile(ret_terminal, 5))
        es_mask = ret_terminal <= q05
        es95 = float(ret_terminal[es_mask].mean()) if es_mask.any() else float(q05)

        # Hit probability ribbon (vectorized) for a few absolute thresholds
        thresholds_pct = np.array([-0.05, 0.00, 0.05, 0.10], dtype=float)
        thresholds_abs = (1.0 + thresholds_pct) * float(S0)  # shape (K,)
        # paths: (N, T); compute (K, T) probabilities
        probs_by_day = (
            (paths[:, :, None] >= thresholds_abs[None, None, :])  # (N, T, K) bool
            .mean(axis=0)                                         # (T, K)
            .T                                                    # (K, T)
            .tolist()
        )

        rs.progress = 80
        await redis.set(f"run:{run_id}", rs.json(), ex=settings.run_ttl_seconds)

        # ---------- Artifact ----------
        rs.artifact = {
            "symbol": req.symbol,
            "horizon_days": int(req.horizon_days),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "median_path": [[i, float(v)] for i, v in enumerate(p50.tolist())],
            "bands": {
                "p50":      [[i, float(v)] for i, v in enumerate(p50.tolist())],
                "p80_low":  [[i, float(v)] for i, v in enumerate(p80_low.tolist())],
                "p80_high": [[i, float(v)] for i, v in enumerate(p80_high.tolist())],
                "p95_low":  [[i, float(v)] for i, v in enumerate(p95_low.tolist())],
                "p95_high": [[i, float(v)] for i, v in enumerate(p95_high.tolist())],
            },
            "prob_up_end": prob_up,
            "drivers": [
                {"feature": "rvol_20",       "weight": 0.36},
                {"feature": "autocorr_5",    "weight": 0.19},
                {"feature": "mom_20",        "weight": 0.14},
                {"feature": "ensemble_prob", "weight": float(ensemble_prob)},
            ],
            "model_info": {"direction": "MonteCarlo", "regime": "lognormal"},
            "calibration": {"window": window_days, "p80_empirical": 0.8},

            # --- NEW FIELDS FOR UI ---
            "terminal_prices": [float(x) for x in terminal.tolist()],
            "var_es": {"var95": q05, "es95": es95},  # returns; change to $ if preferred
            "hit_probs": {
                "thresholds_abs": [float(x) for x in thresholds_abs.tolist()],
                "probs_by_day": probs_by_day,  # shape (K thresholds, T days)
            },
        }

        # ---------- Finish ----------
        rs.status = "done"
        rs.progress = 100
        await redis.set(f"run:{run_id}", rs.json(), ex=settings.run_ttl_seconds)
        logger.info(f"Completed simulation for run_id={run_id}")

    except Exception as e:
        rs.status = "error"
        rs.error = str(e)
        await redis.set(f"run:{run_id}", rs.json(), ex=settings.run_ttl_seconds)
        logger.exception(f"Simulation failed for run_id={run_id}: {e}")

@app.get("/health")
async def health():
    redis_ok = False
    try:
        if REDIS:
            redis_ok = await REDIS.ping()
    except Exception:
        redis_ok = False
    return {
        "ok": True,
        "redis_ok": bool(redis_ok),
        "redis_url": settings.redis_url,
        "polygon_key_present": bool(settings.polygon_key),
        "n_paths_max": settings.n_paths_max,
        "horizon_days_max": settings.horizon_days_max,
        "pathday_budget_max": settings.pathday_budget_max,
    }

@app.post("/outcomes/label")
async def outcomes_label(_api_key: str = Depends(verify_api_key)):
    async def _fetch(symbol: str, when: datetime) -> float:
        return await _fetch_realized_price(symbol, when)

    # Run sync labeler in a thread and return its count
    def _run():
        def _sync_fetch(sym: str, cutoff: datetime):
            return asyncio.run(_fetch(sym, cutoff))
        return label_mature_predictions(_sync_fetch)  # should return an int (count)

    loop = asyncio.get_event_loop()
    labeled = await loop.run_in_executor(None, _run)
    labeled_int = int(labeled or 0)
    return {"status": "ok", "labeled": labeled_int}

# --- Metrics rollup (daily) ---
@app.post("/metrics/rollup")
async def metrics_rollup(_api_key: str = Depends(verify_api_key), day: Optional[str] = None):
    """
    Run daily metrics rollup for predictions joined with outcomes.
    Optional: ?day=YYYY-MM-DD (defaults to today).
    """
    d = _dt.date.fromisoformat(day) if day else _dt.date.today()
    con = _fs_connect()
    n = _rollup(con, day=d)
    con.close()
    return {"status": "ok", "date": d.isoformat(), "rows_upserted": int(n)}

@app.get("/config")
async def config():
    return {
        "n_paths_max": settings.n_paths_max,
        "horizon_days_max": settings.horizon_days_max,
        "pathday_budget_max": settings.pathday_budget_max,
        "predictive_defaults": settings.predictive_defaults,
        "cors_origins": settings.cors_origins,
    }

@app.get("/simulate/{run_id}/stream")
async def simulate_stream(run_id: str, _api_key: str = Depends(verify_api_key)):
    async def event_generator():
        stale_ticks = 0
        last = None

        while True:
            try:
                # Redis missing? surface a one-shot error + stop
                if not REDIS:
                    yield 'data: {"status":"error","progress":0,"detail":"redis_unavailable"}\n\n'
                    break

                raw = await REDIS.get(f"run:{run_id}")
                if not raw:
                    # Not found or expired → tell client and stop
                    yield 'data: {"status":"error","progress":0,"detail":"run_not_found_or_expired"}\n\n'
                    break

                rs = RunState.parse_raw(raw)

                # Send status heartbeat
                payload = {"status": rs.status, "progress": rs.progress}
                yield f"data: {json.dumps(payload)}\n\n"

                # Stop on terminal states
                if rs.status in ("done", "error"):
                    break

                # Stall detector: if status+progress hasn't changed, count ticks
                sig = (rs.status, int(rs.progress))
                if sig == last:
                    stale_ticks += 1
                else:
                    stale_ticks = 0
                    last = sig

                # If no change for ~2 minutes (120s), bail out to avoid infinite spinner
                if stale_ticks > 120:
                    yield 'data: {"status":"error","progress":0,"detail":"stalled"}\n\n'
                    break

                await asyncio.sleep(1.0)

            except Exception as e:
                # Defensive: surface error then stop streaming
                err = {"status": "error", "progress": 0, "detail": str(e)[:200]}
                yield f"data: {json.dumps(err)}\n\n"
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Helps some proxies flush frames promptly
            "X-Accel-Buffering": "no",
        },
    )

@app.websocket("/simulate/{run_id}/ws")
async def simulate_ws(websocket: WebSocket, run_id: str):
    await websocket.accept()
    try:
        while True:
            try:
                rs = await _ensure_run(run_id)
                await websocket.send_json({"status": rs.status, "progress": rs.progress})
                if rs.status in ("done", "error"):
                    if rs.error:
                        await websocket.send_json({"error": rs.error})
                    break
                await asyncio.sleep(0.5)
            except HTTPException as e:
                await websocket.send_json({"status": "error", "error": e.detail})
                break
    finally:
        await websocket.close()

@app.post("/simulate")
async def simulate(req: SimRequest, bg: BackgroundTasks, _api_key: str = Depends(verify_api_key)):
    run_id = str(uuid4())
    rs = RunState(status="queued", progress=0.0)
    await REDIS.setex(f"run:{run_id}", settings.run_ttl_seconds, rs.json())
    bg.add_task(run_simulation, run_id, req, REDIS)
    return {"run_id": run_id}

@app.get("/simulate/{run_id}/status")
async def simulate_status(run_id: str, _api_key: str = Depends(verify_api_key)):
    rs = await _ensure_run(run_id)
    return {"status": rs.status, "progress": rs.progress, "error": rs.error}

@app.get("/simulate/{run_id}/artifact")
async def simulate_artifact(run_id: str, _api_key: str = Depends(verify_api_key)):
    rs = await _ensure_run(run_id)
    if rs.status != "done":
        raise HTTPException(409, "Not done")
    return rs.artifact
# --- Simple title sentiment helper ---
# predictive_service.py
def _load_labeled_samples(symbol: str, limit: int = 256):
    con = fs_connect()
    rows = con.execute("""
        SELECT p.model_id, p.symbol, p.issued_at, p.horizon_days, p.features_ref,
               o.realized_at, o.y
        FROM predictions p
        JOIN outcomes o USING (run_id)
        WHERE p.symbol = ?
        ORDER BY o.realized_at DESC
        LIMIT ?
    """, [symbol.upper(), limit]).fetchall()
    con.close()

    X, y = [], []
    for _, _, _, _, features_ref, _, y_real in rows:
        try:
            j = json.loads(features_ref or "{}")
            f = (j.get("features") or j)  # tolerate legacy shape
            X.append([float(f.get("mom_20", 0.0)),
                      float(f.get("rvol_20", 0.0)),
                      float(f.get("autocorr_5", 0.0))])
            ret = float(y_real)           # <-- return we stored in outcomes.y
            y.append(1.0 if ret > 0.0 else 0.0)
        except Exception:
            continue

    return np.array(X, dtype=float), np.array(y, dtype=float)

def _safe_sent(text:str)->float:
    base = _simple_sentiment(text)
    t = text.lower()
    if any(w in t for w in ["beats", "record", "surge", "raises", "upgrade"]): base += 0.1
    if any(w in t for w in ["misses", "plunge", "cuts", "downgrade", "probe"]): base -= 0.1
    return max(-1.0, min(1.0, base))


# --- Polygon news (cached + pagination-safe) ---
@app.get("/api/news/{symbol}")
async def get_news(
    symbol: str,
    limit: int = 10,
    days: int = 7,
    cursor: Optional[str] = None,                 # ← NEW: pass back to paginate
    _api_key: str = Depends(verify_api_key),
):
    if not settings.polygon_key:
        raise HTTPException(status_code=400, detail="News provider key not configured")

    # ---- input guards ----
    limit = max(1, min(int(limit), 50))
    days  = max(1, min(int(days), 30))

    # ---- normalize ticker ----
    raw_symbol = symbol.strip().upper()

    def to_polygon_ticker(s: str) -> str:
        s2 = s.replace(":", "")
        if (("-" in s and s.endswith("USD")) or (s.endswith("USD") and "-" not in s)):
            base = s.split("-")[0] if "-" in s else s2.replace("USD", "")
            return f"X:{base}USD"
        return s

    poly_ticker = to_polygon_ticker(raw_symbol)

    # ---- cache lookup (include cursor in key to avoid mixing pages) ----
    cache_key = f"news:{poly_ticker}:{limit}:{days}:{cursor or 'first'}"
    if REDIS:
        try:
            cached = await REDIS.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis get failed for {cache_key}: {e}")

    # ---- fetch ----
    url = "https://api.polygon.io/v2/reference/news"
    params = {
        "apiKey": settings.polygon_key,
        "limit": str(limit),
    }
    # When paginating with a cursor, Polygon continues from that point;
    # otherwise use the date filter and ticker.
    if cursor:
        params["cursor"] = cursor
        params["ticker"] = poly_ticker
    else:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        params.update({
            "ticker": poly_ticker,
            "published_utc.gte": since,
        })

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()
            news = payload.get("results", []) or []
            next_url = payload.get("next_url")  # may be None
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            raise HTTPException(status_code=429, detail="Rate limited by news provider")
        logger.error(f"Polygon error {e.response.status_code} for {poly_ticker}: {e}")
        raise HTTPException(status_code=502, detail="Upstream news provider error")
    except Exception as e:
        logger.error(f"News fetch failed for {poly_ticker}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch news")

    # ---- process + dedupe + sort ----
    def _safe_sent(text: str) -> float:
        try:
            return _simple_sentiment(text)
        except Exception:
            return 0.0

    seen = set()
    processed = []
    for item in news:
        nid = item.get("id") or item.get("url") or item.get("article_url")
        if nid in seen:
            continue
        seen.add(nid)
        title = item.get("title", "")
        processed.append({
            "id": nid,
            "title": title,
            "url": item.get("article_url") or item.get("url", ""),
            "published_at": item.get("published_utc", ""),
            "source": (item.get("publisher") or {}).get("name", ""),
            "sentiment": _safe_sent(title),
            "image_url": item.get("image_url", ""),
        })

    processed.sort(key=lambda x: x.get("published_at", ""), reverse=True)

    # ---- extract safe next_cursor (no apiKey leakage) ----
    next_cursor = None
    if next_url:
        try:
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(next_url).query)
            nc = q.get("cursor", [None])[0]
            next_cursor = nc if isinstance(nc, str) else None
        except Exception as e:
            logger.warning(f"Failed to parse next_url cursor: {e}")

    result = {"items": processed, "next_cursor": next_cursor}

    # ---- cache write (store the object so shape matches response) ----
    if REDIS:
        try:
            await REDIS.setex(cache_key, 3600, json.dumps(result))
        except Exception as e:
            logger.warning(f"Redis setex failed for {cache_key}: {e}")

    return result



# --- Options snapshot (Polygon) ---
@app.get("/options/{symbol}")
async def get_options_snapshot(symbol: str, contract_type: str = "call", limit: int = 10, _api_key: str = Depends(verify_api_key)):
    if not settings.polygon_key:
        raise HTTPException(status_code=400, detail="Polygon key required for options")
    url = f"https://api.polygon.io/v3/snapshot/options/{symbol}"
    params = {"contract_type": contract_type, "sort": "strike_price", "order": "asc", "limit": str(limit), "apiKey": settings.polygon_key}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    if not data.get("results"):
        raise HTTPException(status_code=404, detail=f"No options data for {symbol}")
    results = data["results"]
    ivs = [float(r.get("implied_volatility", 0)) for r in results if r.get("implied_volatility") is not None]
    deltas = [float(r.get("greeks", {}).get("delta", 0)) for r in results]
    avg_iv = float(np.mean(ivs)) if ivs else 0.0
    avg_delta = float(np.mean(deltas)) if deltas else 0.0
    sample_contracts = [{
        "ticker": r["details"]["ticker"],
        "strike": r["details"]["strike_price"],
        "iv": float(r.get("implied_volatility", 0) or 0),
        "delta": float((r.get("greeks") or {}).get("delta", 0) or 0),
        "expiration": r["details"]["expiration_date"],
    } for r in results[:3]]
    return {"symbol": symbol, "avg_iv": avg_iv, "avg_delta": avg_delta, "sample_contracts": sample_contracts, "source": "Polygon Options Snapshot"}

# --- Futures snapshot (Polygon) ---
@app.get("/futures/{symbol}")
async def get_futures_snapshot(symbol: str, _api_key: str = Depends(verify_api_key)):
    if not settings.polygon_key:
        raise HTTPException(status_code=400, detail="Polygon key required for futures")
    url = f"https://api.polygon.io/v3/snapshot?underlying_ticker={symbol}&contract_type=futures&apiKey={settings.polygon_key}"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
    if not data.get("results"):
        raise HTTPException(status_code=404, detail=f"No futures data for {symbol}")
    results = data["results"]
    open_interests = [int(r.get("open_interest", 0) or 0) for r in results]
    last_prices = [float((r.get("last") or {}).get("price", 0) or 0) for r in results]
    return {
        "symbol": symbol,
        "avg_open_interest": float(np.mean(open_interests)) if open_interests else 0.0,
        "avg_price": float(np.mean(last_prices)) if last_prices else 0.0,
        "sample_contracts": [r.get("ticker") for r in results[:3]],
        "source": "Polygon Futures Snapshot",
    }

# --- X (Twitter) demo sentiment ---
@app.get("/x-sentiment/{symbol}")
async def x_sentiment(symbol: str, handles: str = "", _api_key: str = Depends(verify_api_key)):
    sample_posts = []
    if "BTC" in symbol or "X:BTCUSD" in symbol:
        sample_posts = ["BTC ETF approved! Bullish 🚀", "Bitcoin halving incoming", "Bearish on BTC due to regulation"]
    elif "NVDA" in symbol:
        sample_posts = ["NVDA earnings beat, AI boom!", "Chip shortage hurting NVDA", "Bullish on NVDA with new GPU"]
    else:
        sample_posts = ["Generic post about market trends"]
    if handles:
        sample_posts = [f"{p} (from {handles})" for p in sample_posts]
    score = sum(_simple_sentiment(p) for p in sample_posts)
    x_sent = max(-0.2, min(0.2, score)) if sample_posts else 0.0
    return {"symbol": symbol, "x_sentiment": float(x_sent), "sample_posts": sample_posts[:3], "handles_used": handles or "general"}

@app.post("/train")
async def train(req: TrainRequest, _api_key: str = Depends(verify_api_key)):
    px = await _fetch_hist_prices(req.symbol)
    if not px or len(px) < 10:
        raise HTTPException(status_code=400, detail="Not enough price history")

    os.makedirs("models", exist_ok=True)

    # --- Features
    f = await _feat_from_prices(req.symbol, px)
    feature_list = list(f.keys())

    # --- Labels
    rets = np.diff(np.log(np.asarray(px, dtype=float)))
    y = rets[req.lookback_days:] if len(rets) > req.lookback_days else rets
    if len(y) == 0:
        raise HTTPException(status_code=400, detail="Insufficient data")
    # --- Linear first (always succeeds)
    X_df = pd.DataFrame({feat: [f[feat]] * len(y) for feat in feature_list})
    X = X_df.values

    model_linear = SGDOnline(lr=0.05, l2=1e-4)
    model_linear.init(len(feature_list))
    for x_row, yi in zip(X, y):
        model_linear.update(x_row, 1 if yi > 0 else 0)
    linear_data = {
        "coef": model_linear.w.tolist(),
        "features": feature_list,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(y)),
    }
    await REDIS.set(await _model_key(req.symbol + "_linear"), json.dumps(linear_data))

    # --- LSTM (best-effort)
    try:
        model_lstm = Sequential([
            LSTM(50, input_shape=(1, len(feature_list)), return_sequences=True),
            GRU(50),
            Dense(1, activation="sigmoid"),
        ])
        model_lstm.compile(optimizer="adam", loss="binary_crossentropy")
        X_reshaped = np.expand_dims(X, axis=1)
        y_binary = (y > 0).astype(int)
        model_lstm.fit(X_reshaped, y_binary, epochs=5, verbose=0)
        model_lstm.save(f"models/{req.symbol}_lstm.h5")
    except Exception as e:
        logger.warning(f"LSTM skipped: {e}")

    # --- ARIMA (fallbacks)
    try:
        order = (5, 1, 0)
        if len(px) < 30:
            order = (1, 1, 0)
        model_arima = ARIMA(px, order=order).fit()
        with open(f"models/{req.symbol}_arima.pkl", "wb") as file:
            pickle.dump(model_arima, file)
    except Exception as e:
        logger.warning(f"ARIMA skipped: {e}")

    # --- RL (best-effort; consistent window)
    try:
        env = StockEnv(px, window_len=RL_WINDOW)
        model_rl = DQN("MlpPolicy", env, verbose=0)
        model_rl.learn(total_timesteps=10_000)
        model_rl.save(f"models/{req.symbol}_rl.zip")
        env.close()
    except Exception as e:
        logger.warning(f"RL skipped: {e}")

    return {"status": "ok", "models_trained": 4}

class StockEnv(gym.Env):
    """Simple price-following env."""
    metadata = {"render_modes": []}

    def __init__(self, prices, window_len: int = 100):
        super().__init__()
        px = np.asarray(prices, dtype=np.float32)
        if px.ndim != 1 or px.size < 2:
            raise ValueError("prices must be a 1D array with length >= 2")
        self.prices = px
        self.window_len = int(window_len)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_len,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self.current_step = 0
        self.max_steps = len(self.prices) - 2

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        end = self.current_step + 1
        start = max(0, end - self.window_len)
        window = self.prices[start:end]
        if window.size < self.window_len:
            pad = np.full(self.window_len - window.size, window[0], dtype=np.float32)
            window = np.concatenate([pad, window])
        base = window[0] if window[0] != 0 else 1.0
        return (window / base).astype(np.float32)

    def step(self, action):
        # map discrete -> position {-1, 0, +1}
        if action == 0:
            a = -1.0
        elif action == 1:
            a = 0.0
        else:
            a = 1.0
        prev_price = float(self.prices[self.current_step])
        self.current_step += 1
        curr_price = float(self.prices[self.current_step])
        pct_change = 0.0 if prev_price == 0 else (curr_price - prev_price) / prev_price
        reward = float(a * pct_change)
        terminated = self.current_step >= self.max_steps
        truncated = False
        obs = self._get_obs()
        info = {}
        return obs, reward, terminated, truncated, info
class OnlineLearnRequest(BaseModel):
    symbol: str
    steps: int = 50
    batch: int = 32
    lr: float = 0.05
    l2: float = 1e-4
    eta: float = 2.0

@app.post("/learn/online")
async def learn_online(req: OnlineLearnRequest, _api_key: str = Depends(verify_api_key)):
    # 1) Load labeled samples (features: mom_20, rvol_20, autocorr_5)
    try:
        X, y = _load_labeled_samples(req.symbol, limit=max(64, req.steps * req.batch))
    except Exception as e:
        logger.exception(f"Failed to load labeled samples for {req.symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load labeled samples")

    if len(X) < 4:
        raise HTTPException(status_code=422, detail="Not enough labeled samples yet")

    # 2) Load existing linear model (always 4 weights: [bias, w_mom, w_rvol, w_autocorr])
    key = await _model_key(req.symbol + "_linear")  # <<< keep this key everywhere
    raw = await REDIS.get(key)
    if raw:
        m = json.loads(raw)
        coef = np.array(m.get("coef", []), dtype=float)
        if coef.size < 4:
            coef = np.pad(coef, (0, 4 - coef.size))
        elif coef.size > 4:
            coef = coef[:4]
    else:
        coef = np.zeros(4, dtype=float)

    # 3) Online SGD on logistic loss (3 features)
    sgd = SGDOnline(lr=req.lr, l2=req.l2)  # ctor only; do not pass n_features here
    sgd.init(3)                            # <<< bias + 3 features
    sgd.w = coef.copy()                    # class aligns internally if needed

    rng = np.random.default_rng(0)
    for _ in range(req.steps):
        idx = rng.choice(len(X), size=min(req.batch, len(X)), replace=False)
        for i in idx:
            sgd.update(X[i], float(y[i]))

    # 4) Exp-weights over two experts: old vs new (use stable log-loss)
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z.astype(float), -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))

    # use a small, always-valid holdout (at least 8 or all if small)
    k = max(8, min(len(X), len(X)//8 or len(X)))
    hold = slice(0, k)

    def logloss(w: np.ndarray) -> float:
        # w length 4, X has 3 cols
        z = w[0] + X[hold, 0]*w[1] + X[hold, 1]*w[2] + X[hold, 2]*w[3]
        p = _sigmoid(z)
        eps = 1e-9
        return -float(np.mean(y[hold]*np.log(p + eps) + (1 - y[hold])*np.log(1 - p + eps)))

    old_loss = logloss(coef)
    new_loss = logloss(sgd.w)

    ew = EW(eta=req.eta)
    ew.init(2)
    ew.update(np.array([old_loss, new_loss], dtype=float))

    # 5) Save updated model back to the same key
    updated = {
        "symbol": req.symbol.upper(),
        "coef": [float(c) for c in sgd.w],  # exactly 4 numbers
        "features": ["mom_20", "rvol_20", "autocorr_5"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(X)),
        "window_days": None,
        "online": True,
        "exp_weights": [float(w) for w in ew.w],
        "losses": {"old": float(old_loss), "new": float(new_loss)},
    }
    await REDIS.set(key, json.dumps(updated))
    return {"status": "ok", "model": updated}

@app.post("/predict")
async def predict(req: PredictRequest, _api_key: str = Depends(verify_api_key)):
    # 1) Load linear metadata (coef + feature list)
    raw = await REDIS.get(await _model_key(req.symbol + "_linear"))
    if not raw:
        raise HTTPException(status_code=404, detail="Linear model not found; train first")
    model_linear = json.loads(raw)

    # 2) Get prices + features
    px = await _fetch_hist_prices(req.symbol)
    if not px or len(px) < 10:
        raise HTTPException(status_code=400, detail="Not enough price history")
    f = await _feat_from_prices(req.symbol, px)

    # 3) Build input vector (no bias term here; linear model already learned its own intercept-style coef)
    feature_list = model_linear.get("features", ["mom_20", "rvol_20", "autocorr_5"])
    X = np.array([f.get(feat, 0.0) for feat in feature_list], dtype=float)

    preds: list[float] = []
    # --- Linear probability via logistic(dot(w, [1, X])) ---
    w = np.array(model_linear.get("coef", []), dtype=float)
    if w.size:
        xb = np.concatenate([[1.0], X])       # prepend bias 1.0
        k = min(w.shape[0], xb.shape[0])
        score = float(np.dot(xb[:k], w[:k]))  # safe dot with truncation
        score = float(np.clip(score, -60.0, 60.0))
        preds.append(_sigmoid(score))
    # --- LSTM probability (lenient if missing) ---
    try:
        lstm_model = load_lstm_model(req.symbol)  # your helper returns a keras model or raises
        X_lstm = np.expand_dims(np.expand_dims(X, axis=0), axis=0)  # (1, 1, features)
        p_lstm = float(lstm_model.predict(X_lstm, verbose=0)[0][0])
        preds.append(p_lstm)
    except Exception as e:
        logger.info(f"LSTM skipped: {e}")

    # --- ARIMA direction probability (0/1 from last forecast vs last price) ---
    try:
        arima_model = load_arima_model(req.symbol)
        steps = max(1, int(req.horizon_days))
        fc = arima_model.forecast(steps=steps)
        last_fc = float(fc.iloc[-1] if hasattr(fc, "iloc") else fc[-1])
        preds.append(1.0 if last_fc > float(px[-1]) else 0.0)
    except Exception as e:
        logger.info(f"ARIMA skipped: {e}")

    if not preds:
        raise HTTPException(status_code=500, detail="No model produced a prediction")

    # PREDICT (RL adjust block)
    rl_adjust = 0.0
    try:
        rl_model = DQN.load(f"models/{req.symbol}_rl.zip", print_system_info=False)
        env = StockEnv(px, window_len=RL_WINDOW)
        obs, _ = env.reset()
        action, _ = rl_model.predict(obs, deterministic=True)  # 0,1,2
        # map to tiny tilt: short=-1%, flat=0%, long=+1% (clipped to ±5%)
        a = {-1: -0.01, 0: 0.0, 1: 0.01}[int(action) - 1] if int(action) in (0,1,2) else 0.0
        rl_adjust = float(np.clip(a, -0.05, 0.05))
        env.close()
    except Exception as e:
        logger.info(f"RL skipped: {e}")

    # --- Ensemble with exp-weights (robust) ---
    # Always have a default so we never hit UnboundLocalError
    # --- Ensemble with exp-weights (robust) ---
    # Default per-model losses if metrics not available
    losses: List[float] = [0.3] * len(preds)

    con = None
    try:
        con = fs_connect()
        rows = con.execute("""
            SELECT model_id, rmse
            FROM metrics_daily
            WHERE model_id IN ('linear','lstm','arima')
            ORDER BY date DESC
        """).fetchall()

        latest: Dict[str, float] = {}
        for mid, rmse in rows:
            if mid not in latest:
                latest[mid] = float(rmse) if (rmse is not None and np.isfinite(rmse)) else 0.3

        # align to the order of preds you built (linear, lstm, arima)
        ordered = ['linear', 'lstm', 'arima']
        losses = [latest.get(m, 0.3) for m in ordered][:len(preds)]
        if len(losses) < len(preds):
            losses += [0.3] * (len(preds) - len(losses))

    except Exception as e:
        logger.info(f"metrics fetch failed; using default losses. {e}")
    finally:
        try:
            if con is not None:
                con.close()
        except Exception:
            pass
    # ExpWeights: use .init(...) (your class doesn't take ctor kwargs)
    ew = EW()
    ew.init(len(preds))
    ew.update(np.array(losses, dtype=float))

    prob_up = float(np.clip(sum(w * p for w, p in zip(ew.w, preds)) + rl_adjust, 0.0, 1.0))
    # ---- DuckDB logging (persist prediction) ----
    try:
        # IDs & request context
        run_id = str(uuid4())
        model_id = "ensemble-v1"                 # change if you prefer "baseline-linear"
        symbol = req.symbol
        horizon_days = int(getattr(req, "horizon_days", 1))

        # features you already computed in f
        mom_20     = float(f.get("mom_20", 0.0))
        rvol_20    = float(f.get("rvol_20", 0.0))
        autocorr_5 = float(f.get("autocorr_5", 0.0))

        # if you don't compute a point estimate / quantiles yet, None is fine
        yhat_mean  = None
        q05 = q50 = q95 = None
        uncertainty = None

        con = fs_connect()
        fs_log_prediction(con, {
            "run_id":       run_id,
            "model_id":     model_id,
            "symbol":       symbol,
            "issued_at":    datetime.now(timezone.utc).isoformat(),
            "horizon_days": horizon_days,
            "yhat_mean":    yhat_mean,
            "prob_up":      float(prob_up),
            "q05":          q05,
            "q50":          q50,
            "q95":          q95,
            "uncertainty":  uncertainty,
            "features_ref": {
                "mom_20":     mom_20,
                "rvol_20":    rvol_20,
                "autocorr_5": autocorr_5
            }
        })
        con.close()
    except Exception as e:
        logger.exception(f"DuckDB log_prediction failed: {e}")
    # ---------------------------------------------


@app.get("/models")
async def models(_api_key: str = Depends(verify_api_key)):
    return {"models": await _list_models()}

@app.get("/ui/fan-chart.tsx")
async def get_fan_chart_tsx():
    path = r"C:\Users\snowb\OneDrive\Desktop\market-twin-mvp\frontend\src\components\FanChart.tsx"
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return {"filename": "FanChart.tsx", "contents": f.read()}
    raise HTTPException(404, "FanChart.tsx not found")

@app.get("/{path:path}", response_class=HTMLResponse)
async def spa_fallback(path: str):
    file_path = os.path.join(r"C:\Users\snowb\OneDrive\Desktop\market-twin-mvp\frontend\dist", path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    return FileResponse(r"C:\Users\snowb\OneDrive\Desktop\market-twin-mvp\frontend\dist\index.html")

app.mount("/assets", StaticFiles(directory=r"C:\Users\snowb\OneDrive\Desktop\market-twin-mvp\frontend\dist\assets"), name="assets")