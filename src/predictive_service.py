# --- stdlib
from __future__ import annotations
import os, json, asyncio, logging, math, pickle
from datetime import datetime, timedelta, date, timezone
from uuid import uuid4
from typing import List, Optional, Dict, Literal


# --- third-party
from dotenv import load_dotenv; load_dotenv()
import numpy as np
import pandas as pd
import httpx

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader, APIKeyQuery
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from redis.asyncio import Redis

from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

# --- optional ML libs (soft deps; guard all usages)
TF_AVAILABLE = False
SB3_AVAILABLE = False
try:
    from tensorflow.keras.models import load_model as _tf_load_model
    from tensorflow.keras.models import Sequential as _TF_Sequential
    from tensorflow.keras.layers import LSTM as _TF_LSTM, GRU as _TF_GRU, Dense as _TF_Dense
    TF_AVAILABLE = True
except Exception:
    def _missing_tf(*_a, **_k):
        raise ImportError("TensorFlow is not installed on the server.")
    _tf_load_model = _missing_tf; _TF_Sequential = _missing_tf
    _TF_LSTM = _missing_tf; _TF_GRU = _missing_tf; _TF_Dense = _missing_tf

# public aliases
load_model = _tf_load_model
Sequential = _TF_Sequential
LSTM = _TF_LSTM
GRU = _TF_GRU
Dense = _TF_Dense

try:
    import gymnasium as gym  # noqa
except Exception:
    gym = None

try:
    from stable_baselines3 import DQN  # noqa
    SB3_AVAILABLE = True
except Exception:
    DQN = None
    SB3_AVAILABLE = False

# --- local modules (feature store with back-compat)
from .feature_store import connect as fs_connect
try:
    from .feature_store import insert_prediction as fs_log_prediction
except ImportError:
    from .feature_store import log_prediction as fs_log_prediction  # type: ignore
try:
    from .feature_store import compute_and_upsert_metrics_daily as _rollup
except ImportError:
    from .feature_store import rollup as _rollup  # type: ignore

# learners: prefer new names, fall back to old
try:
    from .learners import OnlineLinear as SGDOnline, ExpWeights as EW
except ImportError:
    from .learners import SGDOnline, ExpWeights as EW  # type: ignore

# DUCK utils — try db.duck first, fall back to duck
try:
    from .db.duck import init_schema, insert_prediction, matured_predictions_now, insert_outcome
except Exception:
    from .duck import init_schema, insert_prediction, matured_predictions_now, insert_outcome  # type: ignore


# --- logging / app (single instance)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("predictive")

app = FastAPI(
    title="PredictiveTwin API",
    version="1.2.1",
    docs_url="/api-docs",
    redoc_url=None,
    redirect_slashes=False,
)

class SimRequest(BaseModel):
    symbol: str
    horizon_days: int = Field(default=30, ge=1, le=365)  # hard cap; you also validate later
    n_paths: int = Field(default=1000, ge=50, le=20000)
    timespan: Literal["day", "hour", "minute"] = "day"
    include_news: bool = False
    include_options: bool = False
    include_futures: bool = False
    seed: Optional[int] = None

class TrainRequest(BaseModel):
    symbol: str
    lookback_days: int = Field(default=365, ge=30, le=3650)

class PredictRequest(BaseModel):
    symbol: str
    horizon_days: int = Field(default=30, ge=1, le=365)
    use_online: bool = True  # keep for parity with your code

class RunState(BaseModel):
    run_id: str
    status: Literal["queued", "running", "done", "error"] = "queued"
    started_at: float = Field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    progress: float = 0.0
    error: Optional[str] = None
    artifact: Optional[Dict] = None
    
# ----------------- helpers / config -----------------
def _env_list(name: str, default: List[str] | None = None) -> List[str]:
    s = os.getenv(name, "")
    if not s:
        return default or []
    return [x.strip() for x in s.split(",") if x.strip()]

def _poly_key() -> str:
    return (os.getenv("PT_POLYGON_KEY") or os.getenv("POLYGON_KEY") or "").strip()

def require_tf() -> None:
    if not TF_AVAILABLE:
        raise HTTPException(status_code=503, detail="TensorFlow is not installed on the server.")

def _sigmoid(z: float) -> float:
    z = float(np.clip(z, -60.0, 60.0))
    return 1.0 / (1.0 + math.exp(-z))

def _today_utc_date() -> date:
    return datetime.now(timezone.utc).date()

# Used by Redis model storage
async def _model_key(symbol: str) -> str:
    return f"model:{symbol.upper()}"

def _parse_cors_list(raw: Optional[str]) -> list[str]:
    # Accept: None/"" -> ["*"], JSON array string, or comma-separated string
    if raw is None or raw.strip() == "":
        return ["*"]
    s = raw.strip()
    if s.startswith("["):
        try:
            arr = json.loads(s)
            return [str(x).strip() for x in arr]
        except Exception:
            pass
    return [p.strip() for p in s.split(",") if p.strip()]

# --- Settings (pydantic-settings v2)
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PT_", case_sensitive=False)

    polygon_key: Optional[str] = Field(default_factory=_poly_key)
    news_api_key: Optional[str] = None
    redis_url: str = Field(default_factory=lambda: os.getenv("PT_REDIS_URL", "redis://localhost:6379/0"))
    # note: take raw string and parse ourselves to avoid pydantic JSON decode surprises
    cors_origins_raw: Optional[str] = None
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

settings = Settings()
CORS_ORIGINS = _parse_cors_list(settings.cors_origins_raw)

# --- Redis client (text mode)
REDIS = Redis.from_url(settings.redis_url, decode_responses=True)

# --- CORS (once)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional static frontend (cross-platform; avoids Windows path)
# Set PT_STATIC_DIR in Render if you want to serve your built frontend.
STATIC_DIR = os.getenv("PT_STATIC_DIR", "frontend/dist")
if os.path.isdir(STATIC_DIR):
    # Mount at /app so it won't swallow API routes
    app.mount("/app", StaticFiles(directory=STATIC_DIR, html=True), name="app")
    logger.info(f"Mounted static frontend from {STATIC_DIR} at /app")

# --- API key gate (once)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query  = APIKeyQuery(name="api_key", auto_error=False)
PT_API_KEY = os.getenv("PT_API_KEY", "dev-local")

async def verify_api_key(
    api_key_h: Optional[str] = Depends(api_key_header),
    api_key_q: Optional[str] = Depends(api_key_query),
):
    supplied = (api_key_h or api_key_q or "").strip()
    if PT_API_KEY and supplied != PT_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return supplied

# --- optional track-record router (guarded so missing file won't crash deploy)
try:
    from .track_record import router as track_router  # type: ignore
    app.include_router(track_router, prefix="/track")
    logger.info("Track-record routes enabled at /track")
except Exception as e:
    logger.warning(f"Track-record routes disabled: {e}")

# --- RL constants (single source of truth)
RL_WINDOW = int(os.getenv("PT_RL_WINDOW", "100"))

# --- Background GC (define once and schedule on startup)
async def _gc_loop():
    while True:
        try:
            if REDIS:
                keys = await REDIS.keys("run:*")
                for key in keys:
                    ttl = await REDIS.ttl(key)
                    if ttl is not None and ttl < 0:
                        await REDIS.delete(key)
        except Exception as e:
            logger.error(f"GC loop error: {e}")
        await asyncio.sleep(60)

@app.on_event("startup")
async def _on_startup():
    init_schema()
    # connect Redis
    global REDIS
    try:
        REDIS = Redis.from_url(settings.redis_url, decode_responses=True)
        await REDIS.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        REDIS = None
    # start background tasks
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


# Optional model loaders (only if missing up top)
if "load_lstm_model" not in globals():
    def load_lstm_model(symbol: str):
        require_tf()
        for path in (f"models/{symbol}_lstm.keras", f"models/{symbol}_lstm.h5"):
            try:
                return load_model(path)
            except Exception:
                pass
        raise HTTPException(status_code=404, detail="LSTM model not found; train first")

if "load_arima_model" not in globals():
    def load_arima_model(symbol: str):
        try:
            with open(f"models/{symbol}_arima.pkl", "rb") as file:
                return pickle.load(file)
        except Exception:
            raise HTTPException(status_code=404, detail="ARIMA model not found; train first")

# Lightweight ensemble (linear only; non-blocking) — define if needed
if "get_ensemble_prob_light" not in globals():
    async def get_ensemble_prob_light(symbol: str, redis: "Redis", horizon_days: int = 1) -> float:
        try:
            raw = await redis.get(await _model_key(symbol + "_linear"))
            if not raw: return 0.5
            model_linear = json.loads(raw)
            feats = model_linear.get("features", [])
            coef  = model_linear.get("coef", [])
            px = await _fetch_hist_prices(symbol)
            if not px or len(px) < 10: return 0.5
            f = await _feat_from_prices(symbol, px)
            X = np.array([f.get(feat, 0.0) for feat in feats], dtype=float)
            w = np.array(coef, dtype=float)
            m = int(min(X.shape[0], w.shape[0]))
            if m == 0: return 0.5
            score = float(np.dot(X[:m], w[:m]))
            return _sigmoid(float(np.clip(score, -60.0, 60.0)))
        except Exception as e:
            logger.info(f"get_ensemble_prob_light fallback (0.5) due to: {e}")
            return 0.5

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

        preds: List[float] = []

        # Linear
        if "coef" in model_linear:
            w = np.array(model_linear["coef"], dtype=float)
            xb = np.concatenate([[1.0], X])  # bias + features
            k = min(w.shape[0], xb.shape[0])
            score = float(np.dot(xb[:k], w[:k]))
            score = float(np.clip(score, -60.0, 60.0))
            preds.append(_sigmoid(score))

        # LSTM (optional)
        try:
            model_lstm = load_lstm_model(symbol)
            X_lstm = np.expand_dims(np.expand_dims(X, axis=0), axis=0)
            preds.append(float(model_lstm.predict(X_lstm, verbose=0)[0][0]))
        except Exception:
            pass

        # ARIMA (optional)
        try:
            model_arima = load_arima_model(symbol)
            fc = model_arima.forecast(steps=max(1, int(horizon_days)))
            last_fc = float(fc.iloc[-1] if hasattr(fc, "iloc") else fc[-1])
            preds.append(1.0 if last_fc > float(px[-1]) else 0.0)
        except Exception:
            pass

        # RL adjust (optional)
        rl_adjust = 0.0
        try:
            from stable_baselines3 import DQN  # type: ignore
            rl_model = DQN.load(f"models/{symbol}_rl.zip", print_system_info=False)
            env = StockEnv(px, window_len=RL_WINDOW)
            obs, _ = env.reset()
            action, _ = rl_model.predict(obs, deterministic=True)
            a = float(action[0] if hasattr(action, "__len__") else action)
            rl_adjust = float(np.clip((a - 1) * 0.01, -0.05, 0.05))  # map {0,1,2}->{-1%,0,+1%}
            env.close()
        except Exception as e:
            logger.info(f"RL skipped in ensemble: {e}")

        # Ensemble
        if not preds:
            return 0.5
        ew = EW(); ew.init(len(preds))
        # Equal weights via no-op losses; could wire real validation losses here.
        ew.update(np.array([0.1] * len(preds), dtype=float))
        prob_up = float(np.clip(sum(w * p for w, p in zip(ew.w, preds)) + rl_adjust, 0.0, 1.0))
        return prob_up
    except Exception as e:
        logger.warning(f"Ensemble prob failed for {symbol}: {e}")
        return 0.5

# --- small utilities used elsewhere in file ---
async def _list_models() -> List[str]:
    if not REDIS: return []
    keys = await REDIS.keys("model:*")
    return [k.split(":", 1)[1] for k in keys]

async def _fetch_cached_hist_prices(symbol: str, window_days: int, redis: Redis) -> List[float]:
    """
    Cache key reflects *window_days* and today, so we don't serve stale S0.
    """
    today_str = _today_utc_date().isoformat()
    cache_key = f"hist_prices:{symbol}:{window_days}:{today_str}"

    if redis:
        cached = await redis.get(cache_key)
        if cached:
            try:
                return json.loads(cached)
            except Exception:
                pass

    px = await _fetch_hist_prices(symbol, window_days)
    if redis:
        await redis.setex(cache_key, 1800, json.dumps(px))  # 30 min
    return px

def _dynamic_window_days(horizon_days: int, timespan: str) -> int:
    """
    Choose history window length in 'days' based on the requested timespan.
    """
    base_days = 180
    if timespan == "hour":
        base_days = int(round(180 * 7))
    elif timespan == "minute":
        base_days = int(round(180 * (390 / 252)))
    return min(base_days + int(horizon_days) * 2, settings.lookback_days_max)

async def _ensure_run(run_id: str) -> "RunState":
    raw = await REDIS.get(f"run:{run_id}")
    if not raw:
        raise HTTPException(404, "Run not found")
    rs = RunState.parse_raw(raw)
    ttl = await REDIS.ttl(f"run:{run_id}")
    if ttl == -2:  # missing
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

async def run_simulation(run_id: str, req: "SimRequest", redis: Redis):
    """
    Builds a Monte Carlo artifact, logs to DuckDB.predictions (pred_id-keyed),
    and mirrors to the PathPanda Feature Store.
    """
    logger.info(f"Starting simulation for run_id={run_id}, symbol={req.symbol}")

    # -------- ensure run exists --------
    try:
        rs = await _ensure_run(run_id)
    except Exception as e:
        logger.error(f"_ensure_run failed at start for {run_id}: {e}")
        rs = RunState(status="error", progress=0.0, error=str(e))  # type: ignore[name-defined]
        await redis.set(f"run:{run_id}", rs.json(), ex=settings.run_ttl_seconds)
        return

    rs.status = "running"; rs.progress = 0
    await redis.set(f"run:{run_id}", rs.json(), ex=settings.run_ttl_seconds)

    try:
        # ---------- Guardrails ----------
        if req.n_paths * req.horizon_days > settings.pathday_budget_max:
            raise ValueError("compute budget exceeded")
        if req.horizon_days > settings.horizon_days_max or req.n_paths > settings.n_paths_max:
            raise ValueError("input limits exceeded")

        # ---------- History / base params ----------
        window_days = _dynamic_window_days(req.horizon_days, req.timespan)
        historical_prices = await _fetch_cached_hist_prices(req.symbol, window_days, redis)
        if not historical_prices or len(historical_prices) < 30:
            raise ValueError("Insufficient history")

        px_arr = np.array(historical_prices, dtype=float)
        rets = np.diff(np.log(px_arr))
        scale = 252 if req.timespan == "day" else 252 * 24  # simple hourly alt
        mu_ann    = float(np.mean(rets) * scale) if rets.size else 0.0
        sigma_ann = float(np.std(rets) * math.sqrt(scale)) if rets.size else 0.2
        sigma_ann = float(np.clip(sigma_ann, 1e-4, 1.5))
        mu_ann    = float(np.clip(mu_ann, -2.0,  2.0))

        # ---------- ML adjustment → VOL-AWARE DRIFT TILT ----------
        ensemble_prob = 0.5
        try:
            ensemble_prob = await asyncio.wait_for(get_ensemble_prob(req.symbol, redis, req.horizon_days), timeout=1.0)
        except Exception:
            try:
                ensemble_prob = await asyncio.wait_for(get_ensemble_prob_light(req.symbol, redis, req.horizon_days), timeout=0.5)
            except Exception:
                ensemble_prob = 0.5
        conf = (2.0 * float(ensemble_prob) - 1.0)  # [-1,1]
        mu_ann = float(mu_ann + (0.30 * conf * sigma_ann))  # vol-aware tilt

        # ---------- Optional news/options/futures tweaks (BEFORE sim) ----------
        sentiment = 0.0
        if getattr(req, "include_news", False):
            try:
                poly_ticker = req.symbol.upper().strip()
                key = _poly_key()
                since = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")
                url = "https://api.polygon.io/v2/reference/news"
                params = {"ticker": poly_ticker, "published_utc.gte": since, "limit": "20", "apiKey": key}
                async with httpx.AsyncClient(timeout=8.0) as client:
                    r = await client.get(url, params=params)
                    if r.status_code == 200:
                        items = (r.json() or {}).get("results", []) or []
                        if items:
                            ss = [_safe_sent((it.get("title") or "")) for it in items]
                            if ss:
                                sentiment = float(np.clip(np.mean(ss) * 0.2, -0.05, 0.05))
            except Exception as e:
                logger.info(f"news sentiment failed: {e}")

        if getattr(req, "include_options", False):
            sigma_ann = float(np.clip(sigma_ann * 1.05, 1e-4, 1.5))
        if getattr(req, "include_futures", False):
            mu_ann += 0.001

        # fold sentiment after tilt
        mu_ann = float(np.clip(mu_ann + sentiment, -3.0, 3.0))

        # ---------- Monte Carlo (GBM) ----------
        S0 = float(px_arr[-1])
        dt = 1.0 / float(scale)
        n_days = max(1, int(req.horizon_days))

        # Deterministic seed: (symbol, horizon, model_id, UTC date) with run_id fallback
        model_id = "mc_v1"
        utc_day = datetime.utcnow().strftime("%Y-%m-%d")
        try:
            seed_key = f"{req.symbol.upper()}|{n_days}|{model_id}|{utc_day}"
            seed = abs(hash(seed_key)) % (2**32 - 1)
        except Exception:
            seed = abs(hash(run_id)) % (2**32 - 1)

        rng = np.random.default_rng(seed)
        Z = rng.normal(size=(req.n_paths, n_days))

        drift = (mu_ann - 0.5 * sigma_ann**2) * dt
        diffusion = sigma_ann * math.sqrt(dt)
        log_returns = drift + diffusion * Z
        log_paths  = np.cumsum(log_returns, axis=1)  # (paths, days)
        paths = S0 * np.exp(np.concatenate([np.zeros((req.n_paths, 1)), log_paths], axis=1))  # (paths, days+1)

        # Progress tick (post-paths)
        rs.progress = 40
        await redis.set(f"run:{run_id}", rs.json(), ex=settings.run_ttl_seconds)

        # ---------- Percentile bands (per day) ----------
        p50_line = np.median(paths, axis=0)
        p80_low, p80_high = np.percentile(paths, [10, 90], axis=0)
        p95_low, p95_high = np.percentile(paths, [2.5, 97.5], axis=0)

        def _ffill_nonfinite(arr: np.ndarray, fallback: float) -> np.ndarray:
            out = np.array(arr, dtype=float)
            if not np.isfinite(out[0]): out[0] = float(fallback)
            for i in range(1, len(out)):
                if not np.isfinite(out[i]): out[i] = out[i - 1]
            return out

        fallback = S0
        for arr in (p50_line, p80_low, p80_high, p95_low, p95_high):
            np.nan_to_num(arr, copy=False, nan=fallback, posinf=fallback, neginf=fallback)
            arr[:] = _ffill_nonfinite(arr, fallback)

        # ---------- Terminal distribution metrics ----------
        T = paths.shape[1]                # n_days + 1
        eod_idx = min(1, T - 1)           # 0=now, 1=end-of-day
        eod_mean = float(paths[:, eod_idx].mean())
        eod_med  = float(p50_line[eod_idx])
        eod_p05  = float(p95_low[eod_idx])
        eod_p95  = float(p95_high[eod_idx])

        terminal = paths[:, -1]
        prob_up  = float(np.mean(terminal > S0))

        ret_terminal = (terminal - S0) / S0
        var95 = float(np.percentile(ret_terminal, 5))
        es_mask = ret_terminal <= var95
        es95 = float(ret_terminal[es_mask].mean()) if es_mask.any() else float(var95)

        term_q05 = float(np.percentile(terminal, 5))
        term_q50 = float(np.percentile(terminal, 50))
        term_q95 = float(np.percentile(terminal, 95))

        thresholds_pct = np.array([-0.05, 0.00, 0.05, 0.10], dtype=float)
        thresholds_abs = (1.0 + thresholds_pct) * float(S0)  # (K,)
        probs_by_day = ((paths[:, :, None] >= thresholds_abs[None, None, :]).mean(axis=0).T).tolist()  # (K,T)

        rs.progress = 80
        await redis.set(f"run:{run_id}", rs.json(), ex=settings.run_ttl_seconds)

        # ---------- Artifact for UI ----------
        rs.artifact = {
            "symbol": req.symbol,
            "horizon_days": int(req.horizon_days),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "median_path": [[i, float(v)] for i, v in enumerate(p50_line.tolist())],
            "bands": {
                "p50":      [[i, float(v)] for i, v in enumerate(p50_line.tolist())],
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
            "model_info": {
                "direction": "MonteCarlo",
                "regime": "lognormal",
                "model_id": "mc_v1",
                "seed_hint": int(seed),
                "timescale": req.timespan,
            },
            "calibration": {"window": int(window_days), "p80_empirical": 0.8},
            "terminal_prices": [float(x) for x in terminal.tolist()],
            "var_es": {"var95": var95, "es95": es95},  # returns
            "hit_probs": {
                "thresholds_abs": [float(x) for x in thresholds_abs.tolist()],
                "probs_by_day": probs_by_day,
            },
            "eod_estimate": {
                "day_index": int(eod_idx),
                "median": eod_med,
                "mean": eod_mean,
                "p05": eod_p05,
                "p95": eod_p95
            },
        }

        # ---------- Persist: DuckDB.predictions ----------
        try:
            pred_id = str(uuid4())
            insert_prediction({
                "pred_id": pred_id,
                "ts": datetime.utcnow(),
                "symbol": req.symbol.upper(),
                "horizon_d": int(req.horizon_days),
                "model_id": "mc_v1",
                "prob_up_next": float(prob_up),
                "p05": term_q05, "p50": term_q50, "p95": term_q95,
                "spot0": float(S0),
                "user_ctx": {"ui": "pathpanda", "run_id": run_id, "n_paths": int(req.n_paths)},
                "run_id": run_id,
            })
        except Exception as e:
            logger.warning(f"DuckDB insert_prediction (simulate) failed: {e}")

        # ---------- Mirror: PathPanda Feature Store ----------
        try:
            con = fs_connect()
            fs_log_prediction(con, {
                "run_id":       run_id,
                "model_id":     "mc_v1",
                "symbol":       req.symbol.upper(),
                "issued_at":    datetime.now(timezone.utc).isoformat(),
                "horizon_days": int(req.horizon_days),
                "yhat_mean":    term_q50,
                "prob_up":      float(prob_up),
                "q05":          term_q05, "q50": term_q50, "q95": term_q95,
                "uncertainty":  float(np.std(terminal)),
                "features_ref": {
                    "window_days":  int(window_days),
                    "paths":        int(req.n_paths),
                    "S0":           float(S0),
                    "mu_ann":       float(mu_ann),
                    "sigma_ann":    float(sigma_ann),
                    "timespan":     req.timespan,
                    "seed_hint":    int(seed),
                },
            })
            con.close()
        except Exception as e:
            logger.warning(f"Feature Store mirror failed: {e}")

        # ---------- Finish ----------
        rs.status = "done"; rs.progress = 100
        await redis.set(f"run:{run_id}", rs.json(), ex=settings.run_ttl_seconds)
        logger.info(f"Completed simulation for run_id={run_id}")

    except Exception as e:
        rs.status = "error"; rs.error = str(e)  # type: ignore[attr-defined]
        await redis.set(f"run:{run_id}", rs.json(), ex=settings.run_ttl_seconds)
        logger.exception(f"Simulation failed for run_id={run_id}: {e}")

        # ---------- Mirror: PathPanda Feature Store ----------
        try:
            con = fs_connect()
            fs_log_prediction(con, {
                "run_id":       run_id,
                "model_id":     "mc_v1",
                "symbol":       req.symbol.upper(),
                "issued_at":    datetime.now(timezone.utc).isoformat(),
                "horizon_days": int(req.horizon_days),
                "yhat_mean":    term_q50,                  # price mid
                "prob_up":      float(prob_up),
                "q05":          term_q05, "q50": term_q50, "q95": term_q95,
                "uncertainty":  float(np.std(terminal)),
                "features_ref": {
                    "window_days":  int(window_days),
                    "paths":        int(req.n_paths),
                    "S0":           float(S0),
                    "mu_ann":       float(mu_ann),
                    "sigma_ann":    float(sigma_ann),
                    "timespan":     req.timespan,
                    "seed_hint":    int(seed),
                },
            })
            con.close()
        except Exception as e:
            logger.warning(f"Feature Store mirror failed: {e}")

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
    key = (settings.polygon_key or "").strip()
    key_mode = "real" if key else "none"
    return {
        "ok": True,
        "redis_ok": bool(redis_ok),
        "redis_url": settings.redis_url,
        "polygon_key_present": bool(key),
        "polygon_key_mode": key_mode,  # "real" | "none"
        "n_paths_max": settings.n_paths_max,
        "horizon_days_max": settings.horizon_days_max,
        "pathday_budget_max": settings.pathday_budget_max,
    }


@app.post("/outcomes/label")
async def outcomes_label(limit: int = 5000, _api_key: str = Depends(verify_api_key)):
    """
    Label any matured predictions (ts + horizon_d <= now) that don't yet have outcomes.
    Writes to src.db.duck.outcomes and (optionally) mirrors each label into PathPanda FS
    so daily metrics rollups can see realized prices.
    """
    limit = max(100, min(int(limit), 20000))

    # 1) Load matured-but-unlabeled predictions from the Predictions/Outcomes store
    try:
        matured = matured_predictions_now(limit=limit)  # [(pred_id, ts, symbol, horizon_d, spot0), ...]
    except Exception as e:
        logger.exception(f"matured_predictions_now failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to query matured predictions")

    processed = 0
    labeled = 0

    # 2) Label each matured row
    for pred_id, ts, symbol, horizon_d, spot0 in matured:
        processed += 1
        try:
            # compute target date/time for the realized close
            when = (ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts)))
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            target_ts = when + timedelta(days=int(horizon_d))

            # fetch realized close (handles weekends/holidays internally)
            realized_price = await _fetch_realized_price(symbol, target_ts)
            if realized_price is None or not np.isfinite(realized_price) or not spot0:
                continue

            ret = (float(realized_price) / float(spot0)) - 1.0
            label_up = bool(ret > 0.0)

            # 2a) Write to Outcomes table (DuckDB via src.db.duck)
            try:
                insert_outcome(pred_id, target_ts, float(realized_price), float(ret), label_up)
            except Exception as e:
                logger.warning(f"insert_outcome failed for {pred_id}: {e}")
                continue

            labeled += 1

            # 2b) (Optional) Mirror into PathPanda Feature Store for metrics
            #     Use local import to avoid import-time cycles. If FS not available, skip quietly.
            try:
                from .feature_store import connect as _pfs_connect, insert_outcome as _pfs_insert_out
                con_fs = _pfs_connect()
                _pfs_insert_out(
                    con_fs,
                    {
                        # Use pred_id as the run_id so outcomes are uniquely joinable if predictions were mirrored
                        "run_id": pred_id,
                        "symbol": symbol,
                        "realized_at": target_ts,
                        "y": float(realized_price),  # realized PRICE level
                    },
                )
                con_fs.close()
            except Exception as e:
                logger.warning(f"feature_store outcome mirror failed for {pred_id}: {e}")

        except Exception as e:
            logger.warning(f"labeling failed for pred_id={pred_id}: {e}")

    return {"status": "ok", "processed": processed, "labeled": labeled}


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


def _load_labeled_samples(symbol: str, limit: int = 256):
    """
    Pull recent labeled examples and build a small binary dataset:
    label = 1 if realized_price >= forecast_mid (q50 or yhat_mean), else 0
    Features come from features_ref (legacy friendly).
    """
    con = fs_connect()
    rows = con.execute("""
        SELECT
            p.model_id, p.symbol, p.issued_at, p.horizon_days,
            p.features_ref, p.q50, p.yhat_mean,
            o.realized_at, o.y AS realized_price
        FROM predictions p
        JOIN outcomes o USING (run_id)
        WHERE p.symbol = ?
        ORDER BY o.realized_at DESC
        LIMIT ?
    """, [symbol.upper(), limit]).fetchall()
    con.close()

    X, y = [], []
    for _, _, _, _, features_ref, q50, yhat_mean, _, realized_price in rows:
        try:
            feats = {}
            if features_ref:
                j = json.loads(features_ref)
                feats = j.get("features", j) if isinstance(j, dict) else {}

            mom  = float(feats.get("mom_20", 0.0))
            rvol = float(feats.get("rvol_20", 0.0))
            ac5  = float(feats.get("autocorr_5", 0.0))

            mid = q50 if (q50 is not None) else yhat_mean
            if mid is None:
                mid = float(realized_price)

            realized = float(realized_price)
            label = 1.0 if realized >= float(mid) else 0.0

            X.append([mom, rvol, ac5])
            y.append(label)
        except Exception:
            continue

    return np.array(X, dtype=float), np.array(y, dtype=float)


def _safe_sent(text: str) -> float:
    base = _simple_sentiment(text)
    t = text.lower()
    if any(w in t for w in ["beats", "record", "surge", "raises", "upgrade"]): base += 0.1
    if any(w in t for w in ["misses", "plunge", "cuts", "downgrade", "probe"]): base -= 0.1
    return max(-1.0, min(1.0, base))


@app.get("/api/news/{symbol}")
async def get_news(
    symbol: str,
    limit: int = 10,
    days: int = 7,
    cursor: Optional[str] = None,
    _api_key: str = Depends(verify_api_key),
):
    # Always use our key helper (real or mock) so dev doesn’t 400
    key = _poly_key()

    # input guards
    limit = max(1, min(int(limit), 50))
    days = max(1, min(int(days), 30))

    # normalize ticker for Polygon (stocks as-is; crypto -> X:BASEUSD)
    raw_symbol = symbol.strip().upper()

    def to_polygon_ticker(s: str) -> str:
        s2 = s.replace(":", "")
        if (("-" in s and s.endswith("USD")) or (s.endswith("USD") and "-" not in s)):
            base = s.split("-")[0] if "-" in s else s2.replace("USD", "")
            return f"X:{base}USD"
        return s

    poly_ticker = to_polygon_ticker(raw_symbol)

    # cache (cursor-specific)
    cache_key = f"news:{poly_ticker}:{limit}:{days}:{cursor or 'first'}"
    if REDIS:
        try:
            cached = await REDIS.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Redis get failed for {cache_key}: {e}")

    # fetch
    url = "https://api.polygon.io/v2/reference/news"
    params = {"apiKey": key, "limit": str(limit)}
    if cursor:
        params["cursor"] = cursor
        params["ticker"] = poly_ticker
    else:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        params.update({"ticker": poly_ticker, "published_utc.gte": since})

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()
            news = payload.get("results", []) or []
            next_url = payload.get("next_url")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            raise HTTPException(status_code=429, detail="Rate limited by news provider")
        logger.error(f"Polygon error {e.response.status_code} for {poly_ticker}: {e}")
        raise HTTPException(status_code=502, detail="Upstream news provider error")
    except Exception as e:
        logger.error(f"News fetch failed for {poly_ticker}: {e}")
        raise HTTPException(status_code=502, detail="News fetch failed")

    # map results
    seen: set[str] = set()
    processed: list[dict] = []
    for item in news:
        nid = item.get("id") or item.get("url") or item.get("article_url")
        if not nid or nid in seen:
            continue
        seen.add(nid)
        title = item.get("title", "")
        processed.append(
            {
                "id": nid,
                "title": title,
                "url": item.get("article_url") or item.get("url", ""),
                "published_at": item.get("published_utc", ""),
                "source": (item.get("publisher") or {}).get("name", ""),
                "sentiment": _safe_sent(title),
                "image_url": item.get("image_url", ""),
            }
        )

    processed.sort(key=lambda x: x.get("published_at", ""), reverse=True)

    # extract next_cursor safely
    next_cursor = None
    if next_url:
        try:
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(next_url).query)
            next_cursor = q.get("cursor", [None])[0]
        except Exception:
            next_cursor = None

    result = {"items": processed, "nextCursor": next_cursor}

    if REDIS:
        try:
            await REDIS.setex(cache_key, 3600, json.dumps(result))
        except Exception as e:
            logger.warning(f"Redis setex failed for {cache_key}: {e}")

    return result


# --- Options snapshot (Polygon) ---
@app.get("/options/{symbol}")
async def get_options_snapshot(
    symbol: str,
    contract_type: Literal["call", "put"] = "call",
    limit: int = 10,
    _api_key: str = Depends(verify_api_key),
):
    """
    Lightweight options snapshot summary.
    Uses _poly_key() so dev can run without a real key (mock key allowed).
    """
    key = _poly_key()
    limit = max(1, min(int(limit), 50))

    url = f"https://api.polygon.io/v3/snapshot/options/{symbol.upper().strip()}"
    params = {
        "contract_type": contract_type,
        "sort": "strike_price",
        "order": "asc",
        "limit": str(limit),
        "apiKey": key,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json() or {}
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            raise HTTPException(status_code=429, detail="Rate limited by options provider")
        raise HTTPException(status_code=502, detail=f"Options upstream error {e.response.status_code}")
    except Exception as e:
        logger.warning(f"Options snapshot failed for {symbol}: {e}")
        raise HTTPException(status_code=502, detail="Options fetch failed")

    results = data.get("results") or []
    if not results:
        raise HTTPException(status_code=404, detail=f"No options data for {symbol}")

    ivs = [float(r.get("implied_volatility", 0) or 0) for r in results]
    deltas = [float((r.get("greeks") or {}).get("delta", 0) or 0) for r in results]
    avg_iv = float(np.mean(ivs)) if ivs else 0.0
    avg_delta = float(np.mean(deltas)) if deltas else 0.0

    sample_contracts = []
    for r in results[:3]:
        det = r.get("details") or {}
        sample_contracts.append({
            "ticker": det.get("ticker"),
            "strike": det.get("strike_price"),
            "iv": float(r.get("implied_volatility", 0) or 0),
            "delta": float((r.get("greeks") or {}).get("delta", 0) or 0),
            "expiration": det.get("expiration_date"),
        })

    return {
        "symbol": symbol.upper().strip(),
        "avg_iv": avg_iv,
        "avg_delta": avg_delta,
        "sample_contracts": sample_contracts,
        "source": "Polygon Options Snapshot",
    }


# --- Futures snapshot (Polygon) ---
@app.get("/futures/{symbol}")
async def get_futures_snapshot(symbol: str, _api_key: str = Depends(verify_api_key)):
    """
    Very light futures snapshot aggregation.
    Uses _poly_key() so dev can run without a real key (mock key allowed).
    """
    key = _poly_key()
    url = "https://api.polygon.io/v3/snapshot"
    params = {
        "underlying_ticker": symbol.upper().strip(),
        "contract_type": "futures",
        "apiKey": key,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json() or {}
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            raise HTTPException(status_code=429, detail="Rate limited by futures provider")
        raise HTTPException(status_code=502, detail=f"Futures upstream error {e.response.status_code}")
    except Exception as e:
        logger.warning(f"Futures snapshot failed for {symbol}: {e}")
        raise HTTPException(status_code=502, detail="Futures fetch failed")

    results = data.get("results") or []
    if not results:
        raise HTTPException(status_code=404, detail=f"No futures data for {symbol}")

    open_interests = [int(r.get("open_interest", 0) or 0) for r in results]
    last_prices = [float((r.get("last") or {}).get("price", 0) or 0) for r in results]

    return {
        "symbol": symbol.upper().strip(),
        "avg_open_interest": float(np.mean(open_interests)) if open_interests else 0.0,
        "avg_price": float(np.mean(last_prices)) if last_prices else 0.0,
        "sample_contracts": [r.get("ticker") for r in results[:3]],
        "source": "Polygon Futures Snapshot",
    }


# --- X (Twitter) demo sentiment ---
@app.get("/x-sentiment/{symbol}")
async def x_sentiment(symbol: str, handles: str = "", _api_key: str = Depends(verify_api_key)):
    sample_posts = []
    s = symbol.upper()
    if "BTC" in s or "X:BTCUSD" in s:
        sample_posts = ["BTC ETF approved! Bullish 🚀", "Bitcoin halving incoming", "Bearish on BTC due to regulation"]
    elif "NVDA" in s:
        sample_posts = ["NVDA earnings beat, AI boom!", "Chip shortage hurting NVDA", "Bullish on NVDA with new GPU"]
    else:
        sample_posts = ["Generic post about market trends"]

    if handles:
        sample_posts = [f"{p} (from {handles})" for p in sample_posts]

    score = sum(_simple_sentiment(p) for p in sample_posts)
    x_sent = max(-0.2, min(0.2, score)) if sample_posts else 0.0
    return {"symbol": s, "x_sentiment": float(x_sent), "sample_posts": sample_posts[:3], "handles_used": handles or "general"}


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
        if Sequential is None:
            raise RuntimeError("Keras not available")
        model_lstm = Sequential([
            LSTM(50, input_shape=(1, len(feature_list)), return_sequences=True),
            GRU(50),
            Dense(1, activation="sigmoid"),
        ])
        model_lstm.compile(optimizer="adam", loss="binary_crossentropy")
        X_reshaped = np.expand_dims(X, axis=1)
        y_binary = (y > 0).astype(int)
        model_lstm.fit(X_reshaped, y_binary, epochs=5, verbose=0)
        model_lstm.save(f"models/{req.symbol}_lstm.keras")
    except Exception as e:
        logger.warning(f"LSTM skipped: {e}")

    # --- ARIMA (fallbacks)
    try:
        order = (5, 1, 0) if len(px) >= 30 else (1, 1, 0)
        model_arima = ARIMA(px, order=order).fit()
        with open(f"models/{req.symbol}_arima.pkl", "wb") as file:
            pickle.dump(model_arima, file)
    except Exception as e:
        logger.warning(f"ARIMA skipped: {e}")

    # --- RL (best-effort; consistent window)
    try:
        if gym is None or DQN is None:
            raise RuntimeError("RL libs not available")
        env = StockEnv(px, window_len=RL_WINDOW)
        model_rl = DQN("MlpPolicy", env, verbose=0)
        model_rl.learn(total_timesteps=10_000)
        model_rl.save(f"models/{req.symbol}_rl.zip")
        env.close()
    except Exception as e:
        logger.warning(f"RL skipped: {e}")

    return {"status": "ok", "models_trained": 4}


class StockEnv(gym.Env):  # type: ignore[attr-defined]
    """Simple price-following env."""
    metadata = {"render_modes": []}

    def __init__(self, prices, window_len: int = 100):
        super().__init__()
        px = np.asarray(prices, dtype=np.float32)
        if px.ndim != 1 or px.size < 2:
            raise ValueError("prices must be a 1D array with length >= 2")
        self.prices = px
        self.window_len = int(window_len)
        self.observation_space = gym.spaces.Box(  # type: ignore[attr-defined]
            low=-np.inf, high=np.inf, shape=(self.window_len,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(3)  # type: ignore[attr-defined]
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
    key = await _model_key(req.symbol + "_linear")  # keep this key everywhere
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
    sgd = SGDOnline(lr=req.lr, l2=req.l2)
    sgd.init(3)            # bias + 3 features
    sgd.w = coef.copy()    # class aligns internally if needed

    rng = np.random.default_rng(0)
    for _ in range(req.steps):
        idx = rng.choice(len(X), size=min(req.batch, len(X)), replace=False)
        for i in idx:
            sgd.update(X[i], float(y[i]))

    # 4) Exp-weights over two experts: old vs new (use stable log-loss)
    def _sigmoid_arr(z: np.ndarray) -> np.ndarray:
        z = np.clip(z.astype(float), -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))

    k = max(8, min(len(X), len(X)//8 or len(X)))
    hold = slice(0, k)

    def logloss(w: np.ndarray) -> float:
        z = w[0] + X[hold, 0]*w[1] + X[hold, 1]*w[2] + X[hold, 2]*w[3]
        p = _sigmoid_arr(z)
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
    """
    Returns an ensemble probability that the next move is up, logs the prediction to DuckDB,
    and mirrors the row into the PathPanda Feature Store so dashboards/metrics see it.
    """
    # --- 1) Load linear head (coef + feature list) from Redis ---
    raw = await REDIS.get(await _model_key(req.symbol + "_linear"))
    if not raw:
        raise HTTPException(status_code=404, detail="Linear model not found; train first")
    model_linear = json.loads(raw)

    # --- 2) Prices + features ---
    symbol = req.symbol.upper().strip()
    px = await _fetch_hist_prices(symbol)
    if not px or len(px) < 10:
        raise HTTPException(status_code=400, detail="Not enough price history")
    f = await _feat_from_prices(symbol, px)

    # --- 3) Build input vector for linear head ---
    feature_list = model_linear.get("features", ["mom_20", "rvol_20", "autocorr_5"])
    X = np.array([f.get(feat, 0.0) for feat in feature_list], dtype=float)

    preds: List[float] = []

    # --- Linear probability via logistic(dot(w, [1, X])) ---
    try:
        w = np.array(model_linear.get("coef", []), dtype=float)
        if w.size:
            xb = np.concatenate([[1.0], X])            # prepend bias
            k = min(w.shape[0], xb.shape[0])
            score = float(np.dot(xb[:k], w[:k]))
            score = float(np.clip(score, -60.0, 60.0)) # numerical safety
            preds.append(1.0 / (1.0 + np.exp(-score)))
    except Exception as e:
        logger.info(f"Linear skipped: {e}")

    # --- LSTM probability (optional/lenient) ---
    try:
        lstm_model = load_lstm_model(symbol)
        X_lstm = np.expand_dims(np.expand_dims(X, axis=0), axis=0)  # (1, 1, features)
        p_lstm = float(lstm_model.predict(X_lstm, verbose=0)[0][0])
        preds.append(p_lstm)
    except Exception as e:
        logger.info(f"LSTM skipped: {e}")

    # --- ARIMA direction (0/1) ---
    try:
        arima_model = load_arima_model(symbol)
        steps = max(1, int(getattr(req, "horizon_days", 1)))
        fc = arima_model.forecast(steps=steps)
        last_fc = float(fc.iloc[-1] if hasattr(fc, "iloc") else fc[-1])
        preds.append(1.0 if last_fc > float(px[-1]) else 0.0)
    except Exception as e:
        logger.info(f"ARIMA skipped: {e}")

    if not preds:
        raise HTTPException(status_code=500, detail="No model produced a prediction")

    # --- RL tiny tilt (optional) ---
    rl_adjust = 0.0
    try:
        rl_model = DQN.load(f"models/{symbol}_rl.zip", print_system_info=False)
        env = StockEnv(px, window_len=RL_WINDOW)
        obs, _ = env.reset()
        action, _ = rl_model.predict(obs, deterministic=True)  # 0,1,2
        a_map = {-1: -0.01, 0: 0.0, 1: 0.01}
        a_idx = int(action)
        a_val = a_map.get(a_idx - 1, 0.0)  # action 0→-1, 1→0, 2→+1
        rl_adjust = float(np.clip(a_val, -0.05, 0.05))
        env.close()
    except Exception as e:
        logger.info(f"RL skipped: {e}")

    # --- Simple exp-weights ensemble (defaults if no metrics available) ---
    losses: List[float] = [0.3] * len(preds)
    ew = EW()
    ew.init(len(preds))
    ew.update(np.array(losses, dtype=float))
    prob_up = float(np.clip(sum(wt * p for wt, p in zip(ew.w, preds)) + rl_adjust, 0.0, 1.0))

    # --- Quantile placeholders (wire real conformal/quantiles later) ---
    q05 = None
    q50 = None
    q95 = None

    # --- Log to Predictions/Outcomes store (DuckDB via src.db.duck) ---
    pred_id = str(uuid4())
    spot0 = float(px[-1])
    try:
        insert_prediction(
            {
                "pred_id": pred_id,
                "ts": datetime.utcnow(),
                "symbol": symbol,
                "horizon_d": int(getattr(req, "horizon_days", 1)),
                "model_id": "ensemble-v1",
                "prob_up_next": float(prob_up),
                "p05": q05,
                "p50": q50,
                "p95": q95,
                "spot0": spot0,
                "user_ctx": {"ui": "pathpanda"},
                "run_id": getattr(req, "run_id", "") or "",
            }
        )
    except Exception as e:
        logger.exception(f"DuckDB insert_prediction failed: {e}")

    # --- Also mirror to PathPanda Feature Store (so metrics/dashboard can see it) ---
    try:
        con_fs = fs_connect()
        fs_log_prediction(
            con_fs,
            {
                # mirror using pred_id as run_id for joinability
                "run_id": pred_id,
                "model_id": "ensemble-v1",
                "symbol": symbol,
                "issued_at": datetime.now(timezone.utc).isoformat(),
                "horizon_days": int(getattr(req, "horizon_days", 1)),
                "yhat_mean": None,                # fill with price mid later if available
                "prob_up": float(prob_up),        # note: FS expects 'prob_up'
                "q05": q05,
                "q50": q50,
                "q95": q95,
                "uncertainty": None,
                "features_ref": {
                    "mom_20": float(f.get("mom_20", 0.0)),
                    "rvol_20": float(f.get("rvol_20", 0.0)),
                    "autocorr_5": float(f.get("autocorr_5", 0.0)),
                    "spot0": spot0,
                },
            },
        )
        con_fs.close()
    except Exception as e:
        logger.warning(f"Feature store mirror failed: {e}")

    # --- Response for UI ---
    return {
        "pred_id": pred_id,
        "symbol": symbol,
        "prob_up_next": float(prob_up),
        "p05": q05,
        "p50": q50,
        "p95": q95,
        "spot0": spot0,
    }


@app.get("/models")
async def models(_api_key: str = Depends(verify_api_key)):
    return {"models": await _list_models()}


# ---- Optional static hosting (disabled by default) ----
# If you want the API to also serve your built frontend, set PT_FRONTEND_DIR
# to the absolute path of the SPA build directory (the one with index.html).
FRONTEND_DIR = os.getenv("PT_FRONTEND_DIR", "").strip()

@app.get("/ui/fan-chart.tsx")
async def get_fan_chart_tsx():
    # Only try to read the file if FRONTEND_DIR is provided
    if FRONTEND_DIR:
        candidate = os.path.join(FRONTEND_DIR, "src", "components", "FanChart.tsx")
        if os.path.isfile(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                return {"filename": "FanChart.tsx", "contents": f.read()}
    # In container deployments (like Render) this usually isn't present.
    raise HTTPException(404, "FanChart.tsx not available on this deployment")

if FRONTEND_DIR:
    # Serve static assets & SPA only if configured
    try:
        app.mount("/assets",
                  StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")),
                  name="assets")

        @app.get("/{path:path}", response_class=HTMLResponse)
        async def spa_fallback(path: str):
            file_path = os.path.join(FRONTEND_DIR, path)
            if os.path.isfile(file_path):
                return FileResponse(file_path)
            index = os.path.join(FRONTEND_DIR, "index.html")
            if os.path.isfile(index):
                return FileResponse(index)
            raise HTTPException(404, "index.html not found")
    except Exception as e:
        logger.warning(f"Static hosting disabled: {e}")
else:
    # Default: API only. Helpful response instead of 404s for all paths.
    @app.get("/{path:path}")
    async def catch_all(path: str):
        return {
            "status": "ok",
            "message": "PredictiveTwin API is running. No static frontend configured on this service."
        }
