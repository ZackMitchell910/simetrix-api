# src/predictive_api.py
# =============================================================================
# Predictive Service (Render-friendly startup)
# - Heavy/optional deps are loaded in @app.on_event("startup")
# - Immediate bind + health check so Render's port scan succeeds
# =============================================================================

# --- stdlib
from __future__ import annotations
import os, json, asyncio, logging, math, pickle, time, random
from datetime import datetime, timedelta, date, timezone
from uuid import uuid4
from typing import List, Optional, Any, Callable, Dict, Literal, Sequence, Tuple
from contextlib import asynccontextmanager
from pathlib import Path

# Quiet TensorFlow logs if/when it gets imported
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# --- third-party
from dotenv import load_dotenv; load_dotenv()
import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, Depends, HTTPException, WebSocket, Query, Header, Body, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from redis.asyncio import Redis
from secrets import token_urlsafe

# -----------------------------------------------------------------------------
# Logging FIRST so all later imports can log cleanly
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# FastAPI app (bind immediately; health is cheap)
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Simetrix API",
    version="1.3.0",
    docs_url="/api-docs",
    redoc_url=None,
    redirect_slashes=False,
)
APP = app  # back-compat if any decorator still uses APP

@app.get("/")
def root():
    return {"ok": True, "service": "pathpanda-api"}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

# -----------------------------------------------------------------------------
# Lazy loaders for optional libraries (DO NOT CALL AT IMPORT TIME)
# -----------------------------------------------------------------------------
ARIMA_AVAILABLE = False
ARIMA = None

TF_AVAILABLE = False
load_model = None
Sequential = None
LSTM = None
GRU = None
Dense = None

gym = None

SB3_AVAILABLE = False
DQN = None

def load_arima():
    """statsmodels ARIMA (lightweight; safe to call in startup or on demand)"""
    global ARIMA_AVAILABLE, ARIMA
    if not ARIMA_AVAILABLE:
        try:
            from statsmodels.tsa.arima.model import ARIMA as _ARIMA
            ARIMA = _ARIMA
            ARIMA_AVAILABLE = True
            logger.info("ARIMA ready")
        except Exception as e:
            ARIMA_AVAILABLE = False
            logger.warning("ARIMA unavailable: %s", e)

def load_tensorflow():
    """TensorFlow (heavy)"""
    global TF_AVAILABLE, load_model, Sequential, LSTM, GRU, Dense
    if not TF_AVAILABLE:
        try:
            from tensorflow.keras.models import load_model as _tf_load_model
            from tensorflow.keras.models import Sequential as _TF_Sequential
            from tensorflow.keras.layers import LSTM as _TF_LSTM, GRU as _TF_GRU, Dense as _TF_Dense
            load_model = _tf_load_model
            Sequential = _TF_Sequential
            LSTM = _TF_LSTM
            GRU = _TF_GRU
            Dense = _TF_Dense
            TF_AVAILABLE = True
            logger.info("TensorFlow ready")
        except Exception as e:
            TF_AVAILABLE = False
            logger.warning("TensorFlow unavailable: %s", e)

def load_gymnasium():
    """Gymnasium (moderate)"""
    global gym
    if gym is None:
        try:
            import gymnasium as _gym
            gym = _gym
            logger.info("Gymnasium ready")
        except Exception as e:
            gym = None
            logger.warning("Gymnasium unavailable: %s", e)

def load_stable_baselines3():
    """SB3 (heavy-ish)"""
    global SB3_AVAILABLE, DQN
    if not SB3_AVAILABLE:
        try:
            from stable_baselines3 import DQN as _DQN
            DQN = _DQN
            SB3_AVAILABLE = True
            logger.info("Stable-Baselines3 ready")
        except Exception as e:
            SB3_AVAILABLE = False
            logger.warning("Stable-Baselines3 unavailable: %s", e)

# NOTE: We intentionally DO NOT call the loaders here.
# They’re invoked in startup (conditionally) or right before first use.

# -----------------------------------------------------------------------------
# Globals for models from learners
# -----------------------------------------------------------------------------
ENSEMBLE = None
EXP_W = None

def load_learners():
    """Local learners (lightweight)"""
    global ENSEMBLE, EXP_W
    if ENSEMBLE is None:
        from .learners import OnlineLinear, ExpWeights
        ENSEMBLE = OnlineLinear(lr=0.05, l2=1e-4)
        EXP_W = ExpWeights(eta=2.0)

# Keep learners available (this module is light); safe at import
load_learners()

# -----------------------------------------------------------------------------
# Feature store imports (kept as-is; assumed lightweight at import)
# -----------------------------------------------------------------------------
try:
    from .feature_store import connect as fs_connect
    from .feature_store import get_recent_coverage, get_recent_mdape
    try:
        from .feature_store import insert_prediction as _fs_ins  # type: ignore
    except Exception:
        _fs_ins = None  # type: ignore
    try:
        from .feature_store import log_prediction as _fs_log  # type: ignore
    except Exception:
        _fs_log = None  # type: ignore
except Exception:
    # If relative failed, try absolute module name
    try:
        from feature_store import connect as fs_connect  # type: ignore
        from feature_store import get_recent_coverage, get_recent_mdape  # type: ignore
        try:
            from feature_store import insert_prediction as _fs_ins  # type: ignore
        except Exception:
            _fs_ins = None  # type: ignore
        try:
            from feature_store import log_prediction as _fs_log  # type: ignore
        except Exception:
            _fs_log = None  # type: ignore
    except Exception:
        fs_connect = None
        get_recent_coverage = None
        get_recent_mdape = None
        _fs_ins = None
        _fs_log = None
        logger.warning("feature_store unavailable; logging to FS disabled")

# Choose the best available implementation
if _fs_ins is not None:
    _FS_LOG_IMPL = _fs_ins  # prefer explicit insert when present
elif _fs_log is not None:
    _FS_LOG_IMPL = _fs_log
else:
    _FS_LOG_IMPL = None  # feature store not wired for this build

def fs_log_prediction(con, row: dict) -> None:
    """
    Signature-normalized wrapper for feature_store logging.

    Supports both implementations:
      - impl(con, row)  # uses provided DB connection
      - impl(row)       # self-managed connection
    No-op if feature_store is unavailable.
    """
    if _FS_LOG_IMPL is None:
        return  # feature store not wired in this build

    try:
        # Preferred: (con, row)
        return _FS_LOG_IMPL(con, row)
    except TypeError:
        # Fallback: implementation expects only (row)
        return _FS_LOG_IMPL(row)

# DUCK utils — try db.duck first, fall back to duck
try:
    from .db.duck import init_schema, insert_prediction, matured_predictions_now, insert_outcome
except Exception:
    from .duck import init_schema, insert_prediction, matured_predictions_now, insert_outcome  # type: ignore

# -----------------------------------------------------------------------------
# Settings (pydantic-settings v2)
# -----------------------------------------------------------------------------
def _poly_key() -> str:
    env_k = (os.getenv("PT_POLYGON_KEY") or os.getenv("POLYGON_KEY") or "").strip()
    if env_k:
        return env_k
    # During class creation, 'settings' may not exist; we guard it.
    try:
        return (settings.polygon_key or "").strip()
    except Exception:
        return ""

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PT_", case_sensitive=False)

    # API keys / upstreams
    polygon_key: Optional[str] = Field(default_factory=_poly_key)
    news_api_key: Optional[str] = None

    # Infra & app
    redis_url: str = Field(default_factory=lambda: os.getenv("PT_REDIS_URL", "redis://localhost:6379/0"))
    cors_origins_raw: Optional[str] = None

    # Cookies
    cookie_name: str = Field("pt_app", validation_alias="PT_COOKIE_NAME")
    cookie_max_age: int = Field(60 * 60 * 24, validation_alias="PT_COOKIE_MAX_AGE")  # 1 day
    cookie_secure: bool = Field(default_factory=lambda: bool(os.getenv("RENDER")) or os.getenv("ENV","dev") == "prod")

    # Limits / budgets
    n_paths_max: int = Field(default_factory=lambda: int(os.getenv("PT_N_PATHS_MAX", "10000")))
    horizon_days_max: int = Field(default_factory=lambda: int(os.getenv("PT_HORIZON_DAYS_MAX", "3650")))
    lookback_days_max: int = Field(default_factory=lambda: int(os.getenv("PT_LOOKBACK_DAYS_MAX", str(365*10))))
    pathday_budget_max: int = Field(default_factory=lambda: int(os.getenv("PT_PATHDAY_BUDGET_MAX", "500000")))
    max_active_runs: int = Field(default_factory=lambda: int(os.getenv("PT_MAX_ACTIVE_RUNS", "2")))
    run_ttl_seconds: int = Field(default_factory=lambda: int(os.getenv("PT_RUN_TTL_SECONDS", "3600")))

    # Defaults for /simulate convenience
    predictive_defaults: dict = {
        "X:BTCUSD": {"horizon_days": 365, "n_paths": 10000, "lookback_preset": "3y"},
        "NVDA":     {"horizon_days": 30,  "n_paths": 5000,  "lookback_preset": "180d"},
    }

    # Watchlists (strings → comma lists)
    # (defined after TOP_* constants appear; we'll set them later)
    watchlist_equities: str = Field("", validation_alias="PT_WATCHLIST_EQUITIES")
    watchlist_cryptos: str = Field("", validation_alias="PT_WATCHLIST_CRYPTOS")

    # Feature flags
    shortlist_disable: bool = Field(False, validation_alias="PT_SHORTLIST_DISABLE")

    # Access control (optional)
    open_access: bool = Field(default_factory=lambda: os.getenv("PT_OPEN_ACCESS", "1") == "1")
    pt_api_key: Optional[str] = None

settings = Settings()

# -----------------------------------------------------------------------------
# Helpers / config (independent)
# -----------------------------------------------------------------------------
def _env_list(name: str, default: List[str] | None = None) -> List[str]:
    s = os.getenv(name, "")
    if not s:
        return default or []
    return [x.strip() for x in s.split(",") if x.strip()]

def _poly_crypto_to_app(sym: str) -> str:
    # "X:BTCUSD" -> "BTC-USD"
    s = (sym or "").upper()
    if s.startswith("X:") and s.endswith("USD"):
        return f"{s[2:-3]}-USD"
    return s

def require_tf() -> None:
    if not TF_AVAILABLE:
        raise HTTPException(status_code=503, detail="TensorFlow is not installed on the server.")

def _sigmoid(z: float) -> float:
    z = float(np.clip(z, -60.0, 60.0))
    return 1.0 / (1.0 + math.exp(-z))

def _today_utc_date() -> date:
    return datetime.now(timezone.utc).date()

async def maybe_require_key(request: Request):
    # Open testing: allow all when open_access = True
    open_access = bool(getattr(settings, "open_access", True))
    expected = getattr(settings, "pt_api_key", None) or os.getenv("PT_API_KEY", "")
    if open_access or not expected:
        return True

    key = request.headers.get("X-API-Key") or request.query_params.get("api_key") or ""
    if key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

async def _model_key(symbol: str) -> str:
    return f"model:{symbol.upper()}"

TOP_STOCKS = [
    'NVDA', 'MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'AVGO', 'TSLA', 'TSM', 'BRK.B',
    'ORCL', 'JPM', 'WMT', 'LLY', 'V', 'NFLX', 'MA', 'XOM', 'JNJ', 'PLTR',
    'COST', 'ABBV', 'BABA', 'ASML', 'AMD', 'HD', 'BAC', 'PG', 'UNH', 'SAP',
    'GE', 'CVX', 'KO', 'CSCO', 'IBM', 'AZN', 'NVO', 'NVS', 'TMUS', 'WFC',
    'TM', 'MS', 'GS', 'PM', 'CAT', 'CRM', 'ABT', 'HSBC', 'AXP', 'RTX',
    'MRK', 'LIN', 'SHEL', 'MU', 'SHOP', 'MCD', 'RY', 'APP', 'TMO', 'DIS',
    'UBER', 'ANET', 'PEP', 'BX', 'NOW', 'T', 'PDD', 'SONY', 'INTU', 'BLK',
    'ARM', 'LRCX', 'INTC', 'C', 'QCOM', 'MUFG', 'AMAT', 'VZ', 'NEE', 'GEV',
    'SCHW', 'HDB', 'BKNG', 'BA', 'TXN', 'ISRG', 'AMGN', 'ACN', 'TJX', 'APH',
    'SPGI', 'SAN', 'ETN', 'DHR', 'UL', 'PANW', 'GILD', 'ADBE', 'BSX', 'PFE'
]

TOP_CRYPTOS = [
    'BTC', 'ETH', 'USDT', 'BNB', 'XRP', 'SOL', 'USDC', 'DOGE', 'TRX', 'ADA',
    'LINK', 'USDe', 'HYPE', 'SUI', 'XLM', 'AVAX', 'BCH', 'LTC', 'HBAR', 'LEO',
    'MNT', 'SHIB', 'TON', 'CRO', 'DOT', 'XMR', 'DAI', 'UNI', 'WLFI', 'OKB',
    'AAVE', 'BGB', 'ENA', 'PEPE', 'NEAR', 'APT', 'ZEC', 'TAO', 'ETC', 'ASTER',
    'IP', 'ONDO', 'USD1', 'WLD', 'PYUSD', 'POL', 'ICP', 'ARB', 'M', 'KCS',
    'KAS', 'ALGO', 'PUMP', 'ATOM', 'VET', 'PENGU', 'PI', 'SEI', 'FLR', 'RENDER',
    'FIL', 'SKY', 'BONK', 'TRUMP', 'JUP', 'XPL', 'IMX', 'GT', 'SPX', '2Z',
    'XDC', 'CAKE', 'OP', 'QNT', 'INJ', 'PAXG', 'FET', 'TIA', 'FDUSD', 'STX',
    'LDO', 'CRV', 'MYX', 'XAUt', 'AERO', 'DEXE', 'PYTH', 'FLOKI', 'KAIA', 'GRT',
    'ETHFI', 'NEXO', 'RLUSD', 'S', 'ENS', 'PENDLE', 'IOTA', 'CFX', 'RAY', 'XTZ'
]

def _norm_crypto_symbol(s: str) -> str:
    """
    Normalize crypto tickers to BASE-USD (e.g., 'BTC' -> 'BTC-USD').
    If already like 'BASE-USD', keep it. Uppercases everything.
    """
    s = (s or "").strip().upper()
    if not s:
        return ""
    if s.startswith("X:"):  # polygon 'X:BTCUSD' style → keep if your fetchers expect it
        return s
    if "-" in s:           # already 'BASE-QUOTE'
        return s
    if s.endswith("USD"):  # 'BASEUSD' → 'BASE-USD'
        base = s[:-3]
        return f"{base}-USD"
    return f"{s}-USD"      # plain base → USD pair

# Settings watchlists now that constants exist
if not settings.watchlist_equities:
    settings.watchlist_equities = ",".join(TOP_STOCKS)
if not settings.watchlist_cryptos:
    settings.watchlist_cryptos = ",".join(_norm_crypto_symbol(x) for x in TOP_CRYPTOS)

# Canonical watchlist sets derived from settings
WL_EQ = sorted({t.strip().upper() for t in settings.watchlist_equities.split(",") if t.strip()})[:200]
WL_CR = sorted({_norm_crypto_symbol(t) for t in settings.watchlist_cryptos.split(",") if t.strip()})[:200]

# Config knobs (reasonable defaults)
STAT_BARS_QUICK = 252          # ~1y worth of bars to estimate μ/σ
STAT_BARS_DEEP  = 504          # ~2y for deep
MU_CAP_QUICK    = 1.00         # |μ| ≤ 100%/yr for quick
MU_CAP_DEEP     = 0.50         # |μ| ≤ 50%/yr
SIGMA_CAP       = 0.80         # σ ≤ 80%/yr
LONG_HORIZON_DAYS = 365
LONG_HORIZON_SHRINK_FLOOR = 0.25  # never shrink below 25%

def _bars_per_day(timespan: str) -> int:
    if timespan == "day":   return 1
    if timespan == "hour":  return 24
    return 390  # minute

def auto_rel_levels(
    sigma_ann: Optional[float],
    horizon_days: int,
    kind: str = "equity",
) -> Tuple[float, ...]:
    """
    Volatility-aware ladder (percent moves from S0). Adapts to regime.
    Returns a tuple like (-0.20, -0.10, 0.0, 0.10, 0.20).
    """
    if sigma_ann is None or not math.isfinite(sigma_ann):
        sigma_ann = 0.30 if kind == "equity" else 0.80

    # simple mapping: wider bands for higher vol & longer horizon
    scale = float(np.clip(sigma_ann, 0.05, 1.50)) * math.sqrt(max(int(horizon_days), 1) / 252.0)
    base = max(0.05, min(0.40, 1.5 * scale))  # clamp within sane UI range
    rungs = (-2*base, -base, 0.0, base, 2*base)
    return tuple(float(x) for x in rungs)

# ---------- Targets & Odds (first-passage + terminal hit) ----------
def _barrier_stats(paths: np.ndarray, level: float, side: str, bars_per_day: int) -> dict:
    """
    paths: (N, H+1) prices (includes S0 at t=0)
    level: absolute price barrier
    side:  'up' (≥ level) or 'down' (≤ level)
    bars_per_day: 1=day, 24=hour, 390=minute; 0/None allowed (falls back)
    """
    # Vectorized hit matrix (N,H+1)
    hits = (paths >= float(level)) if side == "up" else (paths <= float(level))

    # Ever-hit probability across the horizon
    ever = hits.any(axis=1)
    hitEver = float(ever.mean()) if ever.size else 0.0

    # Terminal beyond-level probability (end-of-horizon)
    hitByEnd = float(hits[:, -1].mean()) if hits.shape[1] > 0 else 0.0

    # First-passage index in bars (nan if never)
    first_idx = np.argmax(hits, axis=1).astype(float)  # first True index; 0 if none
    first_idx[~ever] = np.nan

    # Bars → days for readability in UI (ceil to whole days)
    if bars_per_day and bars_per_day > 0:
        t_days = np.ceil(first_idx / float(bars_per_day))
    else:
        t_days = first_idx  # already "bars" if you ever want that view

    tMedDays = float(np.nanmedian(t_days)) if np.isfinite(t_days).any() else None
    return {"hitEver": hitEver, "hitByEnd": hitByEnd, "tMedDays": tMedDays}

def compute_targets_block(
    paths: np.ndarray,
    S0: float,
    horizon_days: int,
    bars_per_day: int,
    rel_levels: Optional[Sequence[float]] = None,
) -> dict:
    """
    Build the artifact.targets block the UI expects.

    rel_levels: sequence of percent moves (e.g., -0.20 for -20%). If None, we
    auto-generate using volatility-aware ladder (auto_rel_levels).
    """
    # Defensive checks
    if paths.ndim != 2 or paths.shape[1] < 2:
        raise ValueError(f"paths must be (N, H+1) price matrix; got {paths.shape}")
    N, H_plus_1 = paths.shape
    if not math.isfinite(S0) or S0 <= 0:
        raise ValueError(f"S0 must be positive; got {S0}")

    rows: list[dict] = []

    # Determine default ladder if not provided
    # NOTE: You can pass the regime/kind from your symbol or request.
    if rel_levels is None:
        kind = "crypto" if str(S0) and isinstance(S0, (int, float)) else "equity"  # trivial default
        rel_levels = auto_rel_levels(sigma_ann=None, horizon_days=horizon_days, kind=kind)

    # Ensure Spot row (0.0) is present exactly once and in the middle-ish
    levels = list(dict.fromkeys([float(x) for x in rel_levels]))  # de-dup but keep order
    if 0.0 not in levels:
        mid = len(levels) // 2
        levels.insert(mid, 0.0)

    # Build rows
    for r in levels:
        if abs(r) < 1e-12:
            rows.append({
                "label": "Spot",
                "price": float(S0),
                "side": "mid",
                "hitEver": None,
                "hitByEnd": None,
                "tMedDays": None,
            })
            continue

        label = f"{'+' if r > 0 else ''}{int(round(r * 100))}%"
        price = float(S0 * (1.0 + r))
        side = "up" if r > 0 else "down"
        stats = _barrier_stats(paths, price, side, bars_per_day)
        rows.append({"label": label, "price": price, "side": side, **stats})

    return {
        "spot": float(S0),
        "horizon_days": int(horizon_days),
        "levels": rows,
    }

def _horizon_shrink(h_days: int) -> float:
    """Shrink μ as horizon increases: 1.0 at 1y, ~0.32 at 10y, floored at 0.25."""
    if h_days <= LONG_HORIZON_DAYS:
        return 1.0
    s = (LONG_HORIZON_DAYS / float(max(1, min(h_days, 3650)))) ** 0.5
    return max(LONG_HORIZON_SHRINK_FLOOR, min(1.0, s))

def _winsorize(arr: np.ndarray, p_lo=0.005, p_hi=0.995) -> np.ndarray:
    lo, hi = np.quantile(arr, [p_lo, p_hi]) if arr.size else (0.0, 0.0)
    return np.clip(arr, lo, hi)

def _ewma_sigma(returns: np.ndarray, lam: float = 0.94) -> float:
    """EWMA of daily (or bar) returns; returns annualized σ."""
    if returns.size == 0:
        return 0.2
    var = 0.0
    for r in returns[::-1]:
        var = lam * var + (1 - lam) * (r * r)
    sigma_bar = math.sqrt(max(var, 1e-12))
    return sigma_bar  # per-bar (we annualize outside)

def _student_t_noise(df: int, size: tuple[int,...], rng: np.random.Generator) -> np.ndarray:
    """Student-t normalized to unit variance."""
    z = rng.standard_t(df, size=size)
    scale = math.sqrt(df / (df - 2.0))
    return z / scale

def _iv30_or_none(symbol: str) -> float | None:
    """Hook for 30d ATM IV if you have it. Return None to skip anchoring."""
    try:
        return None
    except Exception:
        return None

# ====== Engines ============================================================

def simulate_gbm_student_t(
    S0: float,
    mu_ann: float,
    sigma_ann: float,
    horizon_days: int,
    n_paths: int,
    df_t: int = 7,
    antithetic: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """Return price paths shape (n_paths, H+1) including S0."""
    H = int(horizon_days)
    rng = np.random.default_rng(seed)
    # daily params
    mu_d = mu_ann / 252.0
    sig_d = sigma_ann / math.sqrt(252.0)
    # paths
    half = n_paths // 2 if antithetic else n_paths
    noises = _student_t_noise(df_t, size=(half, H), rng=rng)
    if antithetic:
        noises = np.vstack([noises, -noises])
        if noises.shape[0] < n_paths:  # odd
            extra = _student_t_noise(df_t, size=(1, H), rng=rng)
            noises = np.vstack([noises, extra])
        noises = noises[:n_paths, :]
    # log-GBM increments
    dlogS = (mu_d - 0.5 * sig_d * sig_d) + sig_d * noises
    log_paths = np.cumsum(dlogS, axis=1)
    paths = S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))
    return paths

def simulate_bootstrap_blocks(
    S0: float,
    log_rets_hist: np.ndarray,
    horizon_days: int,
    n_paths: int,
    block_len: int = 15,
    antithetic: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simple moving-block bootstrap of historical log-returns.
    Concatenate random blocks until ≥ H, then trim.
    """
    H = int(horizon_days)
    rng = np.random.default_rng(seed)
    T = log_rets_hist.size
    if T < block_len:
        raise ValueError("Insufficient history for block bootstrap")
    n_blocks = max(1, math.ceil(H / block_len))
    # draw block starts
    starts = rng.integers(low=0, high=T - block_len + 1, size=(n_paths, n_blocks))
    # build log-return sequences
    seq = np.empty((n_paths, n_blocks * block_len), dtype=float)
    for i in range(n_paths):
        chunks = [log_rets_hist[s:s + block_len] for s in starts[i]]
        seq[i] = np.concatenate(chunks)
    seq = seq[:, :H]
    # antithetic pairing in log space (mirror around mean≈0)
    if antithetic:
        half = n_paths // 2
        if half > 0:
            seq[:half] = seq[:half]
            seq[half:half*2] = -seq[:half]
        # odd path count: leave last as-is
    # cum-sum + exponentiate
    log_cum = np.cumsum(seq, axis=1)
    paths = S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_cum]))
    return paths

# ---- CORS allowlist -------------------------------------------------------
def _parse_cors_list(raw: Optional[str]) -> list[str]:
    if not raw or not raw.strip():
        return []
    s = raw.strip()
    if s.startswith("["):
        try:
            arr = json.loads(s)
            return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
    return [p.strip() for p in s.split(",") if p.strip()]

DEFAULT_ORIGINS = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:5174", "http://127.0.0.1:5174",
    "http://localhost:3000", "http://127.0.0.1:3000",
    "https://simetrix.io", "https://www.simetrix.io",
    "https://simetrix.vercel.app",
]

CORS_ORIGINS = _parse_cors_list(settings.cors_origins_raw)
ALLOWED_ORIGINS = sorted(set(CORS_ORIGINS or DEFAULT_ORIGINS))

# --- Auth dependency (define BEFORE any route uses it)
def require_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_pt_key: str | None  = Header(default=None, alias="x-pt-key"),  # compat
    api_key: str | None   = Query(default=None),                     # compat
) -> bool:
    expected = os.getenv("PT_API_KEY", "dev-local")
    provided = (x_api_key or x_pt_key or api_key or "").strip()
    if not provided or provided != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

# --- CORS (apply after ALLOWED_ORIGINS is computed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "X-API-Key", "x-pt-key", "Authorization"],
    max_age=86400,
)

# --- Optional static frontend
STATIC_DIR = os.getenv("PT_STATIC_DIR", "frontend/dist")
static_path = Path(STATIC_DIR).resolve()
if static_path.is_dir():
    app.mount("/app", StaticFiles(directory=str(static_path), html=True), name="app")
    logger.info(f"Mounted static frontend from {static_path} at /app")

# --- Redis client (created on startup so we don't block binding)
REDIS: Redis | None = None

# --- RL constants
RL_WINDOW = int(os.getenv("PT_RL_WINDOW", "100"))

# --- Background GC + startup/shutdown
async def _gc_loop():
    while True:
        try:
            if REDIS:
                async for akey in REDIS.scan_iter(match="artifact:*", count=500):
                    try:
                        run_id = akey.split(":", 1)[1]
                    except Exception:
                        continue
                    exists = await REDIS.exists(f"run:{run_id}")
                    if not exists:
                        await REDIS.delete(akey)
        except Exception as e:
            logger.error(f"GC loop error: {e}")
        await asyncio.sleep(60)

_gc_task: asyncio.Task | None = None

@app.on_event("startup")
async def _startup():
    global REDIS, _gc_task

    # 1) Bind quickly; then do heavy/optional imports
    try:
        # Optional/Heavy libs gated by env flags (defaults chosen to be safe)
        if os.getenv("ENABLE_ARIMA", "1") == "1":
            load_arima()
        if os.getenv("ENABLE_TF", "0") == "1":  # default off on small Render plans
            load_tensorflow()
        if os.getenv("ENABLE_GYM", "0") == "1":
            load_gymnasium()
        if os.getenv("ENABLE_SB3", "0") == "1":
            load_stable_baselines3()
    except Exception as e:
        logger.warning("[startup] optional loader failed: %s", e)

    # 2) Redis (non-blocking creation; connects on first use)
    try:
        REDIS = Redis.from_url(settings.redis_url, decode_responses=True)
        logger.info("Redis client ready: %s", settings.redis_url)
    except Exception as e:
        REDIS = None
        logger.warning("Redis unavailable: %s", e)

    # 3) Kick off background GC without blocking startup
    try:
        _gc_task = asyncio.create_task(_gc_loop())
    except Exception as e:
        logger.warning("Failed to start GC loop: %s", e)

@app.on_event("shutdown")
async def _shutdown():
    global REDIS, _gc_task
    try:
        if _gc_task:
            _gc_task.cancel()
    except Exception:
        pass
    try:
        if REDIS:
            await REDIS.close()
    except Exception:
        pass

# -----------------------------------------------------------------------------
# LLM summary helper (kept intact)
# -----------------------------------------------------------------------------
async def llm_summarize_async(
    prompt_user: dict,
    *,
    prefer_xai: bool,
    xai_key: str | None,
    oai_key: str | None,
    json_schema: dict | None = None,
    timeout: float = 20.0,
) -> dict:
    """
    Returns a dict (parsed from choices[0].message.content). Falls back on error.
    """
    async def _post(url: str, headers: dict, payload: dict) -> dict:
        async with httpx.AsyncClient(timeout=timeout) as cli:
            r = await cli.post(url, headers=headers, json=payload)
            r.raise_for_status()
            return r.json()

    def _fallback() -> dict:
        return {"list": []}

    try:
        if prefer_xai and xai_key:
            data = await _post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {xai_key}"},
                payload={
                    "model": os.getenv("XAI_MODEL", "grok-2-mini"),
                    "messages": [
                        {"role": "system", "content": "Be factual, concise, compliance-safe."},
                        prompt_user,
                    ],
                    # "response_format": {"type": "json_schema", "json_schema": json_schema},
                    "temperature": 0.2,
                },
            )
            content = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
            return content if isinstance(content, dict) else json.loads(content)

        if oai_key:
            data = await _post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {oai_key}"},
                payload={
                    "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    "messages": [
                        {"role": "system", "content": "Be factual, concise, compliance-safe."},
                        prompt_user,
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2,
                },
            )
            content = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
            return content if isinstance(content, dict) else json.loads(content)

        return _fallback()

    except Exception as e:
        logger.info(f"LLM summary failed; fallback used: {e}")
        return _fallback()

# -----------------------------------------------------------------------------
# Lightweight online learner + exp-weights (kept intact)
# -----------------------------------------------------------------------------
class SGDOnline:
    """
    Simple logistic-regression learner with SGD + L2.
    Usage: init(d); update(x, y) with x shape [d], y in {0,1}
    """
    def __init__(self, lr: float = 0.05, l2: float = 1e-4):
        self.lr = float(lr)
        self.l2 = float(l2)
        self.w = None  # includes bias at index 0

    def init(self, d: int):
        # bias + d features
        self.w = np.zeros(int(d) + 1, dtype=float)

    @staticmethod
    def _sigmoid(z: float) -> float:
        z = float(np.clip(z, -60.0, 60.0))
        return 1.0 / (1.0 + math.exp(-z))

    def update(self, x, y):
        if self.w is None:
            raise RuntimeError("Call init(d) before update()")
        xb = np.concatenate([[1.0], np.asarray(x, dtype=float)])
        y = float(y)
        z = float(np.dot(self.w, xb))
        p = self._sigmoid(z)
        # gradient of logloss + L2 (no penalty on bias)
        grad = (p - y) * xb
        grad[1:] += self.l2 * self.w[1:]
        self.w -= self.lr * grad

class EW:
    """
    Exponential Weights combiner for ensembling probabilities.
    Usage: init(n); update(losses)
    """
    def __init__(self, eta: float = 2.0):
        self.eta = float(eta)
        self.w = None

    def init(self, n: int):
        n = int(n)
        if n <= 0:
            raise ValueError("n must be > 0")
        self.w = np.ones(n, dtype=float) / float(n)

    def update(self, losses: np.ndarray):
        if self.w is None:
            raise RuntimeError("Call init(n) before update()")
        L = np.asarray(losses, dtype=float)
        w_new = self.w * np.exp(-self.eta * L)
        s = float(w_new.sum())
        self.w = (w_new / s) if s > 0 else (np.ones_like(w_new) / w_new.size)
        return self.w

# -----------------------------------------------------------------------------
# Ingest (kept intact with a tiny import fix for `time`)
# -----------------------------------------------------------------------------
async def ingest_grouped_daily(d: date):
    """
    Ingest Polygon grouped daily bars for US stocks and global crypto for a given UTC date.
    Creates/updates DuckDB table `bars_daily` (PRIMARY KEY: symbol, ts).
    Returns: {"ok": True, "upserted": <int>}
    """
    key = _poly_key()
    day = d.isoformat()

    async def _fetch_json(cli: httpx.AsyncClient, url: str, params: dict) -> dict:
        base_delays = (0.8, 1.6, 3.2, 6.4)  # gentle backoff
        for _, delay in enumerate((*base_delays, None)):
            try:
                r = await cli.get(url, params=params, timeout=30.0)
                r.raise_for_status()
                return r.json() or {}
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status in (429, 500, 502, 503, 504) and delay is not None:
                    # Honor Retry-After or rate-limit reset if present
                    ra = e.response.headers.get("Retry-After")
                    reset = e.response.headers.get("X-RateLimit-Reset")
                    sleep_s = None
                    if ra:
                        try:
                            sleep_s = float(ra)
                        except Exception:
                            sleep_s = None
                    if (sleep_s is None) and reset:
                        try:
                            sleep_s = max(0.0, float(reset) - time.time())  # reset is epoch seconds
                        except Exception:
                            pass
                    if sleep_s is None:
                        # final fallback: jittered fixed delay
                        sleep_s = delay + random.random()
                    await asyncio.sleep(min(15.0, max(0.5, sleep_s)))
                    continue
                raise
            except httpx.HTTPError:
                if delay is not None:
                    await asyncio.sleep(delay)
                    continue
                raise

    # --- Fetch Polygon grouped aggs ---
    params = {"adjusted": "true", "apiKey": key}
    async with httpx.AsyncClient() as cli:
        stocks = await _fetch_json(
            cli, f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{day}", params
        )
        crypto = await _fetch_json(
            cli, f"https://api.polygon.io/v2/aggs/grouped/locale/global/market/crypto/{day}", params
        )

    rows1 = stocks.get("results") or []
    rows2 = crypto.get("results") or []

    # --- Transform -> upsert payloads ---
    to_upsert = []

    # US stocks (tickers like "AAPL")
    for r in rows1:
        tkr = (r.get("T") or "").upper()
        if not tkr:
            continue
        to_upsert.append((
            tkr,
            day,
            float(r.get("o") or 0.0),
            float(r.get("h") or 0.0),
            float(r.get("l") or 0.0),
            float(r.get("c") or 0.0),
            float(r.get("v") or 0.0),
        ))

    # Crypto (tickers like "X:BTCUSD" -> "BTC-USD")
    for r in rows2:
        raw = r.get("T") or ""
        tkr = _poly_crypto_to_app(raw)  # e.g., "X:BTCUSD" -> "BTC-USD"
        if not tkr or "-USD" not in tkr:
            continue
        to_upsert.append((
            tkr,
            day,
            float(r.get("o") or 0.0),
            float(r.get("h") or 0.0),
            float(r.get("l") or 0.0),
            float(r.get("c") or 0.0),
            float(r.get("v") or 0.0),
        ))

    if not to_upsert:
        return {"ok": True, "upserted": 0}

    # --- DuckDB upsert ---
    if fs_connect is None:
        raise HTTPException(status_code=500, detail="feature_store.connect unavailable")

    con = fs_connect()
    try:
        con.execute("BEGIN")
        con.execute("""
            CREATE TABLE IF NOT EXISTS bars_daily (
                symbol TEXT,
                ts DATE,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY(symbol, ts)
            )
        """)

        con.executemany(
            """
            INSERT OR REPLACE INTO bars_daily
            (symbol, ts, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            to_upsert,
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()

    return {"ok": True, "upserted": len(to_upsert)}

# -----------------------------------------------------------------------------
# Misc helpers (kept intact)
# -----------------------------------------------------------------------------
def _bars_per_day(timespan: str) -> int:
    if timespan == "day":   return 1
    if timespan == "hour":  return 24
    return 390  # minute

# -----------------------------------------------------------------------------
# __main__ (local only). On Render, your Start Command runs the server.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("src.predictive_api:app", host="0.0.0.0", port=port, log_level="info")
