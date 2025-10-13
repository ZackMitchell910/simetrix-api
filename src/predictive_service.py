# --- stdlib
from __future__ import annotations
import os, json, asyncio, logging, math, pickle
from datetime import datetime, timedelta, date, timezone
from uuid import uuid4
from typing import List, Optional, Any, Callable, Dict, Literal, Sequence, Tuple
from contextlib import asynccontextmanager

# --- third-party
from dotenv import load_dotenv; load_dotenv()
import numpy as np
import pandas as pd
import random
import httpx
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, WebSocket, Query, Header, Body, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from redis.asyncio import Redis
from secrets import token_urlsafe
# (optional libs guarded)
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except Exception:
    ARIMA_AVAILABLE = False

TF_AVAILABLE = False
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
from .feature_store import get_recent_coverage, get_recent_mdape
try:
    from .feature_store import insert_prediction as _fs_ins  # type: ignore
except Exception:
    _fs_ins = None  # type: ignore

try:
    from .feature_store import log_prediction as _fs_log  # type: ignore
except Exception:
    _fs_log = None  # type: ignore

# If relative failed, try absolute module name
if _fs_ins is None and _fs_log is None:
    try:
        from feature_store import insert_prediction as _fs_ins  # type: ignore
    except Exception:
        _fs_ins = None  # type: ignore
    try:
        from feature_store import log_prediction as _fs_log  # type: ignore
    except Exception:
        _fs_log = None  # type: ignore

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

# --- logging / app (single instance)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("predictive")
logger = logging.getLogger(__name__)

# ----------------- helpers / config -----------------
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
        # your existing fallback; keep behavior identical
        return {"list": []}  # or whatever you already return

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
                    # Enable when you want strict schema:
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
# --- Lightweight online learner + exp-weights (missing classes) ---
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

def _ensure_redis() -> Redis:
    if REDIS is None:
        raise HTTPException(status_code=503, detail="Redis unavailable")
    return REDIS

def _env_list(name: str, default: List[str] | None = None) -> List[str]:
    s = os.getenv(name, "")
    if not s:
        return default or []
    return [x.strip() for x in s.split(",") if x.strip()]

def _poly_key() -> str:
    env_k = (os.getenv("PT_POLYGON_KEY") or os.getenv("POLYGON_KEY") or "").strip()
    if env_k:
        return env_k
    try:
        return (settings.polygon_key or "").strip()
    except Exception:
        return ""
def _poly_crypto_to_app(sym: str) -> str:
    # "X:BTCUSD" -> "BTC-USD"
    s = (sym or "").upper()
    if s.startswith("X:") and s.endswith("USD"):
        return f"{s[2:-3]}-USD"
    return s
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
        for i, delay in enumerate((*base_delays, None)):
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
                        try: sleep_s = float(ra)
                        except: sleep_s = None
                    if (sleep_s is None) and reset:
                        try:
                            # reset is epoch seconds on many APIs
                            sleep_s = max(0.0, float(reset) - time.time())
                        except:
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
# --- Settings (pydantic-settings v2)
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PT_", case_sensitive=False)
    polygon_key: Optional[str] = Field(default_factory=_poly_key)
    news_api_key: Optional[str] = None
    redis_url: str = Field(default_factory=lambda: os.getenv("PT_REDIS_URL", "redis://localhost:6379/0"))
    cors_origins_raw: Optional[str] = None
    cookie_name: str = Field("pt_app", validation_alias="PT_COOKIE_NAME")
    cookie_max_age: int = Field(60 * 60 * 24, validation_alias="PT_COOKIE_MAX_AGE")  # 1 day
    cookie_secure: bool = Field(default_factory=lambda: bool(os.getenv("RENDER")) or os.getenv("ENV","dev") == "prod")
    n_paths_max: int = Field(default_factory=lambda: int(os.getenv("PT_N_PATHS_MAX", "10000")))
    horizon_days_max: int = Field(default_factory=lambda: int(os.getenv("PT_HORIZON_DAYS_MAX", "3650")))
    lookback_days_max: int = Field(default_factory=lambda: int(os.getenv("PT_LOOKBACK_DAYS_MAX", str(365*10))))
    pathday_budget_max: int = Field(default_factory=lambda: int(os.getenv("PT_PATHDAY_BUDGET_MAX", "500000")))
    max_active_runs: int = Field(default_factory=lambda: int(os.getenv("PT_MAX_ACTIVE_RUNS", "2")))
    run_ttl_seconds: int = Field(default_factory=lambda: int(os.getenv("PT_RUN_TTL_SECONDS", "3600")))
    predictive_defaults: dict = {
        "X:BTCUSD": {"horizon_days": 365, "n_paths": 10000, "lookback_preset": "3y"},
        "NVDA": {"horizon_days": 30, "n_paths": 5000, "lookback_preset": "180d"},
        }
    watchlist_equities: str = Field(
        ",".join(TOP_STOCKS),
        validation_alias="PT_WATCHLIST_EQUITIES",
    )
    watchlist_cryptos: str = Field(
        ",".join(_norm_crypto_symbol(x) for x in TOP_CRYPTOS),
        validation_alias="PT_WATCHLIST_CRYPTOS",
    )
    shortlist_disable: bool = Field(False, validation_alias="PT_SHORTLIST_DISABLE")


settings = Settings()
WL_EQ = sorted({t.strip().upper() for t in settings.watchlist_equities.split(",") if t.strip()})[:200]
WL_CR = sorted({_norm_crypto_symbol(t) for t in settings.watchlist_cryptos.split(",") if t.strip()})[:200]

# Config knobs (reasonable defaults)
STAT_BARS_QUICK = 252          # ~1y worth of bars to estimate μ/σ
STAT_BARS_DEEP  = 504          # ~2y for deep
MU_CAP_QUICK    = 1.00         # |μ| ≤ 100%/yr for quick
MU_CAP_DEEP     = 0.50         # |μ| ≤ 50%/yr for deep
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

# --- (ONCE IV IS IMPLEMENTED IV/Expected-move ladder) ---------------------
# def iv_expected_move(iv_annual: Optional[float], horizon_days: int, denom: int = 365) -> Optional[float]:
#     """
#     Approx expected move as a fraction (1-sigma), using ATM IV.
#     For equities denom=365; for equities-trading-days, denom=252.
#     """
#     if iv_annual is None or not math.isfinite(iv_annual):
#         return None
#     return float(iv_annual) * math.sqrt(max(int(horizon_days), 1) / float(denom))
#
# def rel_levels_from_expected_move(em: float) -> Tuple[float, ...]:
#     """
#     Build symmetric levels around ±EM (multiples). Keep ≤ 8 rows.
#     """
#     m = float(abs(em))
#     # Cap to avoid absurd ladders in extreme IV regimes
#     m = max(0.02, min(m, 0.60))
#     return tuple([-2*m, -1*m, -0.5*m, 0.0, 0.5*m, 1*m, 2*m])


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
        # TODO (optional): infer 'kind' from symbol (e.g., X:BTCUSD -> crypto).
        rel_levels = auto_rel_levels(sigma_ann=None, horizon_days=horizon_days, kind=kind)

        # --- (optional, commented) Use expected-move ladder if IV is available ---
        # iv_annual = None
        # if hasattr(globals(), "options_summary"):  # or bring your IV in scope
        #     iv_annual = options_summary.get("atm_iv_annual")  # define your source
        # em = iv_expected_move(iv_annual, horizon_days)
        # if em is not None:
        #     rel_levels = rel_levels_from_expected_move(em)

    # Ensure Spot row (0.0) is present exactly once and in the middle-ish
    levels = list(dict.fromkeys([float(x) for x in rel_levels]))  # de-dup but keep order
    if 0.0 not in levels:
        # insert Spot at best-effort center
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
    w = 1.0
    var = 0.0
    for r in returns[::-1]:
        var = lam * var + (1 - lam) * (r * r)
    sigma_bar = math.sqrt(max(var, 1e-12))
    return sigma_bar  # per-bar (we annualize outside)

def _student_t_noise(df: int, size: tuple[int,...], rng: np.random.Generator) -> np.ndarray:
    """Student-t normalized to unit variance."""
    # Var(t_v) = v/(v-2) for v>2 → normalize to unit variance
    z = rng.standard_t(df, size=size)
    scale = math.sqrt(df / (df - 2.0))
    return z / scale

def _iv30_or_none(symbol: str) -> float | None:
    """Hook for 30d ATM IV if you have it. Return None to skip anchoring."""
    try:
        # TODO: wire to your options cache if available
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


# ---- CORS allowlist
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

CORS_ORIGINS = _parse_cors_list(settings.cors_origins_raw)
DEFAULT_ORIGINS = [
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:5174", "http://127.0.0.1:5174",
    "http://localhost:3000", "http://127.0.0.1:3000",
    "https://simetrix.io", "https://www.simetrix.io",
    "https://simetrix.vercel.app",
]
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

# --- Create app
app = FastAPI(
    title="Simetrix API",
    version="1.3.0",
    docs_url="/api-docs",
    redoc_url=None,
    redirect_slashes=False,
)
APP = app  # back-compat if any decorator still uses APP

# --- Redis client (text mode)
REDIS: Redis | None = Redis.from_url(settings.redis_url, decode_responses=True)

# --- CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,          # use your computed allowlist
    allow_origin_regex=r"https://.*\.vercel\.app$",
    allow_credentials=True,                 # set True if you’ll use cookies
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

@app.on_event("shutdown")
async def _shutdown():
    try:
        if REDIS:
            await REDIS.close()
            logger.info("Redis closed")
    except Exception as e:
        logger.error(f"Error closing Redis: {e}")

# ----------------- Models -----------------
class SimRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra='ignore')

    # core
    symbol: str
    mode: Literal["quick", "deep"] = Field(
        default="quick",
        description="quick=~6m lookback; deep=up to ~10y"
    )
    horizon_days: int = Field(default=30, ge=1, le=3650)

    # paths (keep alias for legacy payloads using n_paths)
    paths: int = Field(2000, alias="n_paths", ge=100, le=200_000)
    timespan: Literal["day", "hour", "minute"] = "day"

    # optional modules
    include_news: bool = False
    include_options: bool = False
    include_futures: bool = False

    seed: Optional[int] = None

    @property
    def n_paths(self) -> int:  # legacy compatibility
        return self.paths

    def lookback_days(self) -> int:
        return 180 if self.mode == "quick" else 3650

    def bars_per_day(self) -> int:
        return 1 if self.timespan == "day" else (24 if self.timespan == "hour" else 390)

    def young_threshold_bars(self) -> int:
        return 126 * self.bars_per_day()

class TrainRequest(BaseModel):
    symbol: str
    lookback_days: int = Field(default=365, ge=30, le=3650)

class PredictRequest(BaseModel):
    symbol: str
    horizon_days: int = Field(default=30, ge=1, le=365)
    use_online: bool = True

class RunState(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    status: Literal["queued", "running", "done", "error"] = "queued"
    progress: float = 0.0
    symbol: str | None = None
    horizon_days: int | None = None
    paths: int | None = None
    startedAt: str | None = None
    finishedAt: str | None = None
    error: str | None = None

# ----------------- Utilities -----------------
async def _list_models() -> List[str]:
    if not REDIS:
        return []
    out: List[str] = []
    try:
        async for k in REDIS.scan_iter(match="model:*", count=500):
            out.append(k.split(":", 1)[1] if ":" in k else k)
    except Exception as e:
        logger.info(f"_list_models scan failed: {e}")
    return out


async def _fetch_hist_prices(symbol: str, window_days: int | None = None) -> List[float]:
    symbol = (symbol or "").upper().strip()
    key = _poly_key()
    if not key:
        raise HTTPException(status_code=400, detail="Polygon key missing")

    wd = settings.lookback_days_max if window_days is None else max(1, int(window_days))
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=wd)

    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": "50000"}
    headers = {"Authorization": f"Bearer {key}"}

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            r = await client.get(url, params=params, headers=headers)
            r.raise_for_status()
            data = r.json() or {}
            results = data.get("results") or []
            closes = [float(x.get("c")) for x in results if isinstance(x, dict) and "c" in x]
            return closes
    except httpx.HTTPStatusError as e:
        if e.response is not None and e.response.status_code == 429:
            raise HTTPException(status_code=429, detail="Rate limited by Polygon")
        logger.exception(f"Polygon aggs fetch failed for {symbol}: {e}")
        return []
    except Exception as e:
        logger.exception(f"Failed to fetch prices for {symbol}: {e}")
        return []

async def _feat_from_prices(symbol: str, px: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in px if isinstance(v, (int, float)) and math.isfinite(v)], dtype=float)
    out: Dict[str, float] = {"mom_20": 0.0, "rvol_20": 0.0, "autocorr_5": 0.0}
    if arr.size < 3:
        return out

    base = arr[-21] if arr.size > 20 else arr[0]
    out["mom_20"] = float(arr[-1] / base - 1.0) if base > 0 else 0.0

    rets = np.diff(arr) / arr[:-1]
    if rets.size == 0:
        return out

    win_s = min(20, rets.size)
    win_l = min(60, rets.size)
    rv_s = float(np.sqrt(np.mean(np.square(rets[-win_s:])))) if win_s > 0 else 0.0
    rv_l = float(np.sqrt(np.mean(np.square(rets[-win_l:])))) if win_l > 0 else 0.0
    out["rvol_20"] = float(rv_s / rv_l) if rv_l > 1e-12 else 0.0

    w = min(5, rets.size - 1) if rets.size > 1 else 0
    if w >= 2:
        x = rets[-w-1:-1]
        y = rets[-w:]
        sx, sy = np.std(x), np.std(y)
        if sx > 0 and sy > 0:
            out["autocorr_5"] = float(np.corrcoef(x, y)[0, 1])
        else:
            out["autocorr_5"] = 0.0

    return out

async def _fetch_cached_hist_prices(symbol: str, window_days: int, redis: Redis | None) -> List[float]:
    today_str = _today_utc_date().isoformat()

    def _to_app_symbol(s: str) -> str:
        s = (s or "").strip().upper()
        if s.startswith("X:") and s.endswith("USD"):
            return f"{s[2:-3]}-USD"
        if s.endswith("USD") and "-" not in s and not s.startswith("X:"):
            return f"{s[:-3]}-USD"
        return s

    app_sym = _to_app_symbol(symbol)
    cache_key = f"hist_prices:{app_sym}:{window_days}:{today_str}"

    if redis:
        cached = await redis.get(cache_key)
        if cached:
            try:
                js = json.loads(cached)
                if isinstance(js, list) and js and all(isinstance(x, (int, float)) for x in js):
                    return js[-int(window_days):]
            except Exception:
                pass

    closes: List[float] = []
    try:
        con = fs_connect()
        rows = con.execute(
            "SELECT ts, close FROM bars_daily WHERE symbol = ? ORDER BY ts ASC",
            [app_sym]
        ).fetchall()
        con.close()
        if rows:
            closes = [float(r[1]) for r in rows][-int(window_days):]
    except Exception as e:
        logger.info(f"DuckDB bars_daily lookup failed for {app_sym}: {e}")

    free_mode = os.getenv("PT_FREE_MODE","0") == "1"
    if not closes or len(closes) < min(30, int(window_days) // 2 or 1):
        if not free_mode:
            closes = await _fetch_hist_prices(symbol, window_days)
    if redis:
        try:
            await redis.setex(cache_key, 1800, json.dumps(closes))
        except Exception:
            pass
    return closes


def _dynamic_window_days(horizon_days: int, timespan: str) -> int:
    try:
        h = max(0, int(horizon_days))
    except Exception:
        h = 0
    base_map = {
        "day": 180,
        "hour": int(round(180 * 7)),             # ~7x more bars
        "minute": int(round(180 * (390 / 252))), # ~279 trading days of minutes
    }
    base_days = base_map.get((timespan or "day").lower(), 180)
    return min(base_days + h * 2, settings.lookback_days_max)

async def _ensure_run(run_id: str) -> "RunState":
    r = _ensure_redis()
    key = f"run:{run_id}"
    raw = await r.get(key)
    if not raw:
        # double-check TTL to distinguish never vs expired
        ttl = await r.ttl(key)
        if ttl == -2:
            raise HTTPException(status_code=410, detail="Run expired")
        raise HTTPException(status_code=404, detail="Run not found")

    # Accept bytes or str
    txt = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
    try:
        rs = RunState.model_validate_json(txt)
    except Exception:
        raise HTTPException(status_code=500, detail="Corrupt run state")

    return rs

async def _labeling_daemon():
    while True:
        try:
            # TODO: hook your labeler here (e.g., await label_recent_runs(...))
            pass
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"Labeling pass failed: {e}")
        await asyncio.sleep(900)

# --------- Optional model loaders (only if missing up top) ----------
if "load_lstm_model" not in globals():
    def load_lstm_model(symbol: str):
        require_tf()
        for path in (f"models/{symbol}_lstm.keras", f"models/{symbol}_lstm.h5"):
            try:
                # use the alias if available
                if "_tf_load_model" in globals() and _tf_load_model:
                    return _tf_load_model(path)
                from tensorflow.keras.models import load_model as _lm  # late import fallback
                return _lm(path)
            except Exception:
                continue
        raise HTTPException(status_code=404, detail="LSTM model not found; train first")

if "load_arima_model" not in globals():
    def load_arima_model(symbol: str):
        try:
            with open(f"models/{symbol}_arima.pkl", "rb") as file:
                return pickle.load(file)
        except Exception:
            raise HTTPException(status_code=404, detail="ARIMA model not found; train first")

# --------- Ensemble (linear + optional lstm/arima + optional RL) ----------
if "get_ensemble_prob_light" not in globals():
    async def get_ensemble_prob_light(symbol: str, redis: "Redis", horizon_days: int = 1) -> float:
        try:
            # support either async or sync _model_key
            try:
                model_key_name = await _model_key(symbol + "_linear")
            except TypeError:
                model_key_name = _model_key(symbol + "_linear")

            raw = await redis.get(model_key_name)
            if not raw:
                return 0.5

            model_linear = json.loads(raw)
            feats = list(model_linear.get("features", []))
            coef  = list(model_linear.get("coef", []))

            px = await _fetch_hist_prices(symbol)
            if not px or len(px) < 10:
                return 0.5

            f = await _feat_from_prices(symbol, px)
            X = np.array([float(f.get(feat, 0.0)) for feat in feats], dtype=float)
            w = np.array([float(c) for c in coef], dtype=float)

            m = int(min(X.shape[0], w.shape[0]))
            if m <= 0:
                return 0.5

            score = float(np.dot(X[:m], w[:m]))
            return _sigmoid(float(np.clip(score, -60.0, 60.0)))
        except Exception as e:
            logger.info(f"get_ensemble_prob_light fallback (0.5) due to: {e}")
            return 0.5

# --- Ensemble utilities (drop-in) --------------------------------------------
# Replaces: get_ensemble_prob(...) and _meta_weights(...)

from typing import Dict

async def get_ensemble_prob(symbol: str, redis: 'Redis', horizon_days: int = 1) -> float:
    """
    Blend linear/LSTM/ARIMA (and an optional RL shim) into a single up-probability.
    Safe on Render: handles Redis bytes, missing models, and degenerate weights.
    """
    try:
        # Resolve model key (works whether _model_key is async or not)
        try:
            model_key_name = await _model_key(symbol + "_linear")
        except TypeError:
            model_key_name = _model_key(symbol + "_linear")

        # Fetch linear model from Redis (bytes -> str -> json)
        raw = await redis.get(model_key_name)
        if not raw:
            return 0.5
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        model_linear = json.loads(raw)

        # Features (fallbacks if none present)
        feature_list = list(model_linear.get("features", ["mom_20", "rvol_20", "autocorr_5"]))

        # Build X from recent prices → features
        px = await _fetch_hist_prices(symbol)
        if not px or len(px) < 10:
            return 0.5
        f = await _feat_from_prices(symbol, px)
        X = np.array([float(f.get(feat, 0.0)) for feat in feature_list], dtype=float)

        preds: Dict[str, float] = {}

        # --- Linear head (supports intercept if present)
        if "coef" in model_linear:
            w = np.array([float(c) for c in model_linear["coef"]], dtype=float)
            xb = np.concatenate([[1.0], X])  # intercept term
            k = min(w.shape[0], xb.shape[0])
            score = float(np.dot(w[:k], xb[:k]))
            preds["linear"] = _sigmoid(float(np.clip(score, -60.0, 60.0)))

        # --- LSTM (optional)
        try:
            model_lstm = load_lstm_model(symbol)  # you already guard import elsewhere
            X_lstm = np.expand_dims(np.expand_dims(X, axis=0), axis=0)
            lstm_p = float(model_lstm.predict(X_lstm, verbose=0)[0][0])
            preds["lstm"] = float(np.clip(lstm_p, 0.0, 1.0))
        except Exception:
            # logger.debug("LSTM unavailable; skipping", exc_info=True)
            pass

        # --- ARIMA (optional)
        try:
            model_arima = load_arima_model(symbol)
            fc = model_arima.forecast(steps=max(1, int(horizon_days)))
            last_fc = float(fc.iloc[-1] if hasattr(fc, "iloc") else fc[-1])
            preds["arima"] = 1.0 if last_fc > float(px[-1]) else 0.0
        except Exception:
            # logger.debug("ARIMA unavailable; skipping", exc_info=True)
            pass

        # --- RL shim (optional) → currently a no-op adjustment
        rl_adjust = 0.0
        try:
            from stable_baselines3 import DQN  # type: ignore
            _ = DQN  # imported OK; load if/when env is wired
            # rl_model = DQN.load(f"models/{symbol}_rl.zip", print_system_info=False)
            # rl_adjust += some_small_bias
        except Exception as e:
            logger.info(f"RL skipped in ensemble: {e}")

        if not preds:
            return 0.5

        # Weights with recent error nudging
        mw = _meta_weights(symbol, int(horizon_days))

        # Keep order stable and only include heads we actually have
        ordered = [n for n in ("linear", "lstm", "arima") if n in preds]
        comps = np.array([preds[n] for n in ordered], dtype=float)
        wts = np.array([mw.get(n, 0.25) for n in ordered], dtype=float)

        # Normalize or fall back to uniform if degenerate
        s = float(wts.sum())
        if (not np.isfinite(s)) or s <= 0:
            wts = np.full_like(wts, 1.0 / max(1, wts.shape[0]), dtype=float)
        else:
            wts = wts / s

        prob = float(np.clip(float(np.dot(wts, comps)) + rl_adjust, 0.0, 1.0))
        return prob

    except Exception:
        logger.exception("get_ensemble_prob failed; returning 0.5 fallback")
        return 0.5


def _meta_weights(symbol: str, horizon_days: int) -> dict:
    """
    Heuristic ensemble weights nudged by recent MDAPE.
    Falls back cleanly if feature store/metric lookup fails.
    """
    try:
        con = fs_connect()
        mdape = get_recent_mdape(con, symbol, horizon_days, lookback_days=30)
        try:
            con.close()
        except Exception:
            pass
    except Exception:
        mdape = float("nan")

    # Base weights; 'rl' is kept for future expansion (currently 0 in blend)
    base = {"linear": 0.28, "lstm": 0.28, "arima": 0.28, "rl": 0.16}

    if not np.isfinite(mdape):
        return base

    # If recent error was high, give linear a bit more; lightly damp others.
    mult = float(np.clip(1.3 - 0.03 * (mdape - 2.0), 0.7, 1.3))
    w_lin = base["linear"] * (1.0 if mdape > 8.0 else mult)
    w_lstm = base["lstm"] * 0.95
    w_arima = base["arima"] * 0.90
    w_rl = base["rl"] * (1.05 if mdape > 10.0 else 0.95)

    ws = np.array([w_lin, w_lstm, w_arima, w_rl], dtype=float)
    s = float(ws.sum()) or 1.0
    ws = ws / s

    return {
        "linear": float(ws[0]),
        "lstm": float(ws[1]),
        "arima": float(ws[2]),
        "rl": float(ws[3]),
    }
# --- end ensemble utilities ---------------------------------------------------


def _calibration_sigma_scale(symbol: str, horizon_d: int) -> float:
    try:
        con = fs_connect()
        cov, _n = get_recent_coverage(con, symbol, horizon_d, lookback_days=21)
        con.close()
        if not np.isfinite(cov): return 1.0
        target = 0.90
        err = float(np.clip(target - cov, -0.30, 0.30))
        scale = float(np.clip(1.0 + 0.7 * err, 0.85, 1.15))
        return scale
    except Exception:
        return 1.0

def _detect_regime(px: np.ndarray) -> dict:
    arr = np.asarray(px, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if arr.size < 40:
        return {"name": "neutral", "score": 0.0}

    rets = np.diff(np.log(arr))
    if rets.size == 0:
        return {"name": "neutral", "score": 0.0}

    rv20  = float(np.std(rets[-20:]) * math.sqrt(252)) if rets.size >= 20 else float(np.std(rets) * math.sqrt(252))
    mom20 = float((arr[-1] / arr[max(0, arr.size - 21)]) - 1.0)

    v = float(np.clip((rv20 - 0.30) / 0.30, -1.5, 1.5))
    m = float(np.clip(mom20 / 0.10, -1.5, 1.5))

    if v > 0.6 and m < -0.3:   name, score = "vol-shock", -0.7
    elif m > 0.4 and v < 0.3:  name, score = "bull-trend", 0.6
    elif m < -0.4 and v < 0.3: name, score = "bear-trend", -0.5
    else:                      name, score = "neutral", float(np.clip(m - 0.3 * v, -0.4, 0.4))

    return {"name": name, "score": float(np.clip(score, -1.0, 1.0))}

# ----------------- Simulation worker -----------------
async def run_simulation(run_id: str, req: "SimRequest", redis: Redis):
    logger.info(f"Starting simulation for run_id={run_id}, symbol={req.symbol}")
    try:
        rs = await _ensure_run(run_id)
    except Exception as e:
        logger.error(f"_ensure_run failed at start for {run_id}: {e}")
        rs = RunState(status="error", progress=0.0, error=str(e))
        await redis.setex(f"run:{run_id}", settings.run_ttl_seconds, rs.model_dump_json())
        return

    rs.status = "running"; rs.progress = 0.0
    await redis.setex(f"run:{run_id}", settings.run_ttl_seconds, rs.model_dump_json())

    try:
        # ---------- Guardrails ----------
        if req.paths * req.horizon_days > settings.pathday_budget_max:
            raise ValueError("compute budget exceeded")
        if req.horizon_days > settings.horizon_days_max or req.paths > settings.n_paths_max:
            raise ValueError("input limits exceeded")

        # ---------- Determine lookback window ----------
        mode = getattr(req, "mode", "quick")  # "quick" | "deep" (default quick)
        if req.timespan == "day":
            # quick: ~6m history; deep: up to settings cap (~10y)
            window_days = 180 if mode == "quick" else int(min(settings.lookback_days_max, 3650))
        else:
            window_days = _dynamic_window_days(req.horizon_days, req.timespan)

        # ---------- Fetch prices ----------
        historical_prices = await _fetch_cached_hist_prices(req.symbol, window_days, redis)
        if not historical_prices or len(historical_prices) < 2:
            raise ValueError("Insufficient history")

        px_arr = np.array(historical_prices, dtype=float)
        S0 = float(px_arr[-1])

        # ---------- Returns & annualization ----------
        bpd = _bars_per_day(req.timespan)             # bars per day
        scale = 252 * bpd                             # bars per year (annualization)
        rets_all = np.diff(np.log(px_arr))            # full history for bootstrap/statistics

        # ---- μ / σ estimation on a decoupled recent window ----
        stat_bars = (STAT_BARS_QUICK if mode == "quick" else STAT_BARS_DEEP) * bpd
        px_stats = px_arr[-(stat_bars + 1):] if px_arr.size > (stat_bars + 1) else px_arr
        rets_est = np.diff(np.log(px_stats))
        rets_est = _winsorize(rets_est)

        # EWMA σ (per-bar) → annualized
        sigma_bar = _ewma_sigma(rets_est, lam=0.94)
        sigma_ann_raw = float(sigma_bar * math.sqrt(scale))
        sigma_ann = float(np.clip(sigma_ann_raw, 1e-4, SIGMA_CAP))

        # μ from winsorized sample mean (annualized), with caps
        mu_ann_raw = float(np.mean(rets_est) * scale) if rets_est.size else 0.0
        mu_cap = MU_CAP_DEEP if mode == "deep" else MU_CAP_QUICK
        mu_ann = float(np.clip(mu_ann_raw, -mu_cap, mu_cap))

        # Horizon-based drift shrink
        shrink = _horizon_shrink(int(req.horizon_days))
        mu_ann *= shrink

        # Optional IV-anchoring for near-term (if options enabled and IV available)
        iv30 = _iv30_or_none(req.symbol) if getattr(req, "include_options", False) else None
        if iv30 and isinstance(iv30, (float, int)) and iv30 > 0:
            sigma_ann = float(max(sigma_ann, float(iv30)))

        # ---------- Warnings ----------
        warnings: list[str] = []
        if px_arr.size < 126 * bpd:  # ~6 months of bars
            warnings.append(f"{req.symbol} ticker is too young, simulation may be inaccurate due to not enough historical information")
        if shrink < 0.999:
            warnings.append(f"Applied long-horizon drift shrink ({shrink:.2f}×) to improve long-run realism")
        if abs(mu_ann_raw) > mu_cap + 1e-6:
            warnings.append(f"Drift capped at ±{int(mu_cap*100)}%/yr for stability (measured {mu_ann_raw*100:.1f}%/yr)")
        if sigma_ann_raw > SIGMA_CAP + 1e-6:
            warnings.append(f"Volatility capped at {int(SIGMA_CAP*100)}%/yr for stability (measured {sigma_ann_raw*100:.1f}%/yr)")

        # ---------- Regime-aware nudges (before sim) ----------
        try:
            reg = _detect_regime(px_arr)
        except Exception:
            reg = {"name": "unknown", "score": 0.0}

        mu_adj = mu_ann
        sigma_adj = sigma_ann

        if reg["name"] == "vol-shock":
            sigma_adj = float(np.clip(sigma_adj * 1.15, 1e-4, 1.8))
            mu_adj    = float(np.clip(mu_adj - 0.10 * sigma_adj, -3.0, 3.0))
        elif reg["name"] == "bull-trend":
            mu_adj = float(np.clip(mu_adj + 0.08 * sigma_adj, -3.0, 3.0))
        elif reg["name"] == "bear-trend":
            mu_adj = float(np.clip(mu_adj - 0.08 * sigma_adj, -3.0, 3.0))

        # ---------- ML drift tilt (fast) ----------
        try:
            ensemble_prob = await asyncio.wait_for(get_ensemble_prob(req.symbol, redis, req.horizon_days), timeout=1.0)
        except Exception:
            try:
                ensemble_prob = await asyncio.wait_for(get_ensemble_prob_light(req.symbol, redis, req.horizon_days), timeout=0.5)
            except Exception:
                ensemble_prob = 0.5
        conf = (2.0 * float(ensemble_prob) - 1.0)
        mu_adj = float(mu_adj + (0.30 * conf * sigma_adj))

        # ---------- Optional sentiment/options/futures tweaks ----------
        sentiment = 0.0
        if getattr(req, "include_news", False):
            try:
                poly_ticker = req.symbol.upper().strip()
                key = _poly_key()
                since = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")
                url = "https://api.polygon.io/v2/reference/news"
                params = {"ticker": poly_ticker, "published_utc.gte": since, "limit": "20", "apiKey": key}
                async with httpx.AsyncClient(timeout=httpx.Timeout(8.0)) as client:
                    r = await client.get(url, params=params)
                    if r.status_code == 200:
                        items = (r.json() or {}).get("results", []) or []
                        if items:
                            ss = [_safe_sent((it.get("title") or "")) for it in items]
                            if ss:
                                sent_mean = float(np.mean(ss))
                                sent_std  = float(np.std(ss))
                                sem_intensity = float(np.clip(abs(sent_mean) * (1.0 + 0.5 * (1.0 - np.tanh(sent_std))), 0.0, 0.15))
                                semantic_tilt = float(np.clip(np.sign(sent_mean) * sem_intensity, -0.03, 0.03))
                                sentiment = float(np.clip(sentiment + semantic_tilt, -0.07, 0.07))
            except Exception as e:
                logger.info(f"news sentiment failed: {e}")

        if getattr(req, "include_options", False):
            sigma_adj = float(np.clip(sigma_adj * 1.05, 1e-4, 1.5))
        if getattr(req, "include_futures", False):
            mu_adj += 0.001

        mu_adj = float(np.clip(mu_adj + sentiment, -3.0, 3.0))

        # Final small calibration
        sigma_scale = _calibration_sigma_scale(req.symbol, int(req.horizon_days))
        sigma_adj = float(np.clip(sigma_adj * sigma_scale, 1e-4, 2.0))

        # ---------- Seed (stable per-day) ----------
        import zlib
        model_id_hint = "gbm_t7" if mode == "quick" else "bootstrap_b15"
        utc_day = datetime.utcnow().strftime("%Y-%m-%d")
        try:
            seed_key = f"{req.symbol.upper()}|{int(req.horizon_days)}|{model_id_hint}|{utc_day}"
            seed = zlib.adler32(seed_key.encode("utf-8")) & 0xFFFFFFFF  # deterministic across processes
        except Exception:
            seed = abs(zlib.adler32(str(run_id).encode("utf-8"))) & 0xFFFFFFFF
        seed = int(req.seed) if getattr(req, "seed", None) is not None else int(seed)

        # ---------- Simulate ----------
        H  = int(req.horizon_days)
        N  = int(req.n_paths)
        engine_used = ""
        paths_mat: np.ndarray

        if mode == "deep":
            block_len = 15
            min_blocks = 6
            if rets_all.size >= block_len * min_blocks:
                paths_mat = simulate_bootstrap_blocks(
                    S0=S0,
                    log_rets_hist=rets_all,
                    horizon_days=H,
                    n_paths=N,
                    block_len=block_len,
                    antithetic=True,
                    seed=seed,
                )
                engine_used = f"bootstrap_b{block_len}"

                # Inject drift tilt into bootstrap by multiplying by exp(μ_d * t)
                mu_d = mu_adj / 252.0
                if abs(mu_d) > 1e-12:
                    t_idx = np.arange(H + 1, dtype=float)[None, :]  # shape (1, H+1)
                    tilt = np.exp(mu_d * t_idx)                     # broadcast to (N, H+1)
                    paths_mat = paths_mat * tilt
            else:
                warnings.append("Insufficient history for bootstrap; fell back to GBM+t noise")

        if engine_used == "":
            paths_mat = simulate_gbm_student_t(
                S0=S0,
                mu_ann=mu_adj,
                sigma_ann=sigma_adj,
                horizon_days=H,
                n_paths=N,
                df_t=7,
                antithetic=True,
                seed=seed,
            )
            engine_used = "gbm_t7"

        rs.progress = 40.0
        await redis.setex(f"run:{run_id}", settings.run_ttl_seconds, rs.model_dump_json())

        horizon_days_ui = int(H) if bpd == 1 else int(math.ceil(H / float(bpd)))

        targets_block = compute_targets_block(
            paths=paths_mat,
            S0=S0,
            horizon_days=horizon_days_ui,
            bars_per_day=bpd,
            # Optional: customize rungs
            rel_levels=(-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40),
        )

        # ---------- Bands / summary from paths_mat ----------
        paths = paths_mat  # alias
        p50_line = np.median(paths, axis=0)
        p80_low, p80_high = np.percentile(paths, [10, 90], axis=0)
        p95_low, p95_high = np.percentile(paths, [2.5, 97.5], axis=0)
        # add 5th/95th only for EOD readout accuracy
        p05_line, p95_line = np.percentile(paths, [5, 95], axis=0)

        def _ffill_nonfinite(arr: np.ndarray, fallback: float) -> np.ndarray:
            out = np.array(arr, dtype=float)
            if not np.isfinite(out[0]): out[0] = float(fallback)
            for i in range(1, len(out)):
                if not np.isfinite(out[i]): out[i] = out[i - 1]
            return out

        fallback = S0
        for arr in (p50_line, p80_low, p80_high, p95_low, p95_high, p05_line, p95_line):
            np.nan_to_num(arr, copy=False, nan=fallback, posinf=fallback, neginf=fallback)
            arr[:] = _ffill_nonfinite(arr, fallback)

        # ---------- Terminal metrics ----------
        terminal = paths[:, -1].astype(float)
        prob_up  = float(np.mean(terminal > S0))

        # VaR / ES on returns
        ret_terminal = (terminal - S0) / S0
        var95 = float(np.percentile(ret_terminal, 5))
        es_mask = ret_terminal <= var95
        es95 = float(ret_terminal[es_mask].mean()) if es_mask.any() else float(var95)

        # Quantiles (prices)
        term_q05 = float(np.percentile(terminal, 5))
        term_q50 = float(np.percentile(terminal, 50))
        term_q95 = float(np.percentile(terminal, 95))

        # EOD (next-bar) summary
        T = paths.shape[1]
        eod_idx = 1 if T > 1 else 0
        eod_mean = float(paths[:, eod_idx].mean())
        eod_med  = float(p50_line[eod_idx])
        eod_p05  = float(p05_line[eod_idx])
        eod_p95  = float(p95_line[eod_idx])
        prob_up_next = float(np.mean(paths[:, eod_idx] > S0)) if eod_idx > 0 else None

        valid = terminal[terminal > 0]
        if valid.size >= 10:
            logT = np.log(valid)
            # Freedman–Diaconis bin width in log-space
            iqr = float(np.subtract(*np.percentile(logT, [75, 25])))
            bw = 2.0 * iqr * (logT.size ** (-1.0/3.0))
            if not np.isfinite(bw) or bw <= 1e-9:
                # fallback to Silverman's-like scaling on logs
                std = float(np.std(logT))
                bw = max(1e-6, std * (logT.size ** (-1.0/5.0)) if std > 0 else 1e-3)
            # clamp bins to a reasonable range
            bins = int(np.clip(np.ceil((logT.max() - logT.min()) / max(1e-9, bw)), 30, 120))
            counts, edges = np.histogram(logT, bins=bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            mode_price = float(np.exp(centers[np.argmax(counts)]))
        else:
            # Degenerate case: fall back to median terminal price
            mode_price = float(term_q50)

        # ---------- 10% Highest Density Interval (HDI10) ----------
        sorted_t = np.sort(valid) if valid.size else np.array([], dtype=float)
        if sorted_t.size >= 10:
            k = max(1, int(math.ceil(0.10 * sorted_t.size)))
            widths = sorted_t[k-1:] - sorted_t[:sorted_t.size - k + 1]
            j = int(np.argmin(widths))
            hdi10_low, hdi10_high = float(sorted_t[j]), float(sorted_t[j + k - 1])
        else:
            hdi10_low, hdi10_high = float(term_q50), float(term_q50)

        # ---------- Hit probabilities ----------
        thresholds_pct = np.array([-0.05, 0.00, 0.05, 0.10], dtype=float)
        thresholds_abs = (1.0 + thresholds_pct) * float(S0)
        probs_by_day = ((paths[:, :, None] >= thresholds_abs[None, None, :]).mean(axis=0).T).tolist()

        rs.progress = 80.0
        await redis.setex(f"run:{run_id}", settings.run_ttl_seconds, rs.model_dump_json())

        # ---------- Artifact ----------
        artifact = {
            "symbol": req.symbol,
            "horizon_days": int(req.horizon_days),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "spot": float(S0),
            "targets": targets_block,
            "median_path": [[i, float(v)] for i, v in enumerate(p50_line.tolist())],
            "bands": {
                "p50":      [[i, float(v)] for i, v in enumerate(p50_line.tolist())],
                "p80_low":  [[i, float(v)] for i, v in enumerate(p80_low.tolist())],
                "p80_high": [[i, float(v)] for i, v in enumerate(p80_high.tolist())],
                "p95_low":  [[i, float(v)] for i, v in enumerate(p95_low.tolist())],
                "p95_high": [[i, float(v)] for i, v in enumerate(p95_high.tolist())],
            },

            "prob_up_end": float(prob_up),
            "prob_up_next": prob_up_next,
            "var_es": {"var95": float(var95), "es95": float(es95)},
            "terminal_prices": [float(x) for x in terminal.tolist()],

            "hit_probs": {
                "thresholds_abs": [float(x) for x in thresholds_abs.tolist()],
                "probs_by_day": probs_by_day,
            },

            "most_likely_price": float(mode_price),
            "hdi10": {"low": float(hdi10_low), "high": float(hdi10_high)},

            "eod_estimate": {
                "day_index": int(eod_idx),
                "median": float(eod_med),
                "mean": float(eod_mean),
                "p05": float(eod_p05),
                "p95": float(eod_p95),
            },

            "model_info": {
                "engine": engine_used,
                "direction": "MonteCarlo",
                "regime": reg.get("name", "unknown"),
                "regime_score": float(reg.get("score", 0.0)),
                "seed_hint": int(seed),
                "timescale": req.timespan,
            },

            "warnings": list(dict.fromkeys(warnings)),

            "inputs": {
                "S0": float(S0),
                "paths": int(req.paths),
                "horizon_days": int(req.horizon_days),
                "timescale": req.timespan,
                "seed": int(seed),
                "mode": mode,
            },

            "calibration": {
                "window": int(window_days),
                "sigma_scale": float(sigma_scale),
                "mu_annualized": float(mu_adj),
                "sigma_annualized": float(sigma_adj),
                "stat_window_bars": int(stat_bars),
            },
        }

        ttl = int(getattr(settings, "run_ttl_seconds", 7 * 24 * 3600))
        await redis.setex(f"artifact:{run_id}", ttl, json.dumps(artifact))

        # ===== Auto-summarize (fast path + background fallback) ======================
        try:
            try:
                await asyncio.wait_for(_summarize_run(run_id), timeout=3.0)
            except asyncio.TimeoutError:
                asyncio.create_task(_summarize_run(run_id))
        except Exception as e:
            logger.info(f"autosummarize skipped: {e}")

        # ---------- Persist mirrors (best-effort) ----------
        try:
            pred_id = str(uuid4())
            insert_prediction({
                "pred_id": pred_id,
                "ts": datetime.utcnow(),
                "symbol": req.symbol.upper(),
                "horizon_d": int(req.horizon_days),
                "model_id": artifact["model_info"]["engine"],
                "prob_up_next": (float(prob_up_next) if prob_up_next is not None else None),
                "p05": float(term_q05),
                "p50": float(term_q50),
                "p95": float(term_q95),
                "spot0": float(S0),
                "user_ctx": {"ui": "simetrix", "run_id": run_id, "n_paths": int(req.paths)},
                "run_id": run_id,
            })
        except Exception as e:
            logger.warning(f"DuckDB insert_prediction (simulate) failed: {e}")

        try:
            con = fs_connect()
            fs_log_prediction(con, {
                "run_id":       run_id,
                "model_id":     artifact["model_info"]["engine"],
                "symbol":       req.symbol.upper(),
                "issued_at":    datetime.now(timezone.utc).isoformat(),
                "horizon_days": int(req.horizon_days),
                "yhat_mean":    float(term_q50),
                "prob_up":      float(prob_up),
                "q05":          float(term_q05),
                "q50":          float(term_q50),
                "q95":          float(term_q95),
                "uncertainty":  float(np.std(terminal)),
                "features_ref": {
                    "window_days":  int(window_days),
                    "paths":        int(req.paths),
                    "S0":           float(S0),
                    "mu_ann":       float(mu_adj),
                    "sigma_ann":    float(sigma_adj),
                    "timespan":     req.timespan,
                    "seed_hint":    int(seed),
                    "mode":         mode,
                },
            })
            con.close()
        except Exception as e:
            logger.warning(f"Feature Store mirror failed: {e}")

        # ---------- Finish ----------
        rs.status = "done"; rs.progress = 100.0
        await redis.setex(f"run:{run_id}", settings.run_ttl_seconds, rs.model_dump_json())
        logger.info(f"Completed simulation for run_id={run_id}")

    except Exception as e:
        logger.exception(f"Simulation failed for run_id={run_id}: {e}")
        rs.status = "error"; rs.error = str(e)
        await redis.setex(f"run:{run_id}", settings.run_ttl_seconds, rs.model_dump_json())

# ----------------- Simulation routes -----------------
@app.post("/simulate")
async def simulate(req: SimRequest, _ok: bool = Depends(maybe_require_key)):
    """
    Start a simulation run (auth via app-level key; supports open_access toggle).
    Frontend should pass req.mode = "quick" (≈6m) or "deep" (≈10y).
    """
    # normalize symbol early so logs/keys are consistent
    try:
        req.symbol = (req.symbol or "").upper().strip()
    except Exception:
        # if Pydantic model is frozen in your setup, shadow it
        req = req.model_copy(update={"symbol": (req.symbol or "").upper().strip()})

    run_id = uuid4().hex  # hex keeps keys simple (no dashes)
    rs = RunState(
        run_id=run_id,
        status="queued",
        progress=0.0,
        symbol=req.symbol,
        horizon_days=int(req.horizon_days),
        paths=int(req.paths),
        startedAt=datetime.now(timezone.utc).isoformat(),
    )

    # use config TTL (not a hard-coded 3600)
    await REDIS.setex(f"run:{run_id}", settings.run_ttl_seconds, rs.model_dump_json())
    asyncio.create_task(run_simulation(run_id, req, REDIS))
    return {"run_id": run_id}

@app.get("/simulate/{run_id}/state")
async def get_sim_state(run_id: str, _ok: bool = Depends(require_key)):
    rs = await _ensure_run(run_id)
    return rs.model_dump()

@app.get("/simulate/{run_id}/status")
async def simulate_status(run_id: str, _ok: bool = Depends(require_key)):
    rs = await _ensure_run(run_id)
    return {"status": rs.status, "progress": rs.progress, "error": rs.error}

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
@app.get("/simulate/{run_id}/artifact")
async def get_sim_artifact(run_id: str, _ok: bool = Depends(require_key)):
    raw = await REDIS.get(f"artifact:{run_id}")
    if not raw:
        rs = await _ensure_run(run_id)
        raise HTTPException(status_code=202, detail=f"Run {run_id} status={rs.status}; artifact not ready")
    try:
        payload = json.loads(raw)
    except Exception:
        payload = {"artifact": (raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw))}
    try:
        if isinstance(payload, dict) and not payload.get("summary"):
            asyncio.create_task(_summarize_run(run_id))
    except Exception:
        pass
    return JSONResponse(content=payload)
@app.get("/simulate/{run_id}/stream")
async def simulate_stream(run_id: str, _ok: bool = Depends(require_key)):  # was verify_api_key
    async def event_generator():
        stale_ticks = 0
        last = None
        while True:
            try:
                if not REDIS:
                    yield 'data: {"status":"error","progress":0,"detail":"redis_unavailable"}\n\n'
                    break

                raw = await REDIS.get(f"run:{run_id}")
                if not raw:
                    yield 'data: {"status":"error","progress":0,"detail":"run_not_found_or_expired"}\n\n'
                    break

                txt = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                rs = RunState.model_validate_json(txt)

                payload = {"status": rs.status, "progress": float(rs.progress or 0.0)}
                yield f"data: {json.dumps(payload)}\n\n"

                if rs.status in ("done", "error"):
                    break

                sig = (rs.status, round(float(rs.progress or 0.0), 2))  # <- fix stall check
                if sig == last:
                    stale_ticks += 1
                else:
                    stale_ticks = 0
                    last = sig

                if stale_ticks > 120:  # ~2 minutes
                    yield 'data: {"status":"error","progress":0,"detail":"stalled"}\n\n'
                    break

                await asyncio.sleep(1.0)

            except Exception as e:
                yield f'data: {json.dumps({"status":"error","progress":0,"detail":str(e)[:200]})}\n\n'
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control":"no-cache","Connection":"keep-alive","X-Accel-Buffering":"no"},
    )


# ===== Bulk EOD ingest to DuckDB (stocks + crypto) ============================
def _poly_to_app_symbol(s: str) -> str:
    # "X:BTCUSD" -> "BTC-USD"; "BTCUSD" -> "BTC-USD"; equities unchanged
    s = (s or "").strip().upper()
    if not s:
        return s
    if s.startswith("X:") and s.endswith("USD"):
        return f"{s[2:-3]}-USD"
    if s.endswith("USD") and "-" not in s and not s.startswith("X:"):
        return f"{s[:-3]}-USD"
    return s

async def llm_shortlist(kind: str, symbols: list[str], top_k: int = 20) -> list[str]:
    key = os.getenv("XAI_API_KEY", "").strip()
    base = [s.upper() for s in symbols]
    if not base:
        return []
    top_k = max(1, min(int(top_k), len(base)))

    if not key:
        return base[:top_k]  # fallback

    prompt = {
        "role": "user",
        "content": (
            "Given this watchlist, pick the TOP symbols to review today based on likely near-term catalysts, "
            "liquidity, momentum, and risk. Return JSON {list:[...]} with up to "
            f"{top_k} tickers only. Watchlist ({kind}): {', '.join(base)}"
        ),
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as cli:
            r = await cli.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": os.getenv("XAI_MODEL", "grok-4-latest"),
                    "messages": [prompt],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.1,
                },
            )
            r.raise_for_status()
            body = r.json()
            content = body["choices"][0]["message"]["content"]
            js = json.loads(content)
            lst = js.get("list") or []
            # Keep only items that existed in the input, unique, preserve LLM order
            picked = []
            seen = set()
            allowed = set(base)
            for x in lst:
                s = str(x).upper().strip()
                if s in allowed and s not in seen:
                    picked.append(s)
                    seen.add(s)
                if len(picked) >= top_k:
                    break
            # Fallback: fill the remainder from the original order
            if len(picked) < top_k:
                for s in base:
                    if s not in seen:
                        picked.append(s)
                        if len(picked) >= top_k:
                            break
            return picked[:top_k]
    except Exception as e:
        logger.warning(f"llm_shortlist failed; falling back: {e}")
        return base[:top_k]

# ====== Simulation → LLM Summary =============================================
def _band_last(bands: dict, key: str):
    arr = bands.get(key)
    if isinstance(arr, list) and arr:
        last = arr[-1]
        if isinstance(last, (list, tuple)) and len(last) >= 2:
            return float(last[1])
        try: return float(last)
        except: return None
    return None

def _artifact_context(art: dict) -> dict:
    """Trim the artifact to just what's useful for the LLM (no giant arrays)."""
    mp = art.get("median_path") or []
    # --- Spot / terminal from multiple possible locations
    try:
        s0 = (
            float(art.get("spot"))
            if art.get("spot") is not None
            else float(((art.get("inputs") or {}).get("S0") or 0.0))
        )
        if (not s0) and mp and len(mp[0]) >= 2:
            s0 = float(mp[0][1])
    except Exception:
        s0 = 0.0

    try:
        sh = float(mp[-1][1]) if mp and len(mp[-1]) >= 2 else None
    except Exception:
        sh = None

    try:
        med_ret_pct = ((sh / s0) - 1.0) * 100.0 if (s0 and sh) else 0.0
    except Exception:
        med_ret_pct = 0.0

    bands = art.get("bands") or art.get("qbands") or {}
    var_es = art.get("var_es") or {}
    drivers = art.get("drivers") or []
    news = art.get("news") or []

    # Read final band values, supporting both naming schemes
    p80_low  = _band_last(bands, "p80_low")  or _band_last(bands, "p10")
    p80_high = _band_last(bands, "p80_high") or _band_last(bands, "p90")
    p95_low  = _band_last(bands, "p95_low")  or _band_last(bands, "p2_5")
    p95_high = _band_last(bands, "p95_high") or _band_last(bands, "p97_5")

    # Vol lives under calibration.sigma_annualized in your artifact
    calib = art.get("calibration") or {}
    model_info = art.get("model_info") or {}

    ctx = {
        "symbol": art.get("symbol"),
        "horizon_days": art.get("horizon_days"),
        "prob_up_end": art.get("prob_up_end"),
        "prob_up_next": art.get("prob_up_next"),
        "median_return_pct": round(float(med_ret_pct), 2),

        "p80_low": p80_low,
        "p80_high": p80_high,
        "p95_low": p95_low,
        "p95_high": p95_high,

        "var95": var_es.get("var95"),
        "es95": var_es.get("es95"),

        "regime": model_info.get("regime"),
        "regime_score": model_info.get("regime_score"),
        "vol_annualized": calib.get("sigma_annualized"),  # renamed for the LLM
        # features_top is optional/absent in your artifact; omit if not present
        "drivers": drivers[:5],
        "news_top": [
            {"headline": n.get("headline"), "source": n.get("source"), "ts": n.get("ts")}
            for n in (news[:3] if isinstance(news, list) else [])
        ],
    }

    # Compact numeric fields
    for k in ["p80_low","p80_high","p95_low","p95_high","var95","es95","prob_up_end","prob_up_next","vol_annualized"]:
        v = ctx.get(k)
        if isinstance(v, (int, float)):
            try:
                ctx[k] = round(float(v), 4)
            except Exception:
                pass

    return ctx

async def _llm_summarize_context(ctx: dict) -> dict:
    """
    Ask the LLM to write a concise summary. Returns a dict:
      { "summary": str, "what_it_means": [...], "risks": [...], "watch": [...],
        "confidence": "low|medium|high", "metrics": {...} }
    Falls back to a heuristic if no key or an error occurs.
    """
    def _fallback(ctx: dict) -> dict:
        pu  = ctx.get("prob_up_end")
        med = ctx.get("median_return_pct")
        reg = (ctx.get("regime") or "neutral").replace("_", " ")

        direction = (
            "slightly higher" if (isinstance(pu, (int, float)) and pu >= 0.55)
            else "slightly lower" if (isinstance(pu, (int, float)) and pu <= 0.45)
            else "mostly sideways"
        )

        summary = (
            f"{ctx.get('symbol','?')} over the next {ctx.get('horizon_days',0)} days looks {direction}. "
            "The typical path leans that way, but outcomes vary. "
            f"Current regime appears {reg}."
        )

        return {
            "summary": summary,
            "what_it_means": [
                "Median path shows the base case; tails remain possible",
                "Probabilities are not guarantees"
            ],
            "risks": [
                "Macro surprises or earnings shocks",
                "Regime shifts changing volatility quickly"
            ],
            "watch": ["Fresh headlines/sentiment", "Upcoming earnings/data", "Regime stabilization"],
            "confidence": "medium",
            "metrics": {
                "prob_up_end": pu,
                "median_return_pct": med,
                "p80_low": ctx.get("p80_low"),
                "p80_high": ctx.get("p80_high"),
                "p95_low": ctx.get("p95_low"),
                "p95_high": ctx.get("p95_high"),
            },
        }

    import os, json
    oai_key = os.getenv("OPENAI_API_KEY","").strip()
    xai_key = os.getenv("XAI_API_KEY","").strip()
    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    prefer_xai = (provider == "xai") or (not oai_key and bool(xai_key))

    prompt_user = {
        "role": "user",
        "content": (
            "You are a quant PM speaking to a general audience. Given this Monte-Carlo summary JSON, write a plain-English "
            "explanation of what it means.\n"
            "Rules:\n"
            "• Focus on interpretation, not raw stats. You may mention one or two rounded numbers (e.g., 'about 60%').\n"
            "• Explain what the horizon means and why the outlook could change.\n"
            "• Keep it neutral, factual, and compliance-safe. No advice.\n"
            "• Output JSON only.\n\n"
            "Return JSON with:\n"
            "  summary: 70–120 words\n"
            "  what_it_means: 2–3 bullets\n"
            "  risks: 2–3 bullets\n"
            "  watch: up to 3 bullets\n"
            "  confidence: one of ['low','medium','high']\n"
            "  metrics: echo key numbers you used\n\n"
            f"Context JSON:\n{json.dumps(ctx, ensure_ascii=False)}"
        )
    }

    json_schema = {
        "name": "SimSummaryV2",
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "minLength": 40, "maxLength": 700},
                "what_it_means": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 3},
                "risks": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 3},
                "watch": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
                "confidence": {"type": "string", "enum": ["low","medium","high"]},
                "metrics": {
                    "type": "object",
                    "properties": {
                        "prob_up_end": {"type":"number"},
                        "median_return_pct": {"type":"number"},
                        "p80_low": {"type":["number","null"]},
                        "p80_high": {"type":["number","null"]},
                        "p95_low": {"type":["number","null"]},
                        "p95_high": {"type":["number","null"]}
                    },
                    "required": ["prob_up_end","median_return_pct"]
                }
            },
            "required": ["summary","what_it_means","risks","watch","confidence","metrics"],
            "additionalProperties": False
        },
        "strict": True
    }

    try:
        out = await llm_summarize_async(
            prompt_user,
            prefer_xai=prefer_xai,
            xai_key=xai_key,
            oai_key=oai_key,
            json_schema=json_schema,
        )
        # If provider returned a JSON string, llm_summarize_async already json.loads() it.
        if isinstance(out, dict) and out.get("summary"):
            return out
    except Exception as e:
        logger.info(f"_llm_summarize_context: LLM failed; using fallback: {e}")

    return _fallback(ctx)


async def _summarize_run(run_id: str, force: bool = False) -> dict:
    key = f"run:{run_id}:summary:v2"  # bump to v2 to avoid stale cache
    if REDIS and not force:
        cached = await REDIS.get(key)
        if cached:
            try:
                return json.loads(cached)
            except Exception:
                pass

    raw = await REDIS.get(f"artifact:{run_id}")
    if not raw:
        raise HTTPException(status_code=404, detail="artifact not found")

    art = json.loads(raw)
    ctx = _artifact_context(art)
    out = await _llm_summarize_context(ctx)
    out["run_id"] = run_id
    out["symbol"] = ctx.get("symbol")
    out["horizon_days"] = ctx.get("horizon_days")

    if REDIS:
        try:
            await REDIS.setex(key, 7*86400, json.dumps(out))
        except Exception:
            pass
    return out


# ===== DAILY QUANT (shortlist → MC → LLM → persist) ===========================

async def _quick_score(symbol: str) -> dict:
    """Fast prescreen: small history fetch + simple features + light tilt."""
    try:
        sym_fetch = _to_polygon_ticker(symbol)
        px = await _fetch_cached_hist_prices(sym_fetch, 180, REDIS)
        if not px or len(px) < 30:
            return {"symbol": symbol, "ok": False}
        f = await _feat_from_prices(symbol, px)
        reg = _detect_regime(np.asarray(px, dtype=float))
        try:
            p_lin = await asyncio.wait_for(get_ensemble_prob_light(symbol, REDIS, 1), timeout=0.6)
        except Exception:
            p_lin = 0.5
        mom = float(f.get("mom_20", 0.0))
        mom_n = float(np.tanh(mom / 0.10))  # ~±1 for ±10%
        score = float(np.clip(0.55 * p_lin + 0.30 * (0.5 * (mom_n + 1)) + 0.15 * (0.5 * (reg.get("score",0)+1)), 0, 1))
        return {"symbol": symbol, "ok": True, "p_quick": p_lin, "mom_20": mom, "regime": reg.get("name","neutral"), "score": score}
    except Exception as e:
        logger.info(f"quick_score failed for {symbol}: {e}")
        return {"symbol": symbol, "ok": False}

async def _rank_candidates(symbols: Sequence[str], top_k: int = 8) -> list[dict]:
    max_tasks   = int(os.getenv("PT_QS_TASKS", "12"))
    concurrency = int(os.getenv("PT_POLY_CONC", "2"))
    jitter_ms   = int(os.getenv("PT_QS_JITTER_MS", "0"))

    # Deterministic daily shuffle so we don't bias alphabetically
    seed = os.getenv("PT_DAILY_SEED") or datetime.utcnow().strftime("%Y%m%d")
    rng  = random.Random(f"{seed}-rank")  # stable per day
    pool = list(symbols)
    rng.shuffle(pool)
    todo = pool[:max(1, max_tasks)]

    sem = asyncio.Semaphore(max(1, concurrency))

    async def guarded(sym: str):
        async with sem:
            if jitter_ms:
                await asyncio.sleep(rng.random() * (jitter_ms / 1000.0))
            try:
                r = await _quick_score(sym)
            except Exception as e:
                r = {"ok": False, "symbol": sym, "error": f"{type(e).__name__}: {e}"}
            return r

    results = await asyncio.gather(*(guarded(s) for s in todo))
    good = [r for r in results if r.get("ok")]

    # Seeded tiebreaker so equal scores don't revert to alpha order
    def sort_key(r):
        return (r.get("score") or 0.0, rng.random())
    good.sort(key=sort_key, reverse=True)

    # (optional) debug: see who made the cut
    try:
        logger.info("Prescreen finalists: %s",
                    [(g.get("symbol"), round(g.get("score") or 0.0, 4)) for g in good[:top_k]])
    except Exception:
        pass

    return good[:top_k]

async def _mc_for(symbol: str, horizon: int = 30, paths: int = 6000) -> tuple[str, dict]:
    """Run a one-off MC inline (reusing your worker) and summarize for the judge."""
    req = SimRequest(symbol=_to_polygon_ticker(symbol), horizon_days=int(horizon), paths=int(paths))
    run_id = uuid4().hex
    rs = RunState(run_id=run_id, status="queued", progress=0.0, symbol=req.symbol,
                  horizon_days=req.horizon_days, paths=req.paths, startedAt=datetime.now(timezone.utc).isoformat())
    await REDIS.setex(f"run:{run_id}", settings.run_ttl_seconds, rs.model_dump_json())
    await run_simulation(run_id, req, REDIS)  # run inline

    raw = await REDIS.get(f"artifact:{run_id}")
    art = json.loads(raw) if raw else {}
    try:
        S0 = float(art.get("median_path", [[0,0]])[0][1])
        SH = float(art.get("median_path", [[0,0]])[-1][1])
        med_pct = float((SH / S0 - 1) * 100) if S0 else 0.0
    except Exception:
        med_pct = 0.0
    summary = {
        "symbol": symbol,
        "run_id": run_id,
        "horizon_days": int(art.get("horizon_days", horizon)),
        "prob_up_end": float(art.get("prob_up_end", 0.5)),
        "median_return_pct": float(med_pct),
        "var95": (art.get("var_es") or {}).get("var95"),
        "es95": (art.get("var_es") or {}).get("es95"),
        "regime": (art.get("model_info") or {}).get("regime", "neutral"),
    }

    # Optional quantum refinement (once per day; Aer by default)
    try:
        use_q = os.getenv("PT_USE_QUANTUM", "0").strip() == "1"
        if use_q and await _quant_allow(REDIS, max_calls_per_day=int(os.getenv("PT_QUANT_MAX_CALLS", "1"))):
            from .quantum_engine import prob_up_from_terminal
            term = art.get("terminal_prices") or []
            s0 = float(art.get("spot") or 0.0)
            if term and s0 > 0:
                q_prob = prob_up_from_terminal(term, s0, bins=int(os.getenv("PT_QUANT_BINS", "64")))
                if 0.0 <= q_prob <= 1.0:
                    summary["prob_up_end_q"] = float(q_prob)
                    # record which value we actually use downstream
                    use_q_prob = os.getenv("PT_QUANT_OVERRIDE", "1") == "1"
                    if use_q_prob:
                        summary["prob_up_end"] = float(q_prob)
            await _quant_consume(REDIS)
    except Exception as e:
        logger.info(f"Quantum refinement skipped: {e}")

    return symbol, summary


async def _mc_batch(cands: list[dict], horizon=30, paths=8000) -> list[dict]:
    picks = cands[:3]  # compute budget: top-3 only
    tasks = [asyncio.create_task(_mc_for(x["symbol"], horizon, paths)) for x in picks]
    out = []
    for t in asyncio.as_completed(tasks):
        _sym, summary = await t
        out.append(summary)
    out.sort(key=lambda s: (s.get("prob_up_end",0), s.get("median_return_pct",0)), reverse=True)
    return out

# --- LLM adjudicator (xAI/OpenAI) --------------------------------------------
async def _llm_select_and_write(kind: str, summaries: list[dict]) -> tuple[str, str]:
    """
    Return (symbol, blurb). Uses xAI if LLM_PROVIDER=xai or OpenAI missing.
    Falls back to a deterministic heuristic if no key / parse failure.
    """
    if not summaries:
        return "", ""

    best = max(summaries, key=lambda s: (s.get("prob_up_end", 0), s.get("median_return_pct", 0)))
    default_blurb = (
        f"{best['symbol']}: {best.get('horizon_days',30)}d MC P(up)≈{best.get('prob_up_end',0):.2%}, "
        f"median ≈ {best.get('median_return_pct',0):.1f}% • regime {best.get('regime','neutral')}."
    )

    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()  # "xai" | "openai" | ""
    oai_key = os.getenv("OPENAI_API_KEY", "").strip()
    xai_key = os.getenv("XAI_API_KEY", "").strip()
    prefer_xai = (provider == "xai") or (not oai_key and bool(xai_key))

    async def _call_xai() -> str:
        model = os.getenv("XAI_MODEL", "grok-4-latest")
        async with httpx.AsyncClient(timeout=20.0) as cli:
            r = await cli.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {xai_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "Be concise, factual, compliance-safe."},
                        {"role": "user", "content":
                            "You are a risk-aware quantitative analyst. Given JSON candidate summaries, "
                            "pick ONE ticker with the best risk-adjusted outlook for the stated horizon. "
                            "Respond as JSON with keys {symbol, blurb}. The blurb must be ≤280 chars, "
                            "objective, no guarantees, and mention 1–2 drivers.\n\n"
                            f"Candidates ({kind}):\n{json.dumps(summaries, ensure_ascii=False)}"
                        },
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "QuantPick",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "symbol": {"type": "string"},
                                    "blurb": {"type": "string", "maxLength": 280}
                                },
                                "required": ["symbol", "blurb"],
                                "additionalProperties": False
                            },
                            "strict": True
                        },
                    },
                    "temperature": 0.2,
                },
            )
            r.raise_for_status()
            return (r.json().get("choices") or [{}])[0].get("message", {}).get("content", "") or ""

    async def _call_openai() -> str:
        async with httpx.AsyncClient(timeout=20.0) as cli:
            r = await cli.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {oai_key}"},
                json={
                    "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    "messages": [
                        {"role": "system", "content": "Be concise, factual, compliance-safe."},
                        {"role": "user", "content":
                            "You are a risk-aware quantitative analyst. Given JSON candidate summaries, "
                            "pick ONE ticker with the best risk-adjusted outlook for the stated horizon. "
                            "Respond as JSON with keys {symbol, blurb}. The blurb must be ≤280 chars, "
                            "objective, no guarantees, and mention 1–2 drivers.\n\n"
                            f"Candidates ({kind}):\n{json.dumps(summaries, ensure_ascii=False)}"
                        },
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.2,
                },
            )
            r.raise_for_status()
            return (r.json().get("choices") or [{}])[0].get("message", {}).get("content", "") or ""

    try:
        txt = ""
        if prefer_xai and xai_key:
            txt = await _call_xai()
        elif oai_key:
            txt = await _call_openai()
        elif xai_key:
            txt = await _call_xai()
        else:
            return best["symbol"], default_blurb

        js = json.loads(txt.strip())
        sym = (js.get("symbol") or best["symbol"]).upper().strip()
        blurb = (js.get("blurb") or default_blurb).strip()
        return sym, blurb
    except Exception as e:
        logger.info(f"LLM pick failed; using heuristic: {e}")
        return best["symbol"], default_blurb

async def _persist_signal(kind: str, as_of: str, pick: dict):
    """Cache to Redis (7d) and mirror to DuckDB table signals_daily."""
    key = f"quant:daily:{as_of}:{kind}"
    if REDIS:
        try:
            await REDIS.setex(key, 7*86400, json.dumps(pick))
        except Exception as e:
            logger.warning(f"Redis persist failed: {e}")
    # DuckDB mirror (best-effort)
    try:
        con = fs_connect()
        con.execute("""
            CREATE TABLE IF NOT EXISTS signals_daily (
                as_of       DATE,
                kind        VARCHAR,
                symbol      VARCHAR,
                horizon_d   INTEGER,
                prob_up     DOUBLE,
                med_return  DOUBLE,
                var95       DOUBLE,
                es95        DOUBLE,
                run_id      VARCHAR,
                blurb       VARCHAR
            )
        """)
        con.execute(
            "INSERT INTO signals_daily VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                as_of, kind, pick.get("symbol"), int(pick.get("horizon_days", 30)),
                float(pick.get("prob_up_end", 0.5)), float(pick.get("median_return_pct", 0.0)),
                pick.get("var95"), pick.get("es95"), pick.get("run_id"), pick.get("blurb","")
            ],
        )
        con.close()
    except Exception as e:
        logger.warning(f"DuckDB persist failed: {e}")

async def _run_daily_quant(horizon_days: int | None = None) -> dict:
    """
    Orchestrates the daily run:
      1) LLM shortlist per asset class
      2) Prescreen to rank and pick finalists
      3) Monte-Carlo on finalists
      4) Select winners, publish to Redis
      5) Optionally attach top-N and/or full finalist lists
    """
    # --- config ---
    today_str = datetime.now(timezone.utc).date().isoformat()
    horizon = int(horizon_days or int(os.getenv("PT_DEFAULT_HORIZON_DAYS", "30")))
    seed_k  = int(os.getenv("PT_DAILY_SEED_K", "20"))
    rank_k  = int(os.getenv("PT_DAILY_RANK_K", "5"))
    mc_conc = int(os.getenv("PT_MC_CONC", "4"))

    top_n   = max(1, int(os.getenv("PT_TOP_N", "1")))
    expose_finalists = os.getenv("PT_EXPOSE_FINALISTS", "0") == "1"

    # --- watchlists (settings preferred; fall back to globals if present) ---
    eq_watch = getattr(settings, "equity_watch", globals().get("EQUITY_WATCH", []))
    cr_watch = getattr(settings, "crypto_watch", globals().get("CRYPTO_WATCH", []))

    if not eq_watch or not cr_watch:
        logger.warning("Watchlists are empty or missing; equity=%d crypto=%d",
                       len(eq_watch or []), len(cr_watch or []))

    # --- 1) LLM shortlist (deterministic seeded fallback inside llm_shortlist) ---
    eq_seed: list[str] = await llm_shortlist("equity", list(eq_watch), top_k=seed_k)
    cr_seed: list[str] = await llm_shortlist("crypto", list(cr_watch), top_k=seed_k)

    # --- 2) Prescreen/rank finalists ---
    eq_rank: list[dict] = await _rank_candidates(eq_seed, top_k=rank_k)
    cr_rank: list[dict] = await _rank_candidates(cr_seed, top_k=rank_k)

    # Quick visibility in logs
    try:
        logger.info("Equity prescreen finalists: %s",
                    [(r.get("symbol"), round(r.get("score") or 0.0, 4)) for r in eq_rank])
        logger.info("Crypto prescreen finalists: %s",
                    [(r.get("symbol"), round(r.get("score") or 0.0, 4)) for r in cr_rank])
    except Exception:
        pass

    # --- helper: call your Monte-Carlo summarizer ---
    async def _mc_summary(symbol: str, hz: int) -> dict:
        """
        ADJUST HERE if your MC function name differs.
        Expected to return a dict like:
          {
            "ok": True,
            "symbol": "AMD",
            "run_id": "...",
            "horizon_days": 30,
            "prob_up_end": 0.78,
            "median_return_pct": 11.5,
            "var95": -0.11,
            "es95": -0.16,
            "regime": "neutral",
            "blurb": "AMD: 30d ..."
          }
        """
        # Try a few common helper names; tweak as needed for your codebase.
        for name in ("_simulate_and_summarize",
                     "_simulate_mc_summary",
                     "_monte_carlo_summary",
                     "_simulate_symbol",
                     "_simulate_mc",
                     "_monte_carlo"):
            fn = globals().get(name)
            if fn:
                if inspect.iscoroutinefunction(fn):
                    return await fn(symbol, hz)
                # run sync MC off-thread so we don't block the loop
                return await asyncio.to_thread(fn, symbol, hz)
        raise RuntimeError("Monte-Carlo helper not found; adjust _mc_summary() call-site.")

    # --- 3) Run MC on finalists (concurrency-limited) ---
    async def _mc_batch(finalists: Sequence[dict]) -> list[dict]:
        sem = asyncio.Semaphore(max(1, mc_conc))
        async def _one(item: dict) -> dict:
            sym = item.get("symbol")
            async with sem:
                try:
                    res = await _mc_summary(sym, horizon)
                    # Ensure required fields
                    res.setdefault("symbol", sym)
                    res.setdefault("horizon_days", horizon)
                    res.setdefault("ok", True)
                    return res
                except Exception as e:
                    logger.warning("MC failed for %s: %s", sym, e)
                    return {"ok": False, "symbol": sym, "error": f"{type(e).__name__}: {e}"}

        return await asyncio.gather(*(_one(x) for x in finalists))

    eq_mc = await _mc_batch(eq_rank)
    cr_mc = await _mc_batch(cr_rank)

    # Keep only successful MC results
    eq_mc_ok = [x for x in eq_mc if x.get("ok")]
    cr_mc_ok = [x for x in cr_mc if x.get("ok")]

    # --- 4) Pick winners (highest prob_up_end, then median_return_pct as tiebreak) ---
    def _winner(lst: list[dict]) -> dict:
        if not lst:
            return {}
        return max(
            lst,
            key=lambda r: (float(r.get("prob_up_end") or 0.0),
                           float(r.get("median_return_pct") or 0.0))
        )

    # Top-N arrays (truncate safely)
    eq_top = eq_mc_ok[:top_n]
    cr_top = cr_mc_ok[:top_n]
    eq_win = _winner(eq_mc_ok)
    cr_win = _winner(cr_mc_ok)

    payload: dict = {
        "as_of": today_str,
        "equity": eq_win,
        "crypto": cr_win,
    }
    if top_n > 1:
        payload["equity_top"] = eq_top
        payload["crypto_top"] = cr_top
    if expose_finalists:
        payload["equity_finalists"] = eq_mc_ok
        payload["crypto_finalists"] = cr_mc_ok

    # --- 5) Persist to Redis (winners always; extras if enabled) ---
    if REDIS:
        try:
            base = f"quant:daily:{today_str}"
            await REDIS.set(f"{base}:equity", json.dumps(eq_win))
            await REDIS.set(f"{base}:crypto", json.dumps(cr_win))
            if top_n > 1:
                await REDIS.set(f"{base}:equity_top", json.dumps(eq_top))
                await REDIS.set(f"{base}:crypto_top", json.dumps(cr_top))
            if expose_finalists:
                await REDIS.set(f"{base}:equity_finalists", json.dumps(eq_mc_ok))
                await REDIS.set(f"{base}:crypto_finalists", json.dumps(cr_mc_ok))
        except Exception as e:
            logger.warning("Failed to cache daily results in Redis: %s", e)

    return payload

def _poly_key_present() -> bool:
    return bool(os.getenv("PT_POLYGON_KEY") or os.getenv("POLYGON_KEY"))

def _is_weekend(d) -> bool:
    return d.weekday() >= 5  # 5=Sat, 6=Sun

@app.on_event("startup")
async def _on_startup():
    init_schema()
    asyncio.create_task(_gc_loop())
    asyncio.create_task(_labeling_daemon())
    asyncio.create_task(_warm_start())  # warm but gentle

async def _warm_start():
    """
    On cold start, gently prime data & cache without tripping rate limits.
    - Skip Polygon ingest if weekend or key is missing
    - Single-day ingest only (yesterday) if bars_daily is missing
    - Prime today's cache once, using the daily compute budget
    """
    try:
        # spread boots across instances
        jitter_ms = int(os.getenv("PT_WARM_JITTER_MS", "300"))
        if jitter_ms > 0:
            await asyncio.sleep(random.random() * (jitter_ms / 1000.0))

        # 1) If bars_daily is missing, ingest only yesterday (best effort)
        try:
            con = fs_connect()
            has = con.execute(
                "SELECT 1 FROM information_schema.tables WHERE lower(table_name)='bars_daily'"
            ).fetchall()
            con.close()
        except Exception as e:
            logger.info(f"warm table-check skipped: {e}")
            has = []

        if not has:
            y = datetime.now(timezone.utc).date() - timedelta(days=1)
            if _poly_key_present() and not _is_weekend(y):
                try:
                    await ingest_grouped_daily(y)
                except Exception as e:
                    logger.info(f"warm ingest skipped: {e}")
            else:
                logger.info("warm ingest skipped: weekend or no Polygon key")

        # 2) Prime today's cache (idempotent; will no-op if already cached)
        if REDIS:
            d = datetime.now(timezone.utc).date().isoformat()
            base = f"quant:daily:{d}"
            try:
                eq = await REDIS.get(f"{base}:equity")
                cr = await REDIS.get(f"{base}:crypto")
            except Exception as e:
                logger.info(f"warm redis get failed: {e}")
                eq = cr = None

            if not (eq and cr):
                warm_budget = int(os.getenv("PT_WARM_BUDGET", "2"))
                try:
                    if await _quant_allow(REDIS, max_calls_per_day=warm_budget):
                        try:
                            await _run_daily_quant()
                        finally:
                            try:
                                await _quant_consume(REDIS)
                            except Exception:
                                pass
                    else:
                        logger.info("warm quant skipped: budget exhausted")
                except Exception as e:
                    logger.info(f"warm quant skipped: {e}")
        else:
            logger.info("warm start: no REDIS configured; skipping cache prime")

    except Exception as e:
        logger.info(f"warm_start outer skipped: {e}")

@app.post("/admin/cron/daily")
async def admin_cron_daily(
    n: int = 20,              # how many symbols to learn
    steps: int = 50,          # SGD steps per symbol
    batch: int = 32,          # minibatch size
    _ok: bool = Depends(require_key),
):
    # 1) Label anything matured
    lab = await outcomes_label(limit=20000, _api_key=True)  # reuse route logic

    # 2) Learn on the top-N of your watchlists (equity + crypto)
    syms = list(dict.fromkeys(list(WL_EQ)[:n] + list(WL_CR)[:n]))
    learned = []
    for s in syms:
        try:
            res = await learn_online(
                OnlineLearnRequest(symbol=s, steps=steps, batch=batch), _api_key=True
            )
            learned.append({"symbol": s, "status": res.get("status", "ok")})
        except Exception as e:
            learned.append({"symbol": s, "status": "error", "error": str(e)})

    return {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "labeled": lab,
        "n_symbols": len(syms),
        "learned": learned,
    }

# --- IBM Quantum / Qiskit diagnostics ---
@app.get("/quant/aer/diag")
async def quant_aer_diag(_ok: bool = Depends(require_key)):
    info = {
        "ok": False,
        "qiskit_version": None,
        "aer_version": None,
        "shots": 2048,
        "counts": {},
        "p00": None,
        "p11": None,
        "imbalance": None,
        "note": "Bell state on Aer should yield ~50/50 between '00' and '11'.",
        "error": None,
    }
    try:
        from qiskit import QuantumCircuit, transpile, __version__ as qk_ver
        from qiskit_aer import AerSimulator, __version__ as aer_ver

        info["qiskit_version"] = qk_ver
        info["aer_version"] = aer_ver

        # Bell |Φ+> = (|00> + |11>)/√2
        qc = QuantumCircuit(2, 2)
        qc.h(0); qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        sim = AerSimulator(seed_simulator=42, method="automatic")
        tqc = transpile(qc, sim)
        res = sim.run(tqc, shots=info["shots"]).result()
        counts = res.get_counts()
        info["counts"] = counts
        shots = max(1, info["shots"])
        p00 = (counts.get("00", 0) / shots)
        p11 = (counts.get("11", 0) / shots)
        info["p00"] = p00
        info["p11"] = p11
        info["imbalance"] = abs(p00 - p11)
        info["ok"] = True
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


# --------- AER: Simple circuit runner ----------
class AerRunReq(BaseModel):
    circuit: str = "bell"  # "bell" or "ghz"
    n_qubits: int = 2
    shots: int = 2048
    seed: int | None = 123

@app.post("/quant/aer/run")
async def quant_aer_run(req: AerRunReq, _ok: bool = Depends(require_key)):
    out = {
        "ok": False,
        "circuit": req.circuit,
        "n_qubits": req.n_qubits,
        "shots": req.shots,
        "counts": {},
        "error": None,
    }
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator

        if req.circuit.lower() == "ghz":
            n = max(2, int(req.n_qubits))
            qc = QuantumCircuit(n, n)
            qc.h(0)
            for i in range(n - 1):
                qc.cx(i, i + 1)
            qc.measure(range(n), range(n))
        else:
            # default: bell
            qc = QuantumCircuit(2, 2)
            qc.h(0); qc.cx(0, 1)
            qc.measure([0, 1], [0, 1])

        sim = AerSimulator(seed_simulator=req.seed, method="automatic")
        tqc = transpile(qc, sim)
        res = sim.run(tqc, shots=int(req.shots)).result()
        out["counts"] = res.get_counts()
        out["ok"] = True
        return out
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
# --- Quantum budget helpers (per day) ---

def _quant_budget_key() -> str:
    from datetime import datetime, timezone
    return f"quant:budget:{datetime.now(timezone.utc).date().isoformat()}"

async def _quant_allow(redis: Redis, max_calls_per_day: int = 1) -> bool:
    try:
        k = _quant_budget_key()
        calls = await redis.get(k)
        n = int(calls or "0")
        return n < max_calls_per_day
    except Exception:
        return False

async def _quant_consume(redis: Redis) -> None:
    try:
        k = _quant_budget_key()
        p = await redis.pipeline()
        await p.incr(k)
        await p.expire(k, 3 * 24 * 3600)
        await p.execute()
    except Exception:
        pass
# --- Backfill with politeness pause between days ---
@app.post("/admin/ingest/backfill")
async def admin_ingest_backfill(days: int = 7, _ok: bool = Depends(require_key)):
    base = datetime.now(timezone.utc).date()
    total = 0
    pause_s = float(os.getenv("PT_BACKFILL_PAUSE_S", "2.5"))  # <— NEW
    for i in range(int(max(1, days))):
        out = await ingest_grouped_daily(base - timedelta(days=i))
        total += int(out.get("upserted", 0))
        await asyncio.sleep(pause_s)  # <— NEW
    return {"ok": True, "days": int(days), "rows": total}


@app.post("/admin/ingest/daily")
async def admin_ingest_daily(
    d: Optional[str] = Query(None, description="YYYY-MM-DD (UTC)"),
    _ok: bool = Depends(require_key),
):
    """
    Trigger grouped-daily ingest into DuckDB (creates/updates bars_daily).
    Optional query: ?d=YYYY-MM-DD (defaults to today, UTC).
    """
    try:
        as_of = date.fromisoformat(d) if d else datetime.now(timezone.utc).date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid 'd' format; expected YYYY-MM-DD")

    await ingest_grouped_daily(as_of)
    return {"ok": True, "date": as_of.isoformat()}

# ===== DAILY QUANT routes =====================================================

@app.post("/quant/daily/run")
async def quant_daily_run(horizon: int = 30, _ok: bool = Depends(require_key)):
    # Avoid over-computation if multiple schedulers/clients call in the same day
    budget = int(os.getenv("PT_QUANT_BUDGET_PER_DAY", "6"))  # e.g., allow up to 6 runs/day
    if REDIS:
        try:
            if not await _quant_allow(REDIS, max_calls_per_day=budget):
                # Return whatever is cached instead of recomputing
                return await quant_daily_today()
        except Exception:
            pass

    out = await _run_daily_quant(horizon_days=int(horizon))

    if REDIS:
        try: await _quant_consume(REDIS)
        except Exception: pass
    return out

@app.get("/quant/daily/today")
async def quant_daily_today():
    """
    Public: return today’s cached signal; if missing (first hit), compute on-demand.
    Keep this open so the front-end can fetch without CORS preflight/auth.
    """
    d = datetime.now(timezone.utc).date().isoformat()
    out: dict[str, object] = {"as_of": d}
    if REDIS:
        try:
            base = f"quant:daily:{d}"
            eq = await REDIS.get(f"{base}:equity")
            cr = await REDIS.get(f"{base}:crypto")
            if eq: out["equity"] = json.loads(eq)
            if cr: out["crypto"] = json.loads(cr)

            # optional extras if present
            eq_top = await REDIS.get(f"{base}:equity_top")
            cr_top = await REDIS.get(f"{base}:crypto_top")
            if eq_top: out["equity_top"] = json.loads(eq_top)
            if cr_top: out["crypto_top"] = json.loads(cr_top)

            eq_all = await REDIS.get(f"{base}:equity_finalists")
            cr_all = await REDIS.get(f"{base}:crypto_finalists")
            if eq_all: out["equity_finalists"] = json.loads(eq_all)
            if cr_all: out["crypto_finalists"] = json.loads(cr_all)
        except Exception:
            # fall through to compute
            pass

    # If either winner is missing, compute-on-demand (writes winners and extras if enabled)
    if not out.get("equity") or not out.get("crypto"):
        out = await _run_daily_quant()

    return out


@app.get("/quant/daily/history")
async def quant_daily_history(limit: int = 14, _ok: bool = Depends(require_key)):
    """Protected: last N rows from DuckDB mirror."""
    try:
        con = fs_connect()
        rows = con.execute(
            "SELECT * FROM signals_daily ORDER BY as_of DESC LIMIT ?",
            [int(max(1, min(limit, 60)))]
        ).fetchall()
        con.close()
        def to_obj(row):
            return {
                "as_of": row[0], "kind": row[1], "symbol": row[2], "horizon_d": row[3],
                "prob_up": row[4], "med_return": row[5], "var95": row[6], "es95": row[7],
                "run_id": row[8], "blurb": row[9],
            }
        return {"items": [to_obj(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"history failed: {e}")

@app.get("/runs/{run_id}/summary")
async def get_run_summary(run_id: str, refresh: int = 0):
    """
    Public: returns cached LLM summary for a completed run_id.
    - 202 while artifact isn't persisted yet
    - ?refresh=1 recomputes and overwrites cache
    """
    try:
        return await _summarize_run(run_id, force=bool(refresh))
    except HTTPException as e:
        if e.status_code == 404:
            return JSONResponse({"status": "pending"}, status_code=202)
        raise

@app.post("/outcomes/label")
async def outcomes_label(limit: int = 5000, _ok: bool = Depends(require_key)):
    """
    Label any matured predictions (ts + horizon_d <= now) that don't yet have outcomes.
    Writes to src.db.duck.outcomes and (optionally) mirrors each label into PathPanda FS
    so daily metrics rollups can see realized prices.
    """
    # clamp the batch size
    limit = max(100, min(int(limit), 20000))

    # 1) Load matured-but-unlabeled predictions from the Predictions/Outcomes store
    try:
        # [(pred_id, ts, symbol, horizon_d, spot0), ...]
        matured = matured_predictions_now(limit=limit)
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
            when = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
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
async def metrics_rollup(_ok: bool = Depends(require_key), day: Optional[str] = None):
    """
    Run daily metrics rollup for predictions joined with outcomes.
    Optional: ?day=YYYY-MM-DD (defaults to today in UTC).
    """
    try:
        d = date.fromisoformat(day) if day else datetime.now(timezone.utc).date()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid 'day' format; expected YYYY-MM-DD")

    con = fs_connect()
    try:
        n = _rollup(con, day=d)
    finally:
        con.close()

    return {"status": "ok", "date": d.isoformat(), "rows_upserted": int(n)}
@app.get("/accuracy-statements")
def get_accuracy_statements(
    limit: int = Query(50, ge=1, le=200),
    _ok: bool = Depends(require_key),
):
    try:
        from .feature_store import generate_accuracy_statements  # expects a connection + limit
    except Exception:
        raise HTTPException(status_code=500, detail="feature_store.generate_accuracy_statements missing")

    con = fs_connect()
    try:
        statements = generate_accuracy_statements(con, limit=limit)
    finally:
        con.close()
    return {"statements": statements}

@app.get("/config")
async def config():
    return {
        "n_paths_max": settings.n_paths_max,
        "horizon_days_max": settings.horizon_days_max,
        "pathday_budget_max": settings.pathday_budget_max,
        "predictive_defaults": settings.predictive_defaults,
        "cors_origins": ALLOWED_ORIGINS,
    }
@app.post("/session/anon")
def session_anon(response: Response):
    token = token_urlsafe(16)
    response.set_cookie(
        settings.cookie_name,
        token,
        httponly=True,
        secure=bool(settings.cookie_secure),
        samesite="Lax",
        max_age=int(settings.cookie_max_age),
    )
    return {"ok": True}

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

if "_to_polygon_ticker" not in globals():
    def _to_polygon_ticker(raw: str) -> str:
        s = (raw or "").upper().strip()
        if not s:
            return s
        if s.startswith("X:"):
            return s
        # BTC-USD / ETH-USD, etc.
        if s.endswith("-USD"):
            return f"X:{s.replace('-USD','')}USD"
        # BTCUSD style
        if s.endswith("USD") and "-" not in s:
            return f"X:{s}"
        return s

@app.get("/api/news/{symbol}")
async def get_news(
    symbol: str,
    limit: int = 10,
    days: int = 7,
    cursor: Optional[str] = None,
    _ok: bool = Depends(require_key),
):
    key = _poly_key()
    limit = max(1, min(int(limit), 50))
    days = max(1, min(int(days), 30))

    raw_symbol = (symbol or "").strip().upper()
    poly_ticker = _to_polygon_ticker(raw_symbol)

    cache_key = f"news:{poly_ticker}:{limit}:{days}:{cursor or 'first'}"
    if REDIS:
        try:
            cached = await REDIS.get(cache_key)
            if cached:
                try:
                    return json.loads(cached)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Redis get failed for {cache_key}: {e}")

    url = "https://api.polygon.io/v2/reference/news"
    params: dict[str, str] = {"limit": str(limit)}
    headers = {"Authorization": f"Bearer {key}"}

    if cursor:
        params["cursor"] = cursor
        params["ticker"] = poly_ticker
    else:
        since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
        params.update({"ticker": poly_ticker, "published_utc.gte": since})

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            payload = resp.json() or {}
            news = payload.get("results", []) or []
            next_url = payload.get("next_url")
    except httpx.HTTPStatusError as e:
        if e.response is not None and e.response.status_code == 429:
            raise HTTPException(status_code=429, detail="Rate limited by news provider")
        logger.error(f"Polygon error {getattr(e.response,'status_code','?')} for {poly_ticker}: {e}")
        raise HTTPException(status_code=502, detail="Upstream news provider error")
    except Exception as e:
        logger.error(f"News fetch failed for {poly_ticker}: {e}")
        raise HTTPException(status_code=502, detail="News fetch failed")

    seen: set[str] = set()
    processed: list[dict] = []
    for item in news:
        nid = item.get("id") or item.get("url") or item.get("article_url")
        if not nid or nid in seen:
            continue
        seen.add(nid)
        title = item.get("title", "") or ""
        processed.append({
            "id": nid,
            "title": title,
            "url": item.get("article_url") or item.get("url", "") or "",
            "published_at": item.get("published_utc", "") or "",
            "source": (item.get("publisher") or {}).get("name", "") or "",
            "sentiment": _safe_sent(title),
            "image_url": item.get("image_url", "") or "",
        })

    processed.sort(key=lambda x: x.get("published_at", "") or "", reverse=True)

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

    # map results
    seen: set[str] = set()
    processed: list[dict] = []
    for item in news:
        nid = item.get("id") or item.get("url") or item.get("article_url")
        if not nid or nid in seen:
            continue
        seen.add(nid)
        title = item.get("title", "") or ""
        processed.append(
            {
                "id": nid,
                "title": title,
                "url": item.get("article_url") or item.get("url", "") or "",
                "published_at": item.get("published_utc", "") or "",
                "source": (item.get("publisher") or {}).get("name", "") or "",
                "sentiment": _safe_sent(title),
                "image_url": item.get("image_url", "") or "",
            }
        )

    processed.sort(key=lambda x: x.get("published_at", "") or "", reverse=True)

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
    _ok: bool = Depends(require_key),
):
    """
    Lightweight options snapshot summary.
    Uses _poly_key() with Authorization header to avoid logging keys in URLs.
    """
    key = _poly_key()
    limit = max(1, min(int(limit), 50))

    url = f"https://api.polygon.io/v3/snapshot/options/{(symbol or '').upper().strip()}"
    params = {
        "contract_type": contract_type,
        "sort": "strike_price",
        "order": "asc",
        "limit": str(limit),
    }
    headers = {"Authorization": f"Bearer {key}"}

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json() or {}
    except httpx.HTTPStatusError as e:
        if e.response is not None and e.response.status_code == 429:
            raise HTTPException(status_code=429, detail="Rate limited by options provider")
        logger.warning(f"Options upstream error for {symbol}: {e}")
        raise HTTPException(status_code=502, detail=f"Options upstream error {getattr(e.response,'status_code','?')}")
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
        "symbol": (symbol or "").upper().strip(),
        "avg_iv": avg_iv,
        "avg_delta": avg_delta,
        "sample_contracts": sample_contracts,
        "source": "Polygon Options Snapshot",
    }

# --- Futures snapshot (Polygon) ---
@app.get("/futures/{symbol}")
async def get_futures_snapshot(symbol: str, _api_key: str = Depends(require_key)):
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
async def x_sentiment(symbol: str, handles: str = "", _api_key: str = Depends(require_key)):
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

#---------Train--------------------
@app.post("/train")
async def train(req: TrainRequest, _api_key: str = Depends(require_key)):
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
async def learn_online(req: OnlineLearnRequest, _api_key: str = Depends(require_key)):
    """
    Mini online-learning pass on recent labeled samples.
    Uses three core features {mom_20, rvol_20, autocorr_5} from _load_labeled_samples().
    Updates Redis model:{SYMBOL}_linear in-place.
    """
    X, y = _load_labeled_samples(req.symbol, limit=max(128, req.steps * req.batch))
    if X.size == 0 or y.size == 0:
        return {"status": "no_data"}

    # Pull current model (or init)
    key = await _model_key(req.symbol + "_linear")
    w0 = None
    meta = {}
    raw = await REDIS.get(key)
    if raw:
        try:
            js = json.loads(raw)
            w0 = np.asarray(js.get("coef", []), dtype=float)
            feats = js.get("features", ["mom_20", "rvol_20", "autocorr_5"])
            meta = {k: js.get(k) for k in ("trained_at","n_samples")}
            # align dimensionality if needed
            if w0.shape[0] != len(feats) + 1:  # +1 for bias
                w0 = None
        except Exception:
            w0 = None

    # Build batches
    rng = np.random.default_rng(42)
    idx = np.arange(X.shape[0]); rng.shuffle(idx)
    Xs = X[idx]; ys = y[idx].astype(float)

    # Initialize learner
    learner = SGDOnline(lr=float(req.lr), l2=float(req.l2))
    learner.init(X.shape[1])  # bias handled internally in learner

    # Optionally warm start
    if w0 is not None and hasattr(learner, "w"):
        learner.w = w0.astype(float)

    # Train
    steps = int(req.steps)
    bsz   = int(req.batch)
    for t in range(steps):
        lo = (t * bsz) % Xs.shape[0]
        hi = lo + bsz
        xb = Xs[lo:hi]
        yb = ys[lo:hi]
        # simple classification target (>= mid)
        for i in range(xb.shape[0]):
            learner.update(xb[i], float(yb[i]))

    # Persist back to Redis
    model = {
        "coef": learner.w.tolist(),
        "features": ["mom_20", "rvol_20", "autocorr_5"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": int((meta.get("n_samples") or 0) + X.shape[0]),
    }
    await REDIS.set(key, json.dumps(model))
    return {"status": "ok", "model": model}

@app.post("/predict")
async def predict(req: PredictRequest, _api_key: str = Depends(require_key)):
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
async def models(_api_key: str = Depends(require_key)):
    return {"models": await _list_models()}


# ---- Optional static hosting (disabled by default) ----
# If you want the API to also serve your built frontend, set PT_FRONTEND_DIR
# to the absolute path of the SPA build directory (the one with index.html).
FRONTEND_DIR = os.getenv("PT_FRONTEND_DIR", "").strip()
# ----------------- Health -----------------
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
        "polygon_key_mode": key_mode,
        "n_paths_max": settings.n_paths_max,
        "horizon_days_max": settings.horizon_days_max,
        "pathday_budget_max": settings.pathday_budget_max,
    }

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
            "message": "SIMETRIX.IO API Is Running."
        }
