# --- stdlib
from __future__ import annotations
import os, json, asyncio, logging, math, pickle, shutil, io, csv
from datetime import datetime, timedelta, date, timezone
from uuid import uuid4
from typing import List, Optional, Any, Callable, Dict, Literal, Sequence, Tuple, Mapping, Iterable
from contextlib import asynccontextmanager
import time, inspect
from statistics import NormalDist
import hmac, hashlib, base64
# --- third-party
from dotenv import load_dotenv; load_dotenv()
import numpy as np
import random
import httpx
from scipy import stats
from pathlib import Path
from fastapi import (
    FastAPI, Depends, HTTPException, WebSocket, Query, Header, Body,
    Request, Response, Security, status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, SecurityScopes
from pydantic import BaseModel, Field, ConfigDict, model_validator
from redis.asyncio import Redis
from secrets import token_urlsafe
from starlette.middleware.base import BaseHTTPMiddleware
from jose import JWTError, jwt
try:
    from passlib.hash import bcrypt as bcrypt_hash  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    bcrypt_hash = None  # type: ignore
from src.auth_service import all_scopes_for, UserKey
from src.db.migrations import run_all as run_all_migrations
from src.observability import install_observability, log_json, job_ok, job_fail
from src.quotas import (
    enforce_limits,
    rate_limit,
    usage_today,
    usage_today_for_caller,
    set_plan_for_key,
    get_plan_for_key,
    BASE_LIMIT_PER_MIN,
    SIM_LIMIT_PER_MIN,
    CRON_LIMIT_PER_MIN,
    PLAN_DEFAULT,
)
from src.observability import get_recent_logs
from src.infra.backups import create_duckdb_backup
from src.sim_validation import rollforward_validation
from src.services import auth as auth_service
from src.services.quant_daily import fetch_minimal_summary
from src.services.labeling import (
    label_outcomes as service_label_outcomes,
    labeler_pass as service_labeler_pass,
    run_labeling_pass as service_run_labeling_pass,
)
from src.services.quant_utils import (
    LONG_HORIZON_DAYS,
    LONG_HORIZON_SHRINK_FLOOR,
    auto_rel_levels,
    ewma_sigma,
    horizon_shrink,
    rel_levels_from_expected_move,
    simulate_gbm_student_t,
    winsorize,
)
from src.services.quant_mc import latest_mc_metric as service_latest_mc_metric
from src.services.quant_candidates import llm_shortlist as service_llm_shortlist
from src.services.quant_scheduler import (
    quant_budget_key as service_quant_budget_key,
    quant_allow as service_quant_allow,
    quant_consume as service_quant_consume,
    run_daily_quant as service_run_daily_quant,
    trigger_daily_quant_from_health as service_trigger_daily_quant_from_health,
    daily_quant_scheduler as service_daily_quant_scheduler,
)
from src.services.quant_context import detect_regime as service_detect_regime
from src.services import quant_adapters
from src.services.ingestion import fetch_cached_hist_prices as ingestion_fetch_cached_hist_prices
from src.services.training import (
    _feat_from_prices as training_feat_from_prices,
    OnlineLearnRequest,
    learn_online as service_learn_online,
    DEFAULT_TRAIN_PROFILE as TRAIN_DEFAULT_PROFILE,
    linear_model_key_for_profile as training_linear_model_key,
    resolve_training_profile as training_resolve_profile,
    training_profile_lookback as training_profile_lookback,
    _train_models as service_train_models,
    _ensure_trained_models as service_ensure_trained_models,
)
from src.services.inference import (
    get_ensemble_prob as service_get_ensemble_prob,
    get_ensemble_prob_light as service_get_ensemble_prob_light,
)
from src.model_registry import (
    register_model_version,
    get_active_model_version,
    list_model_versions,
    promote_model_version,
)
from src.core import (
    DATA_ROOT,
    BACKUP_DIR,
    EXPORT_ROOT,
    ARROW_ROOT,
    ARROW_PREDICTIONS_DIR,
    ARROW_OUTCOMES_DIR,
    START_TIME,
    JWT_ALGORITHM,
    CALIBRATION_SAMPLE_LIMIT,
    CALIBRATION_SAMPLE_MIN,
    CALIBRATION_TTL_SECONDS,
    CALIBRATION_DB_MAX_AGE,
    CALIBRATION_GRID_SIZE,
    IV_CACHE_TTL,
    settings,
    REDIS,
    SIM_DISPATCHER,
    _ARIMA_MODEL_CACHE,
    _LSTM_MODEL_CACHE,
    _LSTM_INFER_CACHE,
    _ONNX_SESSION_CACHE,
    _RL_MODEL_CACHE,
    _MODEL_META_CACHE,
    _IV_CACHE,
    resolve_artifact_path as core_resolve_artifact_path,
)

try:
    from prometheus_client import Histogram, Gauge, Counter

    SIM_DURATION_SECONDS = Histogram(
        "simetrix_simulation_duration_seconds",
        "Wall clock runtime of simulation jobs",
        ["mode"],
    )
    SIM_ACTIVE_GAUGE = Gauge(
        "simetrix_simulation_active_tasks",
        "Number of simulations currently running",
    )
    NEWS_INGESTED_COUNTER = Counter(
        "news_ingested_total",
        "Count of news rows ingested into feature store",
    )
    NEWS_SCORED_COUNTER = Counter(
        "news_scored_total",
        "Count of news rows sentiment-scored",
    )
    EARNINGS_INGESTED_COUNTER = Counter(
        "earnings_ingested_total",
        "Count of earnings rows ingested",
    )
    MACRO_UPSERTS_COUNTER = Counter(
        "macro_upserts_total",
        "Count of macro snapshot upserts",
    )
    INFERENCE_LATENCY_SECONDS = Histogram(
        "simetrix_inference_latency_seconds",
        "Latency of inference requests",
        ["endpoint"],
    )
    MODEL_USAGE_COUNTER = Counter(
        "simetrix_model_usage_total",
        "Count of model head invocations during inference",
        ["model"],
    )
    CALIBRATION_ERROR_GAUGE = Gauge(
        "simetrix_calibration_error",
        "Difference between calibrated and fallback quantile bands; -1 indicates fallback only",
        ["symbol", "horizon_days"],
    )
except Exception:  # pragma: no cover - metrics optional
    SIM_DURATION_SECONDS = None  # type: ignore
    SIM_ACTIVE_GAUGE = None  # type: ignore
    NEWS_INGESTED_COUNTER = None  # type: ignore
    NEWS_SCORED_COUNTER = None  # type: ignore
    EARNINGS_INGESTED_COUNTER = None  # type: ignore
    MACRO_UPSERTS_COUNTER = None  # type: ignore
    INFERENCE_LATENCY_SECONDS = None  # type: ignore
    MODEL_USAGE_COUNTER = None  # type: ignore
    CALIBRATION_ERROR_GAUGE = None  # type: ignore

try:
    if os.getenv("PT_SKIP_ONNX", "0") == "1":
        raise ImportError("ONNX disabled via PT_SKIP_ONNX")
    import onnx  # type: ignore
    from onnx import helper as onnx_helper, TensorProto
except Exception:  # pragma: no cover
    onnx = None  # type: ignore
    onnx_helper = None  # type: ignore
    TensorProto = None  # type: ignore

try:
    if os.getenv("PT_SKIP_ONNXRUNTIME", "0") == "1":
        raise ImportError("onnxruntime disabled via PT_SKIP_ONNXRUNTIME")
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None  # type: ignore
    
import duckdb

# Restore old date() convenience
duckdb.sql("CREATE OR REPLACE MACRO date(x) AS CAST(x AS DATE);")

# One-time schema guards – these will just no-op if columns already exist
try:
    con = duckdb.connect("./data/pt.duckdb")
    con.execute("ALTER TABLE IF EXISTS predictions ADD COLUMN IF NOT EXISTS issued_at TIMESTAMP;")
    con.execute("ALTER TABLE IF EXISTS predictions ADD COLUMN IF NOT EXISTS features_ref VARCHAR;")
    con.close()
except Exception as e:
    logger.warning("Startup schema check skipped: %s", e)

try:
    con = duckdb.connect("./data/pathpanda.duckdb")
    con.execute("ALTER TABLE IF EXISTS predictions ADD COLUMN IF NOT EXISTS issued_at TIMESTAMP;")
    con.execute("ALTER TABLE IF EXISTS predictions ADD COLUMN IF NOT EXISTS features_ref VARCHAR;")
    con.close()
except Exception as e:
    logger.warning("Startup feature-store schema check skipped: %s", e)
def ensure_pred_schema(con: duckdb.DuckDBPyConnection | None = None, path: str | None = None):
    """Guarantee every DuckDB used anywhere has the correct predictions schema."""
    try:
        if con is None:
            con = duckdb.connect(path or ":memory:")
        cols = {r[1] for r in con.execute("PRAGMA table_info('predictions')").fetchall()}
        if "issued_at" not in cols:
            logger.info("Auto-patching 'issued_at' column in %s", path or '<memory>')
            con.execute("ALTER TABLE predictions ADD COLUMN issued_at TIMESTAMP")
            if "ts" in cols:
                con.execute("UPDATE predictions SET issued_at = ts WHERE issued_at IS NULL")
        if "features_ref" not in cols:
            con.execute("ALTER TABLE predictions ADD COLUMN features_ref VARCHAR")
    except Exception as e:
        # swallow harmless errors for temp DBs that don’t have predictions yet
        if "no such table" not in str(e).lower():
            logger.warning("ensure_pred_schema(%s) failed: %s", path, e)
def _pred_cols_and_order(con, base_cols: Sequence[str]) -> tuple[list[str], str]:
    """
    Build a safe SELECT list and ORDER BY column for reading from `predictions`.
    If the table lacks `issued_at`, we alias `ts` as `issued_at` and order by `ts`.
    """
    # If the table doesn't exist yet, just fall back to ts.
    try:
        cols_now = {r[1] for r in con.execute("PRAGMA table_info('predictions')").fetchall()}
    except Exception:
        cols_now = set()

    select_cols = list(base_cols)
    order_col = "issued_at"

    if "issued_at" not in cols_now:
        # Make sure the column exists in the result set expected by callers
        select_cols = [c for c in select_cols if c != "issued_at"] + ["ts AS issued_at"]
        order_col = "ts"

    return select_cols, order_col


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("predictive")


def _resolve_artifact_path(path: str | Path) -> Path:
    return core_resolve_artifact_path(path)


def _calibration_key(symbol: str, horizon_days: int) -> str:
    return f"calib:{symbol.upper()}:{int(horizon_days)}"


def _fit_residual_distribution(sample: Sequence[float], horizon_days: int) -> dict[str, Any]:
    arr = np.asarray(sample, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < CALIBRATION_SAMPLE_MIN:
        raise ValueError("insufficient residual samples for calibration")

    horizon = max(1, int(horizon_days))
    try:
        df, loc, scale = stats.t.fit(arr)
        if not np.isfinite(df) or not np.isfinite(loc) or not np.isfinite(scale) or scale <= 0:
            raise ValueError
    except Exception:
        df = 7.0
        loc = float(np.median(arr))
        scale = float(np.std(arr, ddof=1) or 1e-3)

    try:
        kde = stats.gaussian_kde(arr)
        span = float(np.std(arr, ddof=1))
        if not np.isfinite(span) or span <= 1e-6:
            span = max(1e-4, float(np.max(np.abs(arr))))
        grid_lo = float(arr.min() - 4.0 * span)
        grid_hi = float(arr.max() + 4.0 * span)
        if not np.isfinite(grid_lo) or not np.isfinite(grid_hi) or grid_hi <= grid_lo:
            grid_lo, grid_hi = float(arr.min() - 0.05), float(arr.max() + 0.05)
        grid = np.linspace(grid_lo, grid_hi, CALIBRATION_GRID_SIZE)
        pdf = kde(grid)
        dx = float(grid[1] - grid[0]) if grid.size > 1 else 1.0
        cdf = np.cumsum(pdf) * dx
        if cdf[-1] <= 0:
            raise ValueError
        cdf /= cdf[-1]
        quantiles = [float(np.interp(p, cdf, grid)) for p in (0.05, 0.50, 0.95)]
    except Exception:
        quantiles = [float(np.percentile(arr, q)) for q in (5, 50, 95)]

    sample_std = float(np.std(arr, ddof=1))
    sigma_ann = float(sample_std * math.sqrt(252.0 / horizon))

    return {
        "df": float(df),
        "loc": float(loc),
        "scale": float(scale),
        "q05": float(quantiles[0]),
        "q50": float(quantiles[1]),
        "q95": float(quantiles[2]),
        "sample_std": sample_std,
        "sigma_ann": sigma_ann,
    }


def _sigma_from_calibration(cal: Mapping[str, Any]) -> float | None:
    sigma = cal.get("sigma_ann")
    if sigma is not None:
        try:
            val = float(sigma)
            if math.isfinite(val) and val > 0:
                return val
        except Exception:
            pass
    df = cal.get("df")
    scale = cal.get("scale")
    try:
        df_val = float(df)
        scale_val = float(scale)
        if df_val > 2 and scale_val > 0:
            return float(scale_val * math.sqrt(df_val / (df_val - 2)))
    except Exception:
        return None
    return None


def _export_daily_snapshots(day: date) -> dict[str, str]:
    """
    Write Parquet snapshots for predictions, outcomes, and metrics for the given day.
    Returns a mapping of artifact type to file path.
    """
    outputs: dict[str, str] = {}
    if export_metrics_parquet is not None:
        try:
            metrics_path = EXPORT_ROOT / "metrics_daily" / f"metrics_{day.isoformat()}.parquet"
            export_metrics_parquet(metrics_path, day=day)
            outputs["metrics"] = str(metrics_path)
        except RuntimeError as exc:
            logger.debug("metrics export skipped (runtime): %s", exc)
        except Exception as exc:
            logger.warning("Metrics export failed for %s: %s", day, exc)
    if export_predictions_parquet is not None:
        try:
            preds_path = EXPORT_ROOT / "predictions_daily" / f"predictions_{day.isoformat()}.parquet"
            export_predictions_parquet(
                preds_path,
                day=day,
                limit=100_000,
            )
            outputs["predictions"] = str(preds_path)
        except RuntimeError as exc:
            logger.debug("predictions export skipped (runtime): %s", exc)
        except Exception as exc:
            logger.warning("Predictions export failed for %s: %s", day, exc)
    if export_outcomes_parquet is not None:
        try:
            outcomes_path = EXPORT_ROOT / "outcomes_daily" / f"outcomes_{day.isoformat()}.parquet"
            export_outcomes_parquet(
                outcomes_path,
                day=day,
                limit=100_000,
            )
            outputs["outcomes"] = str(outcomes_path)
        except RuntimeError as exc:
            logger.debug("outcomes export skipped (runtime): %s", exc)
        except Exception as exc:
            logger.warning("Outcomes export failed for %s: %s", day, exc)
    return outputs


def _maybe_upload_to_s3(path: Path) -> str | None:
    bucket = (os.getenv("PT_EXPORT_S3_BUCKET") or "").strip()
    if not bucket:
        return None
    try:
        import boto3  # type: ignore
    except Exception as exc:
        logger.warning(f"S3 upload skipped (boto3 unavailable): {exc}")
        return None
    prefix = (os.getenv("PT_EXPORT_S3_PREFIX") or "").strip("/")
    try:
        relative = path.relative_to(EXPORT_ROOT)
        rel_key = relative.as_posix()
    except Exception:
        rel_key = path.name
    key = f"{prefix}/{rel_key}".strip("/") if prefix else rel_key
    try:
        s3 = boto3.client("s3")
        s3.upload_file(path.as_posix(), bucket, key)
        return f"s3://{bucket}/{key}"
    except Exception as exc:
        logger.warning(f"S3 upload failed for {path}: {exc}")
        return None


def _prune_arrow_partitions(base_dir: Path, cutoff_date: date) -> None:
    try:
        for child in base_dir.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if "=" in name:
                _, value = name.split("=", 1)
            else:
                value = name
            try:
                part_date = date.fromisoformat(value)
            except ValueError:
                continue
            if part_date < cutoff_date:
                shutil.rmtree(child, ignore_errors=True)
    except Exception as exc:
        logger.debug("Arrow partition prune failed for %s: %s", base_dir, exc)


def _calibration_from_store(symbol: str, horizon_days: int) -> dict[str, Any] | None:
    if fs_connect is None or get_calibration_params is None:
        return None
    try:
        con = fs_connect()
        row = get_calibration_params(con, symbol, horizon_days)
        con.close()
        if not row:
            return None
        updated_at = row.get("updated_at")
        if isinstance(updated_at, str):
            try:
                updated_dt = datetime.fromisoformat(updated_at)
            except Exception:
                updated_dt = None
        else:
            updated_dt = updated_at
        iso_ts: str | None = None
        if isinstance(updated_dt, datetime):
            if updated_dt.tzinfo is None:
                updated_dt = updated_dt.replace(tzinfo=timezone.utc)
            age = datetime.now(timezone.utc) - updated_dt
            if age > CALIBRATION_DB_MAX_AGE:
                return None
            iso_ts = updated_dt.isoformat()
        elif isinstance(updated_at, str):
            iso_ts = updated_at
        return {**row, "updated_at": iso_ts}
    except Exception as exc:
        logger.debug("calibration store lookup failed for %s H=%s: %s", symbol, horizon_days, exc)
        return None


def _compute_calibration_params(symbol: str, horizon_days: int) -> dict[str, Any] | None:
    if fs_connect is None:
        return None
    symbol_norm = symbol.upper()
    try:
        con = fs_connect()
        rows = con.execute(
            """
            SELECT p.spot0, o.y
            FROM predictions p
            JOIN outcomes o ON o.run_id = p.run_id
            WHERE p.symbol = ? AND p.horizon_days = ?
              AND p.spot0 IS NOT NULL AND o.y IS NOT NULL
            ORDER BY o.realized_at DESC
            LIMIT ?
            """,
            [symbol_norm, int(horizon_days), CALIBRATION_SAMPLE_LIMIT],
        ).fetchall()
        con.close()
    except Exception as exc:
        logger.debug("calibration residual fetch failed for %s H=%s: %s", symbol_norm, horizon_days, exc)
        return None

    residuals: list[float] = []
    for spot0, realized in rows:
        try:
            s0 = float(spot0)
            y = float(realized)
            if s0 > 0 and y > 0 and math.isfinite(s0) and math.isfinite(y):
                residuals.append(math.log(y / s0))
        except Exception:
            continue

    if len(residuals) < CALIBRATION_SAMPLE_MIN:
        return None

    try:
        fit = _fit_residual_distribution(residuals, horizon_days=horizon_days)
    except Exception as exc:
        logger.debug("calibration fit failed for %s H=%s: %s", symbol_norm, horizon_days, exc)
        return None

    fit.update(
        {
            "sample_n": len(residuals),
            "symbol": symbol_norm,
            "horizon_days": int(horizon_days),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    if fs_connect is not None and upsert_calibration_params is not None:
        try:
            con2 = fs_connect()
            upsert_calibration_params(
                con2,
                [
                    (
                        symbol_norm,
                        int(horizon_days),
                        fit["df"],
                        fit["loc"],
                        fit["scale"],
                        fit["q05"],
                        fit["q50"],
                        fit["q95"],
                        fit["sample_n"],
                        fit["sigma_ann"],
                        datetime.now(timezone.utc),
                    )
                ],
            )
            con2.close()
        except Exception as exc:
            logger.debug("calibration store upsert failed for %s H=%s: %s", symbol_norm, horizon_days, exc)

    return fit


async def _get_calibration_params(symbol: str, horizon_days: int) -> dict[str, Any] | None:
    symbol_norm = symbol.upper()
    key = _calibration_key(symbol_norm, horizon_days)
    if REDIS:
        try:
            cached = await REDIS.get(key)
            if cached:
                try:
                    return json.loads(cached)
                except Exception:
                    pass
        except Exception:
            pass

    cal = _calibration_from_store(symbol_norm, horizon_days)
    if cal is None:
        cal = await asyncio.to_thread(_compute_calibration_params, symbol_norm, horizon_days)
    if cal and REDIS:
        try:
            await REDIS.setex(key, CALIBRATION_TTL_SECONDS, json.dumps(cal))
        except Exception:
            pass
    return cal


def _compute_fallback_quantiles(
    px: Sequence[float],
    prob_up: float,
    horizon_days: int,
) -> tuple[float, float, float] | None:
    try:
        arr_px = np.asarray(px, dtype=float)
        if arr_px.size < 2:
            return None
        spot0 = float(arr_px[-1])
        if not math.isfinite(spot0) or spot0 <= 0:
            return None
        log_rets = np.diff(np.log(arr_px))
        log_rets = log_rets[np.isfinite(log_rets)]
        if log_rets.size < 5:
            return None
        log_rets = winsorize(log_rets)
        mu_d = float(np.mean(log_rets))
        sig_d = float(np.std(log_rets, ddof=1))
        if not math.isfinite(sig_d) or sig_d <= 1e-8:
            return None
        horizon = max(1, int(horizon_days))
        df = 7
        sigma_h = sig_d * math.sqrt(horizon)
        scale = sigma_h
        if df > 2:
            scale = sigma_h / math.sqrt(df / (df - 2))
        scale = max(scale, 1e-6)
        prob_clip = float(np.clip(prob_up, 1e-4, 1 - 1e-4))
        loc_prob = scale * float(stats.t.ppf(prob_clip, df))
        loc_hist = mu_d * horizon
        loc = 0.7 * loc_prob + 0.3 * loc_hist
        rv = stats.t(df=df, loc=loc, scale=scale)
        q05 = float(spot0 * math.exp(float(rv.ppf(0.05))))
        q50 = float(spot0 * math.exp(float(rv.ppf(0.50))))
        q95 = float(spot0 * math.exp(float(rv.ppf(0.95))))
        return q05, q50, q95
    except Exception:
        return None


async def _get_model_meta(symbol: str) -> dict[str, Any] | None:
    sym = (symbol or "").upper().strip()
    if not sym:
        return None
    cached = _MODEL_META_CACHE.get(sym)
    if cached is not None:
        return cached
    if not REDIS:
        return None
    try:
        raw = await REDIS.get(f"model_meta:{sym}")
        if not raw:
            return None
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        data = json.loads(raw)
        _MODEL_META_CACHE[sym] = data
        return data
    except Exception:
        return None

_loader_lock = asyncio.Lock()
REDIS: Redis | None = None
LS_WEBHOOK_SECRET = (os.getenv("LS_WEBHOOK_SECRET") or "").encode("utf-8")
os.environ.setdefault("SERVICE_NAME", "simetrix-api")

app = FastAPI(
    title="Simetrix API",
    version="1.3.0",
    docs_url="/api-docs",
    redoc_url=None,
    redirect_slashes=False,
)
APP = app  # back-compat if any decorator still uses APP
SCOPES = {
    "simulate": "Run and view simulations",
    "admin": "Administrative operations",
    "cron": "Scheduled automation hooks",
}
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", scopes=SCOPES, auto_error=False)
install_observability(app)
class BaselineRateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip docs/metrics/health to avoid noise
        p = request.url.path
        if p.startswith("/metrics") or p.startswith("/health") or p.startswith("/status") or p.startswith("/api-docs") or p.startswith("/ready"):
            return await call_next(request)
        # Admin can be handled per-endpoint; we still set a mild baseline
        try:
            await rate_limit(REDIS, request, scope="base", limit_per_min=BASE_LIMIT_PER_MIN)
        except HTTPException as e:
            log_json("error", msg="rate_limit_block", scope="base", path=p, detail=e.detail)
            return JSONResponse(status_code=e.status_code, content={"ok": False, "error": e.detail})
        return await call_next(request)

app.add_middleware(BaselineRateLimitMiddleware)
# --- Response helpers ---
def ok(data: Any | None = None, **extra: Any) -> dict:
    payload: dict[str, Any] = {"ok": True}
    if data is not None:
        payload["data"] = data
    if extra:
        payload.update(extra)
    return payload

def fail(detail: str, status_code: int = status.HTTP_400_BAD_REQUEST, **extra: Any) -> JSONResponse:
    content: dict[str, Any] = {"ok": False, "error": str(detail)}
    if extra:
        content.update(extra)
    return JSONResponse(status_code=status_code, content=content)

def _create_access_token(user: UserKey, scopes: Sequence[str]) -> str:
    return auth_service.create_access_token(user, scopes)


def _decode_token(token: str) -> dict:
    return dict(auth_service.decode_token(token))


def _admin_credentials_configured() -> bool:
    return bool((settings.admin_username or "").strip()) and bool((settings.admin_password_hash or "").strip())


def _verify_admin_password(candidate: str) -> bool:
    stored = (settings.admin_password_hash or "").strip()
    if not stored:
        return False
    if stored.startswith("$2") and bcrypt_hash is not None:
        try:
            return bool(bcrypt_hash.verify(candidate, stored))
        except ValueError:
            return False
    if stored.startswith("sha256:"):
        expected = stored.split(":", 1)[1]
        digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        return hmac.compare_digest(digest, expected)
    if stored.startswith("plain:"):
        expected = stored.split(":", 1)[1]
        return hmac.compare_digest(candidate, expected)
    # Unknown hash scheme; fail closed
    return False


def _admin_session_claims(username: str) -> dict[str, Any]:
    issued_at = datetime.now(timezone.utc)
    expires_at = issued_at + timedelta(minutes=int(settings.admin_session_ttl_minutes))
    return {
        "sub": username,
        "type": "admin",
        "scopes": ["admin"],
        "iat": int(issued_at.timestamp()),
        "exp": int(expires_at.timestamp()),
        "jti": uuid4().hex,
    }


def _encode_admin_session(username: str) -> str:
    claims = _admin_session_claims(username)
    return jwt.encode(claims, settings.admin_session_secret, algorithm=JWT_ALGORITHM)


def _set_admin_session_cookie(response: Response, token: str) -> None:
    max_age = int(settings.admin_session_ttl_minutes) * 60
    response.set_cookie(
        key=settings.admin_session_cookie,
        value=token,
        max_age=max_age,
        expires=max_age,
        path="/",
        secure=bool(settings.admin_cookie_secure),
        httponly=True,
        samesite="Strict",
    )


def _clear_admin_session_cookie(response: Response) -> None:
    response.delete_cookie(settings.admin_session_cookie, path="/")


def _decode_admin_session(token: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(token, settings.admin_session_secret, algorithms=[JWT_ALGORITHM])
    except JWTError as exc:
        raise HTTPException(status_code=401, detail="invalid_admin_session") from exc
    if payload.get("type") != "admin":
        raise HTTPException(status_code=403, detail="invalid_admin_session_type")
    return payload


def _get_admin_session(request: Request, *, required: bool = False) -> dict[str, Any] | None:
    token = request.cookies.get(settings.admin_session_cookie)
    if not token:
        if required:
            raise HTTPException(status_code=401, detail="admin_session_required")
        return None
    try:
        payload = _decode_admin_session(token)
    except HTTPException as exc:
        if required:
            raise exc
        return None
    request.state.admin_session = payload
    return payload


def _admin_actor(request: Request) -> str:
    if hasattr(request.state, "admin_session"):
        session = getattr(request.state, "admin_session", {}) or {}
        actor = str(session.get("sub") or "admin")
        return actor
    return str(getattr(request.state, "caller_id", "admin"))

async def _authenticate_bearer(
    request: Request,
    required_scopes: Sequence[str],
    token: str | None,
) -> UserKey:
    return await auth_service.authenticate_bearer(request, required_scopes, token)

async def require_user(
    request: Request,
    security_scopes: SecurityScopes,
    token: str | None = Depends(oauth2_scheme),
) -> UserKey:
    required = security_scopes.scopes or ["simulate"]
    return await _authenticate_bearer(request, required, token)

# --- Auth dependency (define BEFORE any route uses it)
async def require_key(
    request: Request,
    security_scopes: SecurityScopes,
    token: str | None = Depends(oauth2_scheme),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_pt_key: str | None = Header(default=None, alias="x-pt-key"),
    api_key: str | None = Query(default=None),
) -> bool:
    required_scopes = security_scopes.scopes or ["simulate"]
    requires_admin = any(scope == "admin" for scope in required_scopes)

    if requires_admin:
        session_payload = _get_admin_session(
            request,
            required=bool(request.cookies.get(settings.admin_session_cookie)),
        )
        if session_payload:
            actor = str(session_payload.get("sub") or "admin")
            request.state.caller_id = f"admin:{actor}"
            request.state.plan = "inst"
            request.state.scopes = list(sorted(set(session_payload.get("scopes", []) or ["admin"])))
            request.state.user = None
            request.state.auth_source = "admin_session"
            return True
        if request.cookies.get(settings.admin_session_cookie):
            raise HTTPException(status_code=401, detail="admin_session_invalid")

    if token:
        await auth_service.authenticate_bearer(request, required_scopes, token)
        return True

    provided = (x_api_key or x_pt_key or api_key or "").strip()
    if provided:
        return await auth_service.authorize_api_key(request, provided)

    if auth_service.open_access_allowed(required_scopes):
        auth_service.apply_open_access(request)
        return True

    raise HTTPException(status_code=401, detail="Invalid or missing credentials")
@app.post("/auth/token", summary="Obtain an OAuth2 access token")
async def issue_token(form_data: OAuth2PasswordRequestForm = Depends()):
    return auth_service.issue_token(
        form_data.username,
        form_data.password,
        form_data.scopes or [],
    )

@app.get("/auth/me", summary="Return current authenticated user context")
async def auth_me(request: Request, user: UserKey = Security(require_user, scopes=["simulate"])):
    granted = getattr(request.state, "scopes", [])
    return {
        "user": {
            "id": user.id,
            "email": user.email,
            "plan": user.plan,
            "scopes": list(all_scopes_for(user)),
        },
        "granted_scopes": list(granted),
    }

async def ensure_tf():
    global TF_AVAILABLE
    if TF_AVAILABLE:
        return
    async with _loader_lock:
        if TF_AVAILABLE:
            return
        # run the heavy import without blocking the event loop
        await asyncio.to_thread(load_tensorflow)

async def ensure_sb3():
    global SB3_AVAILABLE
    if SB3_AVAILABLE:
        return
    async with _loader_lock:
        if SB3_AVAILABLE:
            return
        await asyncio.to_thread(load_stable_baselines3)

async def ensure_arima():
    global ARIMA_AVAILABLE
    if ARIMA_AVAILABLE:
        return
    async with _loader_lock:
        if ARIMA_AVAILABLE:
            return
        await asyncio.to_thread(load_arima)
@app.post("/admin/warm")
async def warm(_ok: bool = Security(require_key, scopes=["admin"])):
    await asyncio.gather(
        ensure_tf(),
        ensure_arima(),
        # ensure_sb3(),  # include if you actually need RL on web node
    )
    return {"ok": True, "tf": TF_AVAILABLE, "arima": ARIMA_AVAILABLE, "sb3": SB3_AVAILABLE}


@app.post("/admin/login")
async def admin_login(
    request: Request,
    response: Response,
    payload: dict = Body(...),
):
    data = AdminLoginRequest.model_validate(payload)
    if not _admin_credentials_configured():
        raise HTTPException(status_code=503, detail="admin_login_disabled")
    await rate_limit(REDIS, request, "admin_login", int(settings.admin_login_rpm))
    configured = (settings.admin_username or "").strip()
    submitted = (data.username or "").strip()
    password = data.password or ""
    if (submitted.lower() != configured.lower()) or (not _verify_admin_password(password)):
        reason = "invalid_credentials"
        log_json(
            "warning",
            msg="admin_login_failed",
            user=submitted or "<empty>",
            reason=reason,
            remote=request.client.host if request.client else None,
        )
        raise HTTPException(status_code=401, detail="invalid_credentials")

    token = _encode_admin_session(configured)
    _set_admin_session_cookie(response, token)
    response.headers["Cache-Control"] = "no-store"
    log_json(
        "info",
        msg="admin_login_success",
        user=configured,
        remote=request.client.host if request.client else None,
    )
    return {"status": "ok"}


@app.post("/admin/logout")
async def admin_logout(request: Request, response: Response):
    actor = None
    try:
        session = _get_admin_session(request, required=False)
        if session:
            actor = session.get("sub")
    except HTTPException:
        actor = None
    _clear_admin_session_cookie(response)
    response.headers["Cache-Control"] = "no-store"
    log_json(
        "info",
        msg="admin_logout",
        user=actor or "unknown",
        remote=request.client.host if request.client else None,
    )
    return {"status": "ok"}


@app.post("/admin/refresh")
async def admin_refresh(
    request: Request,
    response: Response,
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    session = _get_admin_session(request, required=True) or {}
    actor = str(session.get("sub") or (settings.admin_username or "admin"))
    token = _encode_admin_session(actor)
    _set_admin_session_cookie(response, token)
    response.headers["Cache-Control"] = "no-store"
    log_json(
        "info",
        msg="admin_session_refreshed",
        user=actor,
        remote=request.client.host if request.client else None,
    )
    return {"status": "ok"}
    
@app.get("/models")
async def models(_ok: bool = Depends(require_key)):
    await ensure_tf()
    if not TF_AVAILABLE:
        raise HTTPException(503, "TensorFlow unavailable")
    # return something meaningful for your UI
    return {"models": ["tf_lstm", "tf_gru"], "tf": True}

@app.get("/")
def root():
    return {"ok": True, "service": "simetrix-api"}

# --- Health endpoints ---
@app.get("/healthz")
def healthz():
    # Liveness only: process is up and routing works
    return {"ok": True}

@app.get("/ready")
async def ready():
    # Readiness/diagnostics: safe, non-fatal checks
    redis_ok = False
    try:
        if REDIS:
            # Bound the ping so it can’t hang the endpoint
            redis_ok = await asyncio.wait_for(REDIS.ping(), timeout=0.25)
    except Exception:
        redis_ok = False

    key = (settings.polygon_key or "").strip()
    key_mode = "real" if key else "none"

    await _trigger_daily_quant_from_health(reason="health_probe")

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


# Lazy loaders for optional libraries
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
# Globals for models from learners
ENSEMBLE = None
EXP_W = None

def load_learners():
    """Local learners (lightweight)"""
    global ENSEMBLE, EXP_W
    if ENSEMBLE is None:
        from .learners import OnlineLinear, ExpWeights
        ENSEMBLE = OnlineLinear(lr=0.05, l2=1e-4)
        EXP_W = ExpWeights(eta=2.0)

load_learners()
# --- local modules (feature store with back-compat)
try:
    from .feature_store import connect as fs_connect, DB_PATH as FS_DB_PATH
    from .feature_store import (
        get_recent_coverage,
        get_recent_mdape,
        log_ingest_event,
        insert_news,
        insert_earnings,
        upsert_macro,
        compute_and_upsert_metrics_daily as _rollup,
        upsert_calibration_params,
        get_calibration_params,
        export_predictions_parquet,
        export_outcomes_parquet,
        export_metrics_parquet,
        fetch_metrics_daily_arrow,
    )
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
        from feature_store import connect as fs_connect, DB_PATH as FS_DB_PATH  # type: ignore
        from feature_store import (  # type: ignore
            get_recent_coverage,
            get_recent_mdape,
            log_ingest_event,
            insert_news,
            insert_earnings,
            upsert_macro,
            compute_and_upsert_metrics_daily as _rollup,
            upsert_calibration_params,
            get_calibration_params,
            export_predictions_parquet,
            export_outcomes_parquet,
            export_metrics_parquet,
            fetch_metrics_daily_arrow,
        )
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
        log_ingest_event = None  # type: ignore
        insert_news = None  # type: ignore
        insert_earnings = None  # type: ignore
        upsert_macro = None  # type: ignore
        _fs_ins = None
        _fs_log = None
        _rollup = None
        export_predictions_parquet = None  # type: ignore
        export_outcomes_parquet = None  # type: ignore
        export_metrics_parquet = None  # type: ignore
        FS_DB_PATH = None  # type: ignore
        logger.warning("feature_store unavailable; logging to FS disabled")

# Choose the best available implementation
if _fs_ins is not None:
    _FS_LOG_IMPL = _fs_ins  # prefer explicit insert when present
elif _fs_log is not None:
    _FS_LOG_IMPL = _fs_log
else:
    _FS_LOG_IMPL = None  # feature store not wired for this build

try:
    from .labeler import label_mature_predictions
except Exception:
    try:
        from labeler import label_mature_predictions  # type: ignore
    except Exception:
        label_mature_predictions = None  # type: ignore

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


def _determine_news_provider() -> str:
    provider = (os.getenv("PT_NEWS_PROVIDER") or "").strip().lower()
    if provider:
        return provider
    if settings.news_api_key:
        return "newsapi"
    return "polygon"


def _as_utc_datetime(value: datetime | date | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)
    return None


async def get_sentiment_features(symbol: str, days: int = 7) -> dict:
    base = {"avg_sent_7d": 0.0, "last24h": 0.0, "n_news": 0}
    if not fs_connect:
        return base
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return base
    window_days = int(max(1, days))
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    try:
        con = fs_connect()
    except Exception as exc:
        logger.debug("get_sentiment_features: fs_connect failed: %s", exc)
        return base
    try:
        rows = con.execute(
            """
            SELECT ts, sentiment
            FROM news_articles
            WHERE symbol = ? AND ts >= ?
            ORDER BY ts DESC
            """,
            [symbol, cutoff],
        ).fetchall()
    except Exception as exc:
        logger.debug("get_sentiment_features: query failed for %s: %s", symbol, exc)
        rows = []
    finally:
        try:
            con.close()
        except Exception:
            pass
    if not rows:
        return base

    now = datetime.now(timezone.utc)
    vals = []
    last24_vals = []
    for ts, sentiment in rows:
        val = float(sentiment or 0.0)
        vals.append(val)
        ts_dt = _as_utc_datetime(ts)
        if ts_dt is None:
            continue
        if (now - ts_dt).total_seconds() <= 86400:
            last24_vals.append(val)

    base["avg_sent_7d"] = float(np.mean(vals)) if vals else 0.0
    base["last24h"] = float(np.mean(last24_vals)) if last24_vals else 0.0
    base["n_news"] = len(vals)
    return base


async def get_earnings_features(symbol: str) -> dict:
    out = {
        "surprise_last": 0.0,
        "guidance_delta": 0.0,
        "days_since_earn": None,
        "days_to_next": None,
    }
    if not fs_connect:
        return out
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return out
    try:
        con = fs_connect()
    except Exception as exc:
        logger.debug("get_earnings_features: fs_connect failed: %s", exc)
        return out
    try:
        last = con.execute(
            """
            SELECT report_date, surprise, guidance_delta
            FROM earnings
            WHERE symbol = ?
            ORDER BY report_date DESC
            LIMIT 1
            """,
            [symbol],
        ).fetchone()
    except Exception as exc:
        logger.debug("get_earnings_features: query failed for %s: %s", symbol, exc)
        last = None
    finally:
        try:
            con.close()
        except Exception:
            pass

    if last:
        report_date = last[0]
        surprise = last[1]
        guidance_delta = last[2]
        if surprise is not None:
            out["surprise_last"] = float(surprise)
        if guidance_delta is not None:
            out["guidance_delta"] = float(guidance_delta)
        rd = report_date
        if isinstance(rd, datetime):
            rd = rd.date()
        if isinstance(rd, date):
            out["days_since_earn"] = (datetime.now(timezone.utc).date() - rd).days
    return out


async def get_macro_features() -> dict:
    base = {"rff": None, "cpi_yoy": None, "u_rate": None}
    if not fs_connect:
        return base
    try:
        con = fs_connect()
    except Exception as exc:
        logger.debug("get_macro_features: fs_connect failed: %s", exc)
        return base
    try:
        row = con.execute(
            """
            SELECT rff, cpi_yoy, u_rate
            FROM macro_daily
            ORDER BY as_of DESC
            LIMIT 1
            """
        ).fetchone()
    except Exception as exc:
        logger.debug("get_macro_features: query failed: %s", exc)
        row = None
    finally:
        try:
            con.close()
        except Exception:
            pass
    if not row:
        return base
    return {
        "rff": float(row[0]) if row[0] is not None else None,
        "cpi_yoy": float(row[1]) if row[1] is not None else None,
        "u_rate": float(row[2]) if row[2] is not None else None,
    }


def _normalize_store_symbol(symbol: str) -> str:
    s = (symbol or "").strip().upper()
    if not s:
        return s
    if s.startswith("X:") and s.endswith("USD"):
        return f"{s[2:-3]}-USD"
    if s.endswith("USD") and "-" not in s and not s.startswith("X:"):
        return f"{s[:-3]}-USD"
    return s


def _first_number(values: Iterable[Any]) -> float | None:
    for value in values:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            val = value.strip()
            if not val or val == ".":
                continue
            try:
                return float(val)
            except Exception:
                continue
    return None


def _news_ts(value: Any, default: datetime) -> datetime:
    if isinstance(value, datetime):
        return _as_utc_datetime(value) or default
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return default
        if txt.endswith("Z"):
            txt = txt[:-1] + "+00:00"
        try:
            return _as_utc_datetime(datetime.fromisoformat(txt)) or default
        except Exception:
            pass
    return default


async def _news_rows_from_newsapi(symbol: str, since: datetime, limit: int, api_key: str) -> list[tuple]:
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": symbol,
        "from": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": int(max(1, min(limit, 100))),
    }
    headers = {"X-Api-Key": api_key}
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json() or {}
    items = data.get("articles", []) or []
    rows: list[tuple] = []
    seen: set[tuple] = set()
    for item in items:
        ts = _news_ts(item.get("publishedAt"), since)
        if ts < since:
            continue
        url_item = (item.get("url") or "").strip()
        dedupe_key = (ts.isoformat(), url_item)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        source_name = ""
        src_obj = item.get("source") or {}
        if isinstance(src_obj, dict):
            source_name = (src_obj.get("name") or "").strip()
        title = (item.get("title") or "").strip()
        summary = (item.get("description") or item.get("content") or "").strip()
        rows.append((symbol, ts, source_name or "NewsAPI", title, url_item, summary, None))
        if len(rows) >= limit:
            break
    return rows


async def _news_rows_from_polygon(symbol: str, since: datetime, limit: int, api_key: str) -> list[tuple]:
    url = "https://api.polygon.io/v2/reference/news"
    params = {
        "ticker": symbol,
        "limit": int(max(1, min(limit, 100))),
        "order": "desc",
        "published_utc.gte": since.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "apiKey": api_key,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json() or {}
    items = data.get("results", []) or []
    rows: list[tuple] = []
    for item in items:
        try:
            tickers = [str(t).upper() for t in (item.get("tickers") or [])]
        except Exception:
            tickers = []
        if symbol.upper() not in tickers:
            continue
        ts = _news_ts(item.get("published_utc"), since)
        if ts < since:
            continue
        publisher = ""
        pub_obj = item.get("publisher")
        if isinstance(pub_obj, dict):
            publisher = (pub_obj.get("name") or "").strip()
        title = (item.get("title") or "").strip()
        url_item = (item.get("article_url") or item.get("url") or "").strip()
        summary = (item.get("description") or item.get("excerpt") or "").strip()
        rows.append((symbol, ts, publisher or "Polygon", title, url_item, summary, None))
        if len(rows) >= limit:
            break
    return rows


async def _fetch_news_articles(symbol: str, since: datetime, provider: str, limit: int = 100) -> list[tuple]:
    provider = (provider or "polygon").strip().lower()
    limit = int(max(1, min(limit, 200)))
    if provider == "newsapi":
        api_key = settings.news_api_key or os.getenv("NEWS_API_KEY", "").strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="NewsAPI key missing (set PT_NEWS_API_KEY).")
        return await _news_rows_from_newsapi(symbol, since, limit, api_key)
    if provider == "polygon":
        key = _poly_key()
        if not key:
            raise HTTPException(status_code=400, detail="Polygon key missing for news ingest.")
        return await _news_rows_from_polygon(symbol, since, limit, key)
    raise HTTPException(status_code=400, detail=f"Unsupported news provider '{provider}'.")


def _determine_earnings_provider() -> str:
    provider = (settings.earnings_source or os.getenv("PT_EARNINGS_PROVIDER", "polygon")).strip().lower()
    return provider or "polygon"


async def _fetch_earnings_polygon(symbol: str, lookback_days: int, limit: int = 16) -> list[tuple]:
    key = _poly_key()
    if not key:
        raise HTTPException(status_code=400, detail="Polygon key missing for earnings ingest.")
    url = "https://api.polygon.io/v3/reference/financials"
    params = {
        "ticker": symbol,
        "timeframe": "quarterly",
        "limit": int(max(1, min(limit, 50))),
        "order": "desc",
        "sort": "reportPeriod",
        "apiKey": key,
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json() or {}
    results = data.get("results", []) or []
    since_date = datetime.now(timezone.utc).date() - timedelta(days=int(max(1, lookback_days)))
    rows: list[tuple] = []
    for item in results:
        rep_raw = item.get("reportPeriod") or item.get("calendarDate")
        try:
            rep_date = date.fromisoformat(str(rep_raw))
        except Exception:
            continue
        if rep_date < since_date:
            continue
        eps_val = _first_number([
            item.get("eps"),
            item.get("epsDiluted"),
            (item.get("earnings") or {}).get("eps"),
            (item.get("incomeStatement") or {}).get("basicEPS"),
            (item.get("incomeStatement") or {}).get("dilutedEPS"),
        ])
        est_val = _first_number([
            item.get("estimateEPS"),
            (item.get("earnings") or {}).get("epsEstimate"),
            (item.get("analystEstimates") or {}).get("epsEstimate"),
        ])
        surprise = None
        if eps_val is not None and est_val not in (None, 0.0):
            denom = abs(est_val)
            if denom > 1e-9:
                surprise = (eps_val - est_val) / denom
        revenue_val = _first_number([
            item.get("revenue"),
            (item.get("incomeStatement") or {}).get("revenue"),
            (item.get("incomeStatement") or {}).get("totalRevenue"),
        ])
        guidance_val = _first_number([
            (item.get("guidance") or {}).get("revenue"),
            item.get("guidanceRevenue"),
        ])
        guidance_delta = None
        if guidance_val is not None and revenue_val is not None:
            try:
                guidance_delta = float(guidance_val) - float(revenue_val)
            except Exception:
                guidance_delta = None
        rows.append((
            symbol,
            rep_date,
            eps_val,
            surprise,
            revenue_val,
            guidance_delta,
        ))
        if len(rows) >= limit:
            break
    return rows


async def _fetch_earnings_rows(symbol: str, lookback_days: int, provider: str, limit: int = 16) -> list[tuple]:
    provider = provider.strip().lower()
    if provider == "polygon":
        return await _fetch_earnings_polygon(symbol, lookback_days, limit=limit)
    raise HTTPException(status_code=400, detail=f"Unsupported earnings provider '{provider}'.")


async def _fred_fetch_series(series_id: str, api_key: str, limit: int) -> list[tuple[date, float]]:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": int(max(1, limit)),
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(20.0)) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json() or {}
    obs = data.get("observations", []) or []
    rows: list[tuple[date, float]] = []
    for item in obs:
        val = (item.get("value") or "").strip()
        if not val or val == ".":
            continue
        try:
            value = float(val)
        except Exception:
            continue
        try:
            d = date.fromisoformat(str(item.get("date")))
        except Exception:
            continue
        rows.append((d, value))
    return rows


async def _fetch_macro_rows(provider: str) -> list[tuple]:
    provider = (provider or "fred").strip().lower()
    if provider != "fred":
        raise HTTPException(status_code=400, detail=f"Unsupported macro provider '{provider}'.")
    api_key = (os.getenv("PT_FRED_API_KEY") or os.getenv("FRED_API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="FRED API key missing (set PT_FRED_API_KEY).")
    rff_series, cpi_series, u_series = await asyncio.gather(
        _fred_fetch_series("DGS3MO", api_key, limit=15),
        _fred_fetch_series("CPIAUCSL", api_key, limit=30),
        _fred_fetch_series("UNRATE", api_key, limit=15),
    )

    def _latest(series: list[tuple[date, float]]) -> tuple[date | None, float | None]:
        series_sorted = sorted(series, key=lambda x: x[0], reverse=True)
        for d, v in series_sorted:
            return d, v
        return (None, None)

    rff_date, rff_val = _latest(rff_series)
    u_date, u_val = _latest(u_series)

    cpi_sorted = sorted(cpi_series, key=lambda x: x[0], reverse=True)
    cpi_date, cpi_val = _latest(cpi_sorted)
    prev_val = None
    if cpi_date:
        for d, v in cpi_sorted[1:]:
            delta_months = (cpi_date.year - d.year) * 12 + (cpi_date.month - d.month)
            if delta_months >= 12:
                prev_val = v
                break
    cpi_yoy = None
    if cpi_val is not None and prev_val not in (None, 0.0):
        denom = abs(prev_val)
        if denom > 1e-9:
            cpi_yoy = ((cpi_val - prev_val) / denom) * 100.0

    candidates = [d for d in (rff_date, cpi_date, u_date) if d is not None]
    if not candidates:
        return []
    as_of = max(candidates)
    return [(as_of, rff_val, cpi_yoy, u_val)]


def _load_price_series_with_dates(
    symbol: str,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> list[tuple[date, float]]:
    if not fs_connect:
        return []
    app_sym = _normalize_store_symbol(symbol)
    try:
        con = fs_connect()
    except Exception as exc:
        logger.debug("load_price_series: fs_connect failed for %s: %s", symbol, exc)
        return []

    params: list[Any] = [app_sym]
    clauses: list[str] = []
    if start:
        clauses.append("ts >= ?")
        params.append(start.isoformat())
    if end:
        clauses.append("ts <= ?")
        params.append(end.isoformat())

    where = ""
    if clauses:
        where = " AND " + " AND ".join(clauses)

    try:
        rows = con.execute(
            f"SELECT ts, close FROM bars_daily WHERE symbol = ?{where} ORDER BY ts ASC",
            params,
        ).fetchall()
    except Exception as exc:
        logger.debug("load_price_series query failed for %s: %s", symbol, exc)
        rows = []
    finally:
        try:
            con.close()
        except Exception:
            pass

    out: list[tuple[date, float]] = []
    for ts_val, close_val in rows:
        if close_val is None:
            continue
        dt_val: date | None
        if isinstance(ts_val, datetime):
            dt_val = ts_val.date()
        elif isinstance(ts_val, date):
            dt_val = ts_val
        elif isinstance(ts_val, str):
            try:
                dt_val = datetime.fromisoformat(ts_val.replace("Z", "+00:00")).date()
            except Exception:
                try:
                    dt_val = date.fromisoformat(ts_val[:10])
                except Exception:
                    dt_val = None
        else:
            dt_val = None
        if dt_val is None:
            continue
        out.append((dt_val, float(close_val)))
    return out
# DUCK utils - try db.duck first, fall back to duck
try:
    from .db.duck import (
        init_schema,
        insert_prediction,
        matured_predictions_now,
        insert_outcome,
        recent_predictions,
        DB_PATH as CORE_DB_PATH,
    )
except Exception:
    from .duck import (  # type: ignore
        init_schema,
        insert_prediction,
        matured_predictions_now,
        insert_outcome,
        recent_predictions,
    )
    CORE_DB_PATH = os.getenv("PT_DUCKDB_PATH", str((DATA_ROOT / "pt.duckdb").resolve()))

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
try:
    applied = run_all_migrations()
    if applied:
        logger.info("Migrations applied: %s", applied)
except Exception:
    logger.exception("Failed to execute startup migrations")

# Validation tuning knobs (parsed from settings; fall back to sane defaults)
def _parse_int_list(raw: str) -> list[int]:
    vals: list[int] = []
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            v = int(token)
        except ValueError:
            continue
        if v > 0:
            vals.append(v)
    return vals

VALIDATION_LOOKBACKS = _parse_int_list(settings.validation_lookbacks) or [20, 60, 120]
VALIDATION_TARGET_MAPE = float(settings.validation_target_mape or 5.0)
VALIDATION_MAX_SAMPLES = int(max(20, min(settings.validation_max_samples, 500)))
VALIDATION_BARS_PER_YEAR = 252.0

BACKUP_KEEP = max(1, int(settings.backup_keep))
BACKUP_INTERVAL_SECONDS = max(3600, int(settings.backup_interval_minutes) * 60)
BACKUP_TARGETS: list[tuple[str, Path]] = []
if "CORE_DB_PATH" in globals() and CORE_DB_PATH:
    BACKUP_TARGETS.append(("core", Path(CORE_DB_PATH).expanduser()))
if "FS_DB_PATH" in globals() and FS_DB_PATH:
    BACKUP_TARGETS.append(("feature_store", Path(FS_DB_PATH).expanduser()))

def _latest_mc_metric(symbol: str, horizon_days: int) -> dict | None:
    return service_latest_mc_metric(symbol, horizon_days)

# -----------------------------------------------------------------------------
# Helpers / config (independent)
# -----------------------------------------------------------------------------
# --- validator core ---
async def _validate_mc_paths(symbols: list[str], days: int, n_paths: int) -> dict:
    """
    Run roll-forward Monte Carlo validation for each symbol and persist tuning artifacts.
    """
    try:
        from .feature_store import upsert_mc_params, insert_mc_metrics  # type: ignore
    except Exception:
        from feature_store import upsert_mc_params, insert_mc_metrics  # type: ignore

    today = datetime.now(timezone.utc).date()
    as_of = today.isoformat()
    seeded_by = "symbol|horizon"

    results: list[dict[str, Any]] = []
    params_rows: list[tuple[str, float, float, int, int]] = []
    metric_rows: list[tuple[Any, ...]] = []

    for sym in [s.strip().upper() for s in symbols if s.strip()]:
        try:
            window_days = max(400, days + max(VALIDATION_LOOKBACKS) + 50)
            px = await _fetch_cached_hist_prices(sym, window_days=window_days, redis=REDIS)
            arr = np.asarray([p for p in px if isinstance(p, (int, float)) and math.isfinite(p)], float)
        except Exception as exc:
            results.append({"symbol": sym, "skipped": True, "reason": f"history_fetch_failed:{exc}"})
            continue

        if arr.size < (days + min(VALIDATION_LOOKBACKS) + 5):
            results.append({"symbol": sym, "skipped": True, "reason": "insufficient_history"})
            continue

        try:
            tune = rollforward_validation(
                arr,
                horizon_days=days,
                lookbacks=VALIDATION_LOOKBACKS,
                n_paths=n_paths,
                target_mape=VALIDATION_TARGET_MAPE,
                max_samples=VALIDATION_MAX_SAMPLES,
                bars_per_year=VALIDATION_BARS_PER_YEAR,
            )
        except ValueError as exc:
            results.append({"symbol": sym, "skipped": True, "reason": str(exc)})
            continue

        best = tune.best
        mu_daily = float(best.mu_ann / VALIDATION_BARS_PER_YEAR)
        sigma_daily = float(best.sigma_ann / math.sqrt(VALIDATION_BARS_PER_YEAR))

        params_rows.append((sym, mu_daily, sigma_daily, int(best.lookback), int(best.lookback)))

        variants = [
            {
                "lookback": r.lookback,
                "samples": r.samples,
                "mape": r.mape,
                "mdape": r.mdape,
                "mu_ann": r.mu_ann,
                "sigma_ann": r.sigma_ann,
            }
            for r in tune.results
        ]

        results.append({
            "symbol": sym,
            "horizon_days": int(days),
            "mape": best.mape,
            "mdape": best.mdape,
            "n": int(best.samples),
            "mu": mu_daily,
            "sigma": sigma_daily,
            "mu_ann": best.mu_ann,
            "sigma_ann": best.sigma_ann,
            "lookback": int(best.lookback),
            "recommended_n_paths": int(tune.recommended_n_paths),
            "target_mape": VALIDATION_TARGET_MAPE,
            "candidates": variants,
        })

        metric_rows.append((
            as_of,
            sym,
            int(days),
            float(best.mape),
            float(best.mdape),
            int(best.samples),
            mu_daily,
            sigma_daily,
            int(tune.recommended_n_paths),
            int(best.lookback),
            int(best.lookback),
            seeded_by,
        ))

    con = fs_connect()
    try:
        for (sym, mu_final, sig_final, lb_mu, lb_sig) in params_rows:
            upsert_mc_params(con, sym, mu_final, sig_final, lb_mu, lb_sig)
        if metric_rows:
            insert_mc_metrics(con, metric_rows)
    finally:
        con.close()

    return {
        "as_of": as_of,
        "horizon_days": int(days),
        "n_paths": int(n_paths),
        "target_mape": VALIDATION_TARGET_MAPE,
        "lookbacks": VALIDATION_LOOKBACKS,
        "items": results,
    }

async def _backup_once() -> None:
    if not BACKUP_TARGETS:
        return
    for name, db_path in BACKUP_TARGETS:
        try:
            dest = await asyncio.to_thread(create_duckdb_backup, db_path, BACKUP_DIR, keep=BACKUP_KEEP)
            log_json("info", msg="duckdb_backup_ok", db=name, dest=str(dest))
        except FileNotFoundError:
            log_json("warning", msg="duckdb_backup_missing", db=name, path=str(db_path))
        except Exception as exc:
            log_json("error", msg="duckdb_backup_fail", db=name, error=str(exc))

async def _backup_loop() -> None:
    if not BACKUP_TARGETS or BACKUP_INTERVAL_SECONDS <= 0:
        return
    await asyncio.sleep(random.uniform(5.0, 25.0))
    while True:
        await _backup_once()
        await asyncio.sleep(BACKUP_INTERVAL_SECONDS)


# --- routes ---
@app.post("/admin/backup", summary="Trigger an immediate DuckDB backup")
async def admin_backup(_ok: bool = Security(require_key, scopes=["admin"])):
    await _backup_once()
    return {"ok": True, "targets": [name for name, _ in BACKUP_TARGETS]}

@app.post("/admin/validate/mc")
async def admin_validate_mc(
    days: int = Query(30, ge=1, le=365),
    symbols: str | None = Query(None, description="Comma-separated, defaults to watchlist"),
    n_paths: int = Query(4000, ge=500, le=200000),
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    # default universe: any list you keep in env/redis; fallback to a small set
    syms = [s.strip() for s in (symbols.split(",") if symbols else ["SPY","QQQ","BTC-USD","ETH-USD"])]
    out = await _validate_mc_paths(syms, days=days, n_paths=n_paths)
    return {"ok": True, "data": out}

@app.get("/v1/metrics/mc")
async def get_mc_metrics_api(
    symbol: str | None = None,
    limit: int = Query(200, ge=1, le=1000),
):
    try:
        from .feature_store import get_mc_metrics  # type: ignore
    except Exception:
        from feature_store import get_mc_metrics  # type: ignore

    con = fs_connect()
    try:
        rows = get_mc_metrics(con, symbol.strip().upper() if symbol else None, limit=limit)
    finally:
        con.close()
    return {"ok": True, "data": rows}
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

async def _model_key(symbol: str) -> str:
    return f"model:{symbol.upper()}"

TOP_STOCKS = [
    'NVDA', 'MSFT', 'AAPL', 'AMZN', 'GOOGL', 'META', 'RIOT', 'KR', 'TSM', 'AVGO',
    'TSLA', 'WMT', 'JPM', 'V', 'UNH', 'XOM', 'MA', 'PG', 'JNJ', 'COST',
    'HD', 'ASML', 'CVX', 'ABBV', 'TMUS', 'MRK', 'LLY', 'WFC', 'NFLX', 'AMD',
    'KO', 'BAC', 'CRM', 'ABT', 'DHR', 'TXN', 'LIN', 'ACN', 'QCOM', 'PM',
    'NEE', 'COP', 'ORCL', 'GE', 'AMGN', 'T', 'SPGI', 'UBER', 'ISRG', 'RTX',
    'VZ', 'PFE', 'ABNB', 'C', 'ETN', 'UNP', 'IBM', 'SYK', 'BSX', 'MU',
    'CAT', 'SCHW', 'KLAC', 'TJX', 'DE', 'LMT', 'MDT', 'ADP', 'GILD', 'ZTS',
    'CB', 'LOW', 'HON', 'USB', 'INTU', 'PGR', 'BKNG', 'AXP', 'GS', 'MMC',
    'BLK', 'AMT', 'PLD', 'SBUX', 'CMG', 'BX', 'REGN', 'CBRE', 'SNPS', 'CDNS',
    'ICE', 'PANW', 'MELI', 'ADI', 'MDLZ', 'MO', 'CSX', 'BMY', 'KL', 'STZ'
]

TOP_CRYPTOS = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'SOL-USD', 'DOGE-USD', 'TRX-USD', 'ADA-USD', 'HYPE-USD', 'LINK-USD',
    'XLM-USD', 'BCH-USD', 'SUI-USD', 'AVAX-USD', 'LEO-USD', 'HBAR-USD', 'LTC-USD', 'SHIB-USD', 'MNT-USD', 'TON-USD',
    'XMR-USD', 'CRO-USD', 'DOT-USD', 'UNI-USD', 'TAO-USD', 'OKB-USD', 'AAVE-USD', 'ZEC-USD', 'BGB-USD', 'PEPE-USD',
    'NEAR-USD', 'ENA-USD', 'ASTER-USD', 'APT-USD', 'ETC-USD', 'ONDO-USD', 'POL-USD', 'WLD-USD', 'ICP-USD', 'ARB-USD',
    'ALGO-USD', 'ATOM-USD', 'KAS-USD', 'VET-USD', 'PENGU-USD', 'FLR-USD', 'RENDER-USD', 'SKY-USD', 'GT-USD', 'SEI-USD',
    'PUMP-USD', 'CAKE-USD', 'JUP-USD', 'FIL-USD', 'IMX-USD', 'SPX-USD', 'XDC-USD', 'QNT-USD', 'INJ-USD', 'TIA-USD',
    'LDO-USD', 'STX-USD', 'OP-USD', 'FET-USD', 'AERO-USD', 'CRV-USD', 'NEXO-USD', 'GRT-USD', 'PYTH-USD', 'KAIA-USD',
    'SNX-USD', 'FLOKI-USD', 'ATH-USD', 'XTZ-USD', 'ENS-USD', 'ETHFI-USD', 'MORPHO-USD', 'PENDLE-USD', 'IOTA-USD'
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

if not settings.watchlist_equities:
    settings.watchlist_equities = ",".join(TOP_STOCKS)
if not settings.watchlist_cryptos:
    settings.watchlist_cryptos = ",".join(TOP_CRYPTOS)


def _dedupe_upper(seq: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in seq:
        s = str(raw or "").strip().upper()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _dedupe_crypto(seq: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in seq:
        sym = _norm_crypto_symbol(raw)
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


equity_items = settings.watchlist_equities.split(",") if settings.watchlist_equities else list(TOP_STOCKS)
crypto_items = settings.watchlist_cryptos.split(",") if settings.watchlist_cryptos else list(TOP_CRYPTOS)

# Canonical watchlist sets derived from settings (preserve declared order)
WL_EQ = _dedupe_upper(equity_items)[:200]
WL_CR = _dedupe_crypto(crypto_items)[:200]

# Retraining tiers
if settings.retrain_daily_symbols:
    RETRAIN_DAILY = _dedupe_upper(settings.retrain_daily_symbols.split(","))
else:
    RETRAIN_DAILY = WL_EQ[: min(15, len(WL_EQ))]

if settings.retrain_weekly_symbols:
    RETRAIN_WEEKLY = _dedupe_upper(settings.retrain_weekly_symbols.split(","))
else:
    RETRAIN_WEEKLY = _dedupe_upper(WL_EQ[15:50] + WL_CR[:15])

# Derived arrays for daily quant and other jobs
if not getattr(settings, "equity_watch", None):
    settings.equity_watch = WL_EQ.copy()
if not getattr(settings, "crypto_watch", None):
    settings.crypto_watch = WL_CR.copy()

# Config knobs (reasonable defaults)
STAT_BARS_QUICK = 252          # ~1y worth of bars to estimate μ/σ
STAT_BARS_DEEP  = 504          # ~2y for deep
MU_CAP_QUICK    = 1.00         # |μ| ≤ 100%/yr for quick
MU_CAP_DEEP     = 0.50         # |μ| ≤ 50%/yr for deep
SIGMA_CAP       = 0.80         # σ ≤ 80%/yr

_BG_TASKS: list[asyncio.Task] = []

def _run_bg(name: str, coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
    """
    Run a background coroutine with observability.
    - Logs start/finish
    - Records job_ok / job_fail
    - Captures exceptions so they don't crash the loop
    """
    async def _wrapper():
        t0 = time.perf_counter()
        started_at = datetime.now(timezone.utc).isoformat()
        log_json("info", msg="bg_task_start", task=name, started_at=started_at)
        try:
            result = await coro
            dur = round(time.perf_counter() - t0, 3)
            job_ok(name, duration_s=dur)
            log_json("info", msg="bg_task_ok", task=name, duration_s=dur)
            return result
        except asyncio.CancelledError:
            # treat as graceful stop
            dur = round(time.perf_counter() - t0, 3)
            log_json("info", msg="bg_task_cancelled", task=name, duration_s=dur)
            raise
        except Exception as e:
            dur = round(time.perf_counter() - t0, 3)
            job_fail(name, err=str(e), duration_s=dur)
            log_json("error", msg="bg_task_fail", task=name, duration_s=dur, error=str(e))
            # don't re-raise to avoid bubbling
            return None

    task = asyncio.create_task(_wrapper(), name=name)
    _BG_TASKS.append(task)
    return task

# ---------- startup ----------
@app.on_event("startup")
async def _on_startup():
    # init_schema (measure+log)
    t0 = time.perf_counter()
    try:
        init_schema()
        dur = round(time.perf_counter() - t0, 3)
        job_ok("init_schema", duration_s=dur)
        log_json("info", msg="init_schema_ok", duration_s=dur)
    except Exception as e:
        dur = round(time.perf_counter() - t0, 3)
        job_fail("init_schema", err=str(e), duration_s=dur)
        log_json("error", msg="init_schema_fail", duration_s=dur, error=str(e))
        # don't crash startup
    # Background loops (observable)
    _run_bg("gc_loop", _gc_loop())
    _run_bg("labeling_daemon", _labeling_daemon())
    _run_bg("metrics_rollup", _metrics_rollup_loop())
    _run_bg("data_retention", _data_retention_loop())
    _run_bg("model_retrain", _model_retrain_loop())
    _run_bg("warm_start", _warm_start())  # warm but gentle
    _run_bg("duckdb_backup", _backup_loop())
    _run_bg("quant_scheduler", _daily_quant_scheduler())
    _run_bg("macro_scheduler", _macro_scheduler_loop())
    _run_bg("earnings_scheduler", _earnings_scheduler_loop())
    _run_bg("news_scheduler", _news_scheduler_loop())
    if SIM_DISPATCHER:
        await SIM_DISPATCHER.start()
        log_json("info", msg="sim_dispatcher_started", workers=settings.sim_queue_concurrency)

    


# ---------- graceful shutdown ----------
@app.on_event("shutdown")
async def _on_shutdown():
    log_json("info", msg="shutdown_begin", n_tasks=len(_BG_TASKS))
    # cancel & await (best-effort)
    for t in _BG_TASKS:
        if not t.done():
            t.cancel()
    # give tasks a moment to exit gracefully
    try:
        await asyncio.wait(_BG_TASKS, timeout=2.5)
    except Exception as e:
        log_json("error", msg="shutdown_wait_error", error=str(e))
    # log final states
    states = [{"name": getattr(t, "get_name", lambda: "bg_task")(), "done": t.done(), "cancelled": t.cancelled()} for t in _BG_TASKS]
    log_json("info", msg="shutdown_complete", task_states=states)
    if SIM_DISPATCHER:
        await SIM_DISPATCHER.stop()

# ---------- warm start (instrumented) ----------
async def _warm_start():
    """
    On cold start, gently prime data & cache without tripping rate limits.
    - Skip Polygon ingest if weekend or key is missing
    - Single-day ingest only (yesterday) if bars_daily is missing
    - Prime today's cache once, using the daily compute budget
    """
    job = "warm_start"
    t0 = time.perf_counter()
    try:
        # spread boots across instances
        jitter_ms = int(os.getenv("PT_WARM_JITTER_MS", "300"))
        if jitter_ms > 0:
            jt = random.random() * (jitter_ms / 1000.0)
            log_json("info", msg="warm_start_jitter", job=job, sleep_s=round(jt, 3))
            await asyncio.sleep(jt)

        # 1) Check for bars_daily
        try:
            con = fs_connect()
            has = con.execute(
                "SELECT 1 FROM information_schema.tables WHERE lower(table_name)='bars_daily'"
            ).fetchall()
            con.close()
            log_json("info", msg="warm_table_check", job=job, bars_daily_present=bool(has))
        except Exception as e:
            log_json("error", msg="warm_table_check_failed", job=job, error=str(e))
            has = []

        if not has:
            y = datetime.now(timezone.utc).date() - timedelta(days=1)
            if _poly_key_present() and not _is_weekend(y):
                try:
                    t_ing = time.perf_counter()
                    await ingest_grouped_daily(y)
                    log_json("info", msg="warm_ingest_ok", job=job, day=y.isoformat(),
                             duration_s=round(time.perf_counter() - t_ing, 3))
                except Exception as e:
                    log_json("error", msg="warm_ingest_skip", job=job, day=y.isoformat(), error=str(e))
            else:
                log_json("info", msg="warm_ingest_skipped", job=job, reason="weekend_or_no_polygon_key")

        # 2) Prime today's cache (idempotent)
        if REDIS:
            d = datetime.now(timezone.utc).date().isoformat()
            base = f"quant:daily:{d}"
            try:
                eq = await REDIS.get(f"{base}:equity")
                cr = await REDIS.get(f"{base}:crypto")
                log_json("info", msg="warm_cache_probe", job=job, equity_cached=bool(eq), crypto_cached=bool(cr))
            except Exception as e:
                log_json("error", msg="warm_cache_probe_fail", job=job, error=str(e))
                eq = cr = None

            if not (eq and cr):
                warm_budget = int(os.getenv("PT_WARM_BUDGET", "2"))
                try:
                    if await _quant_allow(REDIS, max_calls_per_day=warm_budget):
                        t_quant = time.perf_counter()
                        err = None
                        try:
                            await _run_daily_quant()
                        except Exception as e:
                            err = str(e)
                        finally:
                            try:
                                await _quant_consume(REDIS)
                            except Exception:
                                pass
                        dur = round(time.perf_counter() - t_quant, 3)
                        if err:
                            log_json("error", msg="warm_quant_fail", job=job, duration_s=dur, error=err)
                        else:
                            log_json("info", msg="warm_quant_ok", job=job, duration_s=dur)
                    else:
                        log_json("info", msg="warm_quant_skipped", job=job, reason="budget_exhausted")
                except Exception as e:
                    log_json("error", msg="warm_quant_outer_fail", job=job, error=str(e))
        else:
            log_json("info", msg="warm_skip", job=job, reason="no_redis_configured")

        dur_total = round(time.perf_counter() - t0, 3)
        job_ok(job, duration_s=dur_total)
        log_json("info", msg="warm_done", job=job, duration_s=dur_total)

    except Exception as e:
        dur_total = round(time.perf_counter() - t0, 3)
        job_fail(job, err=str(e), duration_s=dur_total)
        log_json("error", msg="warm_outer_fail", job=job, duration_s=dur_total, error=str(e))
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
        t_days = first_idx  
    tMedDays = float(np.nanmedian(t_days)) if np.isfinite(t_days).any() else None
    return {"hitEver": hitEver, "hitByEnd": hitByEnd, "tMedDays": tMedDays}

def _iv30_or_none(symbol: str) -> float | None:
    """Estimate a 30-day annualized volatility using Polygon data (fallback to realized vol)."""
    symbol_norm = (symbol or "").upper().strip()
    if not symbol_norm:
        return None
    cached = _IV_CACHE.get(symbol_norm)
    if cached:
        value, ts = cached
        if datetime.now(timezone.utc) - ts < IV_CACHE_TTL:
            return value

    key = _poly_key()
    if not key:
        return None

    poly_symbol = _to_polygon_ticker(symbol_norm)
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=60)
    url = f"https://api.polygon.io/v2/aggs/ticker/{poly_symbol}/range/1/day/{start_date}/{end_date}"
    params = {"adjusted": "true", "sort": "desc", "limit": "120"}
    headers = {"Authorization": f"Bearer {key}"}

    try:
        resp = httpx.get(url, params=params, headers=headers, timeout=5.0)
        resp.raise_for_status()
        data = resp.json() or {}
        results = data.get("results") or []
        closes = [
            float(item.get("c"))
            for item in results
            if isinstance(item, dict) and item.get("c") is not None
        ]
        closes = closes[:60]
        if len(closes) < 30:
            return None
        closes = closes[::-1]
        arr = np.asarray(closes, dtype=float)
        rets = np.diff(np.log(arr))
        rets = rets[np.isfinite(rets)]
        if rets.size < 20:
            return None
        sigma_daily = float(np.std(rets, ddof=1))
        if not math.isfinite(sigma_daily) or sigma_daily <= 0:
            return None
        sigma_ann = float(np.clip(sigma_daily * math.sqrt(252.0), 1e-4, 5.0))
        _IV_CACHE[symbol_norm] = (sigma_ann, datetime.now(timezone.utc))
        return sigma_ann
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response else "?"
        logger.debug("Polygon IV fetch error %s for %s: %s", status, symbol_norm, exc)
    except Exception as exc:
        logger.debug("IV fetch failed for %s: %s", symbol_norm, exc)
    return None

def _instrument_kind(symbol: str | None, S0: float) -> str:
    s = (symbol or "").upper()
    if s.startswith("X:") or s.endswith("-USD") or (s.endswith("USD") and "-" not in s):
        return "crypto"
    if S0 >= 100000 and s.startswith("BTC"):
        return "crypto"
    return "equity"




def compute_targets_block(
    paths: np.ndarray,
    S0: float,
    horizon_days: int,
    bars_per_day: int,
    rel_levels: Optional[Sequence[float]] = None,
    *,
    symbol: str | None = None,
    sigma_hint: float | None = None,
    calibration_hint: Optional[Mapping[str, Any]] = None,
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
        kind = _instrument_kind(symbol, S0)
        sigma_cal = _sigma_from_calibration(calibration_hint) if calibration_hint else None
        sigma_for_levels = sigma_hint
        if sigma_for_levels is not None and sigma_cal is not None:
            sigma_for_levels = float(np.clip(0.5 * (sigma_for_levels + sigma_cal), 1e-4, 3.0))
        elif sigma_for_levels is None:
            sigma_for_levels = sigma_cal
        if sigma_for_levels is None:
            sigma_for_levels = 0.35 if kind == "crypto" else 0.20
        expected_move = float(np.clip(sigma_for_levels * math.sqrt(max(1, horizon_days) / 252.0), 0.01, 3.0))
        rel_levels = rel_levels_from_expected_move(expected_move, kind=kind)

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
# ====== Engines ============================================================
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

# --- CORS (apply after ALLOWED_ORIGINS is computed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "X-API-Key", "x-pt-key", "Authorization", "X-Requested-With", "X-Polygon-Key"],
    max_age=86400,
)
# --- Optional static frontend
STATIC_DIR = os.getenv("PT_STATIC_DIR", "frontend/dist")
static_path = Path(STATIC_DIR).resolve()
if static_path.is_dir():
    app.mount("/app", StaticFiles(directory=str(static_path), html=True), name="app")
    logger.info(f"Mounted static frontend from {static_path} at /app")
FRONTEND_DIR = os.getenv("PT_FRONTEND_DIR", "").strip()

# --- RL constants
RL_WINDOW = int(os.getenv("PT_RL_WINDOW", "100"))
USE_SIM_BIAS = os.getenv("PT_SIM_BIAS", "1") == "1"
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

    def _coerce_json_obj(content: Any) -> dict:
        """
        Providers occasionally wrap JSON in code fences or prepend prose.
        Strip common wrappers and attempt to parse; raise on failure so caller
        can fall back cleanly.
        """
        if isinstance(content, dict):
            return content

        if not isinstance(content, str):
            raise ValueError("LLM response was not JSON or string encodable")

        text = content.strip()
        if not text:
            raise ValueError("LLM response was empty")

        # Remove Markdown-style ```json fences
        if text.startswith("```"):
            lines = text.splitlines()
            # drop first fence line
            if lines:
                lines = lines[1:]
            # drop trailing fence line if present
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        import json as _json

        try:
            return _json.loads(text)
        except _json.JSONDecodeError:
            # Try to salvage by locating JSON object substring
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return _json.loads(text[start : end + 1])
                except _json.JSONDecodeError:
                    pass
            raise

    try:
        if prefer_xai and xai_key:
            response_format = None
            if json_schema:
                response_format = {"type": "json_schema", "json_schema": json_schema}

            data = await _post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {xai_key}"},
                payload={
                    "model": os.getenv("XAI_MODEL", "grok-2-mini"),
                    "messages": [
                        {"role": "system", "content": "Be factual, concise, compliance-safe."},
                        prompt_user,
                    ],
                    "temperature": 0.2,
                    **({"response_format": response_format} if response_format else {}),
                },
            )
            raw_content = data.get("choices", [{}])[0].get("message", {}).get("content")
            content = raw_content.strip() if isinstance(raw_content, str) else raw_content
            return _coerce_json_obj(content)

        if oai_key:
            if json_schema:
                response_format = {"type": "json_schema", "json_schema": json_schema}
            else:
                response_format = {"type": "json_object"}

            data = await _post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {oai_key}"},
                payload={
                    "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    "messages": [
                        {"role": "system", "content": "Be factual, concise, compliance-safe."},
                        prompt_user,
                    ],
                    "response_format": response_format,
                    "temperature": 0.2,
                },
            )
            raw_content = data.get("choices", [{}])[0].get("message", {}).get("content")
            content = raw_content.strip() if isinstance(raw_content, str) else raw_content
            return _coerce_json_obj(content)

        return _fallback()

    except Exception as e:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("llm_summarize_async: failed to parse LLM content; using fallback", exc_info=e)
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

    def proba(self, x) -> float:
        if self.w is None:
            raise RuntimeError("Call init(d) before proba()")
        xb = np.concatenate([[1.0], np.asarray(x, dtype=float)])
        if xb.size != self.w.size:
            if xb.size < self.w.size:
                xb = np.pad(xb, (0, self.w.size - xb.size))
            else:
                xb = xb[: self.w.size]
        z = float(np.dot(self.w, xb))
        return self._sigmoid(z)

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

async def ingest_grouped_daily(d: date):
    """
    Ingest Polygon grouped daily bars for US stocks and global crypto for a given UTC date.
    Creates/updates DuckDB table `bars_daily` (PRIMARY KEY: symbol, ts).
    Returns: {"ok": True, "upserted": <int>}
    """
    t_start = time.perf_counter()
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

    payload_hash = hashlib.sha256(json.dumps(to_upsert, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()

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
        if log_ingest_event:
            try:
                duration_ms = int((time.perf_counter() - t_start) * 1000)
                log_ingest_event(con, as_of=day, source="polygon_grouped", row_count=len(to_upsert), sha256=payload_hash, duration_ms=duration_ms, ok=True)
            except Exception as exc:
                logger.debug(f"log_ingest_event_failed: {exc}")
    except Exception as e:
        con.execute("ROLLBACK")
        if log_ingest_event:
            try:
                duration_ms = int((time.perf_counter() - t_start) * 1000)
                log_ingest_event(con, as_of=day, source="polygon_grouped", row_count=len(to_upsert), sha256=payload_hash, duration_ms=duration_ms, ok=False, error=str(e))
            except Exception:
                pass
        raise
    finally:
        con.close()

    return {"ok": True, "upserted": len(to_upsert)}

def _ensure_redis() -> Redis:
    if REDIS is None:
        raise HTTPException(status_code=503, detail="Redis unavailable")
    return REDIS

def _bars_per_day(timespan: str) -> int:
    if timespan == "day":   return 1
    if timespan == "hour":  return 24
    return 390  # minute


def _llm_background_enabled() -> bool:
    """
    Return True when background/automated LLM jobs (news scoring, quant adjudication)
    are permitted. Defaults to off so LLM usage only occurs during explicit simulations.
    """
    val = (
        os.getenv("PT_LLM_BACKGROUND")
        or os.getenv("PT_LLM_AUTO")
        or "0"
    )
    return val.strip().lower() in {"1", "true", "yes", "on"}


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

TRAIN_REFRESH_HOURS = float(os.getenv("PT_TRAIN_REFRESH_HOURS", "12") or 0)
TRAIN_REFRESH_DELTA = timedelta(hours=TRAIN_REFRESH_HOURS) if TRAIN_REFRESH_HOURS > 0 else None
QUICK_TRAIN_LOOKBACK_DAYS = max(30, min(settings.lookback_days_max, int(os.getenv("PT_TRAIN_QUICK_LOOKBACK_DAYS", "180"))))
DEEP_TRAIN_LOOKBACK_DAYS = max(30, min(settings.lookback_days_max, int(os.getenv("PT_TRAIN_DEEP_LOOKBACK_DAYS", "3650"))))

class AdminLoginRequest(BaseModel):
    username: str
    password: str

class TrainRequest(BaseModel):
    symbol: str
    lookback_days: int = Field(default=365, ge=30, le=3650)


class PromoteModelRequest(BaseModel):
    version: str = Field(..., min_length=1, max_length=128)


class PredictRequest(BaseModel):
    symbol: str
    horizon_days: int = Field(default=30, ge=1, le=365)
    use_online: bool = True


class BacktestReq(BaseModel):
    symbols: List[str]
    horizon_days: int = Field(30, ge=1, le=365)
    start: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    end: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    limit_per_symbol: Optional[int] = Field(300, ge=5, le=2000)


class BacktestResp(BaseModel):
    n: int
    mdape: Dict[str, float]
    coverage90: Dict[str, float]
    brier: Optional[Dict[str, float]] = None
    crps: Optional[Dict[str, float]] = None


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
    status_detail: str | None = None
    owner: str | None = None
#------------Lemon Squezzy checkout------------
def _ls_valid_sig(raw: bytes, sig_b64: str) -> bool:
    if not LS_WEBHOOK_SECRET or not sig_b64:
        return False
    mac = hmac.new(LS_WEBHOOK_SECRET, msg=raw, digestmod=hashlib.sha256).digest()
    expected = base64.b64encode(mac).decode("utf-8")
    return hmac.compare_digest(expected, sig_b64)

async def _grant_plan(email: str, plan: str, ttl_days: int | None = 365) -> None:
    if not REDIS or not email: return
    key = f"user:{email.lower()}:plan"
    if ttl_days:
        await REDIS.set(key, plan, ex=ttl_days*24*3600)
    else:
        await REDIS.set(key, plan)

async def _revoke_or_downgrade(email: str) -> None:
    if not REDIS or not email: return
    key = f"user:{email.lower()}:plan"
    await REDIS.set(key, "free", ex=30*24*3600)  # keep a short TTL

def _norm_plan_name(s: str | None) -> str:
    s = (s or "").strip().lower()
    if "enterprise" in s: return "enterprise"
    if "pro" in s: return "pro"
    return "free"

def _payload_email(attrs: dict) -> str:
    return (
        attrs.get("user_email") or
        attrs.get("customer_email") or
        attrs.get("email") or
        ""
    )

@app.post("/billing/webhook")
async def lemon_webhook(request: Request):
    raw = await request.body()
    sig = request.headers.get("X-Signature", "")
    if not _ls_valid_sig(raw, sig):
        raise HTTPException(status_code=401, detail="invalid signature")

    payload = json.loads(raw.decode("utf-8") or "{}")
    meta = payload.get("meta") or {}
    event = (meta.get("event_name") or "").strip().lower()
    data  = payload.get("data") or {}
    attrs = (data.get("attributes") or {})

    # --- Idempotency guard: skip if we've seen this event id ---
    evt_id = meta.get("event_id") or data.get("id")
    if REDIS and evt_id:
        if await REDIS.sismember("ls:events:seen", evt_id):
            return {"ok": True, "idempotent": True}
        await REDIS.sadd("ls:events:seen", evt_id)
        await REDIS.expire("ls:events:seen", 14*24*3600)

    email = (_payload_email(attrs) or "").lower()
    variant_name = _norm_plan_name(attrs.get("variant_name") or attrs.get("name"))

    # Handle events
    if event in {
        "order_created",
        "subscription_created",
        "subscription_updated",
    }:
        await _grant_plan(email, variant_name, ttl_days=365)
    elif event in {
        "subscription_cancelled",
        "subscription_expired",
        "order_refunded",
    }:
        await _revoke_or_downgrade(email)
    else:
        # ignore others
        pass

    return {"ok": True, "event": event, "email": email, "plan": variant_name}

# Simple plan lookup (use with real auth later)
@app.get("/me")
async def me(email: str):
    email = (email or "").strip().lower()
    plan = "free"
    if REDIS and email:
        p = await REDIS.get(f"user:{email}")
        if p: plan = p
        else:
            p2 = await REDIS.get(f"user:{email}:plan")
            if p2: plan = p2
    return {"email": email, "plan": plan}
#------------End Lemon Squezzy checkout------------

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

def _basic_features_from_array(arr: np.ndarray) -> Dict[str, float]:
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


async def _feat_from_prices(symbol: str, px: List[float]) -> Dict[str, float]:
    return await training_feat_from_prices(symbol, px)


async def _fetch_cached_hist_prices(symbol: str, window_days: int, redis: Redis | None) -> List[float]:
    return await ingestion_fetch_cached_hist_prices(symbol, window_days, redis)

def _labeler_pass(limit: int) -> dict[str, int]:
    return service_labeler_pass(limit)


async def _run_labeling_pass(limit: int) -> dict[str, int]:
    return await service_run_labeling_pass(limit)


async def _run_labeling_pass(limit: int) -> dict[str, int]:
    return await asyncio.to_thread(_labeler_pass, limit)


async def _labeling_daemon():
    if label_mature_predictions is None:
        logger.info("Labeler module unavailable; background labeling disabled.")
        return

    interval = max(60, int(settings.labeling_interval_seconds))
    limit = max(100, int(settings.labeling_batch_limit))
    backoff = interval

    # tiny jitter so parallel replicas don't hammer providers simultaneously
    await asyncio.sleep(random.uniform(5.0, 15.0))

    while True:
        try:
            stats = await _run_labeling_pass(limit)
            processed = int(stats.get("processed", 0))
            labeled = int(stats.get("labeled", 0))
            log_json(
                "info",
                msg="labeling_pass",
                processed=processed,
                labeled=labeled,
                limit=limit,
            )
            job_ok("labeling_pass", processed=processed, labeled=labeled)
            if labeled > 0:
                day = datetime.now(timezone.utc).date()
                artifacts = await asyncio.to_thread(_export_daily_snapshots, day)
                if artifacts:
                    log_json(
                        "info",
                        msg="labeling_exports",
                        day=day.isoformat(),
                        artifacts=artifacts,
                    )
            # If nothing to do, stretch the next run slightly (up to 2x interval)
            if processed == 0 or labeled == 0:
                backoff = min(interval * 2, interval + 600)
            else:
                backoff = interval
        except asyncio.CancelledError:
            break
        except Exception as e:
            job_fail("labeling_pass", err=str(e))
            logger.warning(f"Labeling pass failed: {e}")
            backoff = min(interval * 2, interval + 900)
        await asyncio.sleep(backoff)


async def _metrics_rollup_loop():
    if fs_connect is None or _rollup is None:
        logger.info("Feature store unavailable; metrics rollup disabled.")
        return

    interval = max(300, int(settings.metrics_rollup_interval_seconds))
    await asyncio.sleep(random.uniform(30.0, 90.0))

    while True:
        day = (datetime.now(timezone.utc) - timedelta(days=1)).date()
        try:
            con = fs_connect()
            rows = _rollup(con, day=day)
            artifacts = await asyncio.to_thread(_export_daily_snapshots, day)
            log_json("info", msg="metrics_rollup_pass", day=day.isoformat(), rows=rows, artifacts=artifacts)
            job_ok("metrics_rollup", day=day.isoformat(), rows=rows, artifacts=artifacts)
        except Exception as e:
            job_fail("metrics_rollup", err=str(e), day=day.isoformat())
            logger.warning(f"Metrics rollup failed for {day}: {e}")
        finally:
            try:
                con.close()
            except Exception:
                pass
        await asyncio.sleep(interval)


async def _maybe_retrain_symbol(sym: str, *, max_age: timedelta, lookback_days: int, tier: str) -> None:
    try:
        meta = await _get_model_meta(sym)
    except Exception:
        meta = None

    should_retrain = False
    if not meta:
        should_retrain = True
    else:
        trained_at = _parse_trained_at(meta.get("trained_at"))
        if trained_at is None:
            should_retrain = True
        else:
            age = datetime.now(timezone.utc) - trained_at
            if age >= max_age:
                should_retrain = True

    if not should_retrain:
        return

    try:
        min_samples = 80 if lookback_days >= 365 else max(40, lookback_days // 3)
        await _train_models(sym, lookback_days=lookback_days, min_samples=min_samples)
        log_json("info", msg="model_retrain_ok", symbol=sym, tier=tier, lookback_days=lookback_days)
    except HTTPException as exc:
        log_json("warning", msg="model_retrain_http_fail", symbol=sym, tier=tier, error=exc.detail)
    except Exception as exc:
        log_json("error", msg="model_retrain_fail", symbol=sym, tier=tier, error=str(exc))


async def _model_retrain_loop():
    if not REDIS:
        logger.info("Redis unavailable; model retraining disabled.")
        return

    poll = max(300, int(settings.model_retrain_poll_seconds))
    daily_age = timedelta(hours=int(settings.model_retrain_daily_hours))
    weekly_age = timedelta(hours=int(settings.model_retrain_weekly_hours))
    daily_lookback = min(settings.lookback_days_max, 540)
    weekly_lookback = min(settings.lookback_days_max, 720)

    await asyncio.sleep(random.uniform(60.0, 180.0))

    while True:
        try:
            for sym in RETRAIN_DAILY:
                await _maybe_retrain_symbol(sym, max_age=daily_age, lookback_days=daily_lookback, tier="daily")
                await asyncio.sleep(0.1)
            for sym in RETRAIN_WEEKLY:
                await _maybe_retrain_symbol(sym, max_age=weekly_age, lookback_days=weekly_lookback, tier="weekly")
                await asyncio.sleep(0.1)
        except Exception as exc:
            logger.warning(f"Model retrain loop error: {exc}")
        await asyncio.sleep(poll)


async def _data_retention_loop():
    if fs_connect is None:
        logger.info("Feature store unavailable; data retention disabled.")
        return
    days = int(getattr(settings, "data_retention_days", 0))
    if days <= 0:
        logger.info("Data retention disabled (data_retention_days <= 0).")
        return
    interval = max(3600, int(settings.data_retention_poll_seconds))
    retention_delta = timedelta(days=days)

    await asyncio.sleep(random.uniform(600.0, 1800.0))

    while True:
        cutoff_ts = datetime.now(timezone.utc) - retention_delta
        cutoff_iso = cutoff_ts.isoformat()
        cutoff_date = cutoff_ts.date()
        try:
            con = fs_connect()
            con.execute("DELETE FROM predictions WHERE issued_at < ?", [cutoff_iso])
            con.execute("DELETE FROM outcomes WHERE realized_at < ?", [cutoff_iso])
            con.execute("DELETE FROM metrics_daily WHERE date < ?", [cutoff_date.isoformat()])
            con.close()
            _prune_arrow_partitions(ARROW_PREDICTIONS_DIR, cutoff_date)
            _prune_arrow_partitions(ARROW_OUTCOMES_DIR, cutoff_date)
            log_json("info", msg="data_retention_ok", cutoff=cutoff_iso, days=days)
        except Exception as exc:
            logger.warning(f"Data retention pass failed: {exc}")
        await asyncio.sleep(interval)

# --------- Optional model loaders (only if missing up top) ----------
if "load_lstm_model" not in globals():
    def load_lstm_model(symbol: str):
        sym = (symbol or "").upper().strip()
        if not sym:
            raise HTTPException(status_code=400, detail="symbol_required")
        cached = _LSTM_MODEL_CACHE.get(sym)
        if cached is not None:
            return cached
        require_tf()
        candidates: list[Path] = []
        try:
            entry = get_active_model_version("lstm", sym)
        except Exception as exc:
            logger.debug("Model registry lookup failed for lstm %s: %s", sym, exc)
            entry = None
        if entry and entry.get("artifact_path"):
            try:
                candidates.append(_resolve_artifact_path(entry["artifact_path"]))
            except Exception as exc:
                logger.debug("Unable to resolve LSTM registry artifact for %s: %s", sym, exc)
        candidates.extend(
            [
                _lstm_saved_model_dir(sym),
                Path("models") / f"{sym}_lstm.keras",
                Path("models") / f"{sym}_lstm.h5",
            ]
        )
        for path in candidates:
            try:
                if not path.exists():
                    continue
                if "_tf_load_model" in globals() and _tf_load_model:
                    model = _tf_load_model(path.as_posix())
                else:
                    from tensorflow.keras.models import load_model as _lm  # late import fallback
                    model = _lm(path.as_posix())
                _LSTM_MODEL_CACHE[sym] = model
                return model
            except Exception:
                continue
        raise HTTPException(status_code=404, detail="LSTM model not found; train first")

if "load_arima_model" not in globals():
    def load_arima_model(symbol: str):
        sym = (symbol or "").upper().strip()
        if not sym:
            raise HTTPException(status_code=400, detail="symbol_required")
        cached = _ARIMA_MODEL_CACHE.get(sym)
        if cached is not None:
            return cached
        try:
            with open(f"models/{sym}_arima.pkl", "rb") as file:
                model = pickle.load(file)
                _ARIMA_MODEL_CACHE[sym] = model
                return model
        except Exception:
            raise HTTPException(status_code=404, detail="ARIMA model not found; train first")


def load_rl_model(symbol: str):
    sym = (symbol or "").upper().strip()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol_required")
    if gym is None:
        raise HTTPException(status_code=503, detail="rl_unavailable")
    if DQN is None:
        try:
            load_stable_baselines3()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"rl_unavailable:{exc}") from exc
    if DQN is None:
        raise HTTPException(status_code=503, detail="rl_unavailable")
    cached = _RL_MODEL_CACHE.get(sym)
    if cached is not None:
        return cached
    path = Path("models") / f"{sym}_rl.zip"
    if not path.exists():
        raise HTTPException(status_code=404, detail="RL model not found; train first")
    try:
        model = DQN.load(path.as_posix(), print_system_info=False)
        _RL_MODEL_CACHE[sym] = model
        return model
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to load rl model: {exc}") from exc

# --------- Ensemble (linear + optional lstm/arima + optional RL) ----------
if "get_ensemble_prob_light" not in globals():
    async def get_ensemble_prob_light(
        symbol: str,
        redis: "Redis",
        horizon_days: int = 1,
        profile: str | None = None,
    ) -> float:
        try:
            return await service_get_ensemble_prob_light(
                symbol,
                redis,
                horizon_days,
                profile=profile,
            )
        except HTTPException:
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
        feature_vector = [float(f.get(feat, 0.0)) for feat in feature_list]
        X = np.array(feature_vector, dtype=float)

        preds: Dict[str, float] = {}

        # --- Linear head (supports ONNX or intercept if present)
        onnx_prob = _run_linear_onnx(symbol, feature_vector)
        if onnx_prob is not None:
            preds["linear"] = float(np.clip(onnx_prob, 0.0, 1.0))

        if "linear" not in preds and "coef" in model_linear:
            w = np.array([float(c) for c in model_linear["coef"]], dtype=float)
            xb = np.concatenate([[1.0], X])  # intercept term
            k = min(w.shape[0], xb.shape[0])
            score = float(np.dot(w[:k], xb[:k]))
            preds["linear"] = _sigmoid(float(np.clip(score, -60.0, 60.0)))

        # --- LSTM (optional)
        lstm_prob = _run_lstm_savedmodel(symbol, feature_vector)
        if lstm_prob is not None:
            preds["lstm"] = float(np.clip(lstm_prob, 0.0, 1.0))
        else:
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
    return service_detect_regime(px)


async def _update_run_state(
    redis: Redis,
    run_id: str,
    rs: RunState,
    *,
    status: str | None = None,
    progress: float | None = None,
    detail: str | None = None,
    error: str | None = None,
) -> None:
    if status is not None:
        rs.status = status
    if progress is not None:
        rs.progress = float(progress)
    if detail is not None:
        rs.status_detail = detail
    if error is not None:
        rs.error = error
    await redis.setex(f"run:{run_id}", settings.run_ttl_seconds, rs.model_dump_json())


# ----------------- Simulation worker -----------------
async def run_simulation(run_id: str, req: "SimRequest", redis: Redis):
    mode_label = str(getattr(req, "mode", "quick") or "quick").lower()
    started_at = time.perf_counter()
    if SIM_ACTIVE_GAUGE:
        try:
            SIM_ACTIVE_GAUGE.inc()
        except Exception:
            pass

    try:
        logger.info(f"Starting simulation for run_id={run_id}, symbol={req.symbol}")
        try:
            rs = await _ensure_run(run_id)
        except Exception as e:
            logger.error(f"_ensure_run failed at start for {run_id}: {e}")
            rs = RunState(status="error", progress=0.0, error=str(e))
            await redis.setex(f"run:{run_id}", settings.run_ttl_seconds, rs.model_dump_json())
            return

        await _update_run_state(
            redis,
            run_id,
            rs,
            status="running",
            progress=1.0,
            detail="Initializing simulation",
        )

        try:
            await _update_run_state(redis, run_id, rs, progress=4.0, detail="Validating request limits")
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

            # Ensure prerequisite models are trained with an appropriate lookback
            desired_lookback = DEEP_TRAIN_LOOKBACK_DAYS if mode == "deep" else QUICK_TRAIN_LOOKBACK_DAYS
            desired_lookback = int(min(settings.lookback_days_max, desired_lookback))
            await _ensure_trained_models(
                req.symbol,
                required_lookback=desired_lookback,
                profile=mode,
            )

            # ---------- Fetch prices ----------
            await _update_run_state(redis, run_id, rs, progress=10.0, detail="Fetching historical prices")
            historical_prices = await _fetch_cached_hist_prices(req.symbol, window_days, redis)
            if not historical_prices or len(historical_prices) < 2:
                raise ValueError("Insufficient history")

            px_arr = np.array(historical_prices, dtype=float)
            S0 = float(px_arr[-1])

            # ---------- Returns & annualization ----------
            bpd = _bars_per_day(req.timespan)             # bars per day
            scale = 252 * bpd                             # bars per year (annualization)
            rets_all = np.diff(np.log(px_arr))            # full history for bootstrap/statistics
            await _update_run_state(redis, run_id, rs, progress=20.0, detail="Analyzing historical returns")

            # ---- μ / σ estimation on a decoupled recent window ----
            stat_bars = (STAT_BARS_QUICK if mode == "quick" else STAT_BARS_DEEP) * bpd
            px_stats = px_arr[-(stat_bars + 1):] if px_arr.size > (stat_bars + 1) else px_arr
            rets_est = np.diff(np.log(px_stats))
            rets_est = winsorize(rets_est)

            # EWMA σ (per-bar) → annualized
            sigma_bar = ewma_sigma(rets_est, lam=0.94)
            sigma_ann_raw = float(sigma_bar * math.sqrt(scale))
            sigma_ann = float(np.clip(sigma_ann_raw, 1e-4, SIGMA_CAP))

            # μ from winsorized sample mean (annualized), with caps
            mu_ann_raw = float(np.mean(rets_est) * scale) if rets_est.size else 0.0
            mu_cap = MU_CAP_DEEP if mode == "deep" else MU_CAP_QUICK
            mu_ann = float(np.clip(mu_ann_raw, -mu_cap, mu_cap))

            # Horizon-based drift shrink
            shrink = horizon_shrink(int(req.horizon_days))
            mu_ann *= shrink

            # Optional IV-anchoring for near-term (if options enabled and IV available)
            iv30 = _iv30_or_none(req.symbol) if getattr(req, "include_options", False) else None
            if iv30 and isinstance(iv30, (float, int)) and iv30 > 0:
                sigma_ann = float(max(sigma_ann, float(iv30)))

            recommended_paths: int | None = None
            mc_tuning = _latest_mc_metric(req.symbol, int(req.horizon_days))
            if mc_tuning:
                try:
                    mu_tuned_ann = float(mc_tuning.get("mu") or 0.0) * VALIDATION_BARS_PER_YEAR
                    sigma_tuned_ann = float(mc_tuning.get("sigma") or 0.0) * math.sqrt(VALIDATION_BARS_PER_YEAR)
                    mu_tuned_ann *= shrink  # align with horizon shrink applied to fresh estimate
                    mu_ann = float(np.clip((mu_ann + mu_tuned_ann) * 0.5, -mu_cap, mu_cap))
                    sigma_ann = float(np.clip((sigma_ann + sigma_tuned_ann) * 0.5, 1e-4, SIGMA_CAP))
                    recommended_paths = int(mc_tuning.get("n_paths") or 0)
                except Exception as exc:
                    logger.debug("Monte Carlo tuning blend failed: %s", exc)

            await _update_run_state(redis, run_id, rs, progress=32.0, detail="Estimating drift and volatility")

            fusion_diag: dict | None = None
            if USE_SIM_BIAS:
                def _safe_sent_dict() -> dict:
                    return {"avg_sent_7d": 0.0, "last24h": 0.0, "n_news": 0}

                def _safe_earn_dict() -> dict:
                    return {"surprise_last": 0.0, "guidance_delta": 0.0, "days_since_earn": None, "days_to_next": None}

                def _safe_macro_dict() -> dict:
                    return {"rff": None, "cpi_yoy": None, "u_rate": None}

                try:
                    sent = await get_sentiment_features(req.symbol)
                except Exception as exc:
                    logger.debug("sim_bias: sentiment fetch failed for %s: %s", req.symbol, exc)
                    sent = _safe_sent_dict()
                try:
                    earn = await get_earnings_features(req.symbol)
                except Exception as exc:
                    logger.debug("sim_bias: earnings fetch failed for %s: %s", req.symbol, exc)
                    earn = _safe_earn_dict()
                try:
                    macr = await get_macro_features()
                except Exception as exc:
                    logger.debug("sim_bias: macro fetch failed: %s", exc)
                    macr = _safe_macro_dict()

                mu_ann_pre = float(mu_ann)
                sigma_ann_pre = float(sigma_ann)
                sent_avg = float(sent.get("avg_sent_7d") or 0.0)
                earn_surprise = float(earn.get("surprise_last") or 0.0)
                mu_bias = float(np.clip(sent_avg * 0.15 + earn_surprise * 0.05, -0.15, 0.15))
                mu_ann = float(np.clip(mu_ann + mu_bias, -mu_cap, mu_cap))

                sigma_ann_post = sigma_ann
                if float(sent.get("last24h") or 0.0) > 0.20:
                    sigma_ann_post = float(np.clip(sigma_ann_post * 0.97, 1e-4, SIGMA_CAP))
                sigma_ann = sigma_ann_post

                fusion_diag = {
                    "use_sim_bias": True,
                    "mu_ann_pre": mu_ann_pre,
                    "mu_ann_post": float(mu_ann),
                    "sigma_ann_pre": sigma_ann_pre,
                    "sigma_ann_post": float(sigma_ann),
                    "mu_bias": float(mu_bias),
                    "sent": {
                        "avg_sent_7d": sent_avg,
                        "last24h": float(sent.get("last24h") or 0.0),
                        "n_news": int(sent.get("n_news") or 0),
                    },
                    "earn": {
                        "surprise_last": earn_surprise,
                        "guidance_delta": float(earn.get("guidance_delta") or 0.0),
                        "days_since_earn": earn.get("days_since_earn"),
                        "days_to_next": earn.get("days_to_next"),
                    },
                    "macro": {
                        "rff": (float(macr.get("rff")) if macr.get("rff") is not None else None),
                        "cpi_yoy": (float(macr.get("cpi_yoy")) if macr.get("cpi_yoy") is not None else None),
                        "u_rate": (float(macr.get("u_rate")) if macr.get("u_rate") is not None else None),
                    },
                }

            await _update_run_state(redis, run_id, rs, progress=42.0, detail="Blending context signals")

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
                ensemble_prob = await asyncio.wait_for(
                    get_ensemble_prob(
                        req.symbol,
                        redis,
                        req.horizon_days,
                        profile=mode,
                    ),
                    timeout=1.0,
                )
            except Exception:
                try:
                    ensemble_prob = await asyncio.wait_for(
                        get_ensemble_prob_light(
                            req.symbol,
                            redis,
                            req.horizon_days,
                            profile=mode,
                        ),
                        timeout=0.5,
                    )
                except Exception:
                    ensemble_prob = 0.5
            conf = (2.0 * float(ensemble_prob) - 1.0)
            mu_adj = float(mu_adj + (0.30 * conf * sigma_adj))

            # ---------- Optional sentiment/options/futures tweaks ----------
            sentiment = 0.0
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
            if recommended_paths:
                # Respect global budgets and user request; only scale up.
                max_budget_paths = max(1, settings.pathday_budget_max // max(1, H))
                target_paths = min(int(recommended_paths), settings.n_paths_max, max_budget_paths)
                if target_paths > N:
                    N = target_paths
                    warnings.append(
                        f"Auto-tuned path count increased to {N} to target {VALIDATION_TARGET_MAPE:.1f}% MAPE calibration"
                    )
            N = int(np.clip(N, 500, settings.n_paths_max))
            engine_used = ""
            paths_mat: np.ndarray

            await _update_run_state(redis, run_id, rs, progress=52.0, detail="Generating Monte Carlo paths")

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

            await _update_run_state(redis, run_id, rs, progress=62.0, detail="Summarizing simulation outputs")

            horizon_days_ui = int(H) if bpd == 1 else int(math.ceil(H / float(bpd)))

            calibration_hint = await _get_calibration_params(req.symbol, horizon_days_ui)

            targets_block = compute_targets_block(
                paths=paths_mat,
                S0=S0,
                horizon_days=horizon_days_ui,
                bars_per_day=bpd,
                symbol=req.symbol,
                sigma_hint=sigma_adj,
                calibration_hint=calibration_hint,
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

            await _update_run_state(redis, run_id, rs, progress=82.0, detail="Computing risk metrics")

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

            if fusion_diag:
                artifact.setdefault("diagnostics", {}).update(fusion_diag)

            await _update_run_state(redis, run_id, rs, progress=90.0, detail="Persisting results")

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
                    "features_ref": json.dumps({
                        "window_days":  int(window_days),
                        "paths":        int(req.paths),
                        "S0":           float(S0),
                        "mu_ann":       float(mu_adj),
                        "sigma_ann":    float(sigma_adj),
                        "timespan":     req.timespan,
                        "seed_hint":    int(seed),
                        "mode":         mode,
                    }),
                })
                con.close()
            except Exception as e:
                logger.warning(f"Feature Store mirror failed: {e}")

            # ---------- Finish ----------
            rs.finishedAt = datetime.now(timezone.utc).isoformat()
            await _update_run_state(redis, run_id, rs, status="done", progress=100.0, detail="Simulation complete")
            logger.info(f"Completed simulation for run_id={run_id}")

        except Exception as e:
            logger.exception(f"Simulation failed for run_id={run_id}: {e}")
            rs.finishedAt = datetime.now(timezone.utc).isoformat()
            await _update_run_state(
                redis,
                run_id,
                rs,
                status="error",
                detail=f"Failed: {e}",
                error=str(e),
            )


    finally:
        if SIM_ACTIVE_GAUGE:
            try:
                SIM_ACTIVE_GAUGE.dec()
            except Exception:
                pass
        if SIM_DURATION_SECONDS:
            try:
                SIM_DURATION_SECONDS.labels(mode_label).observe(time.perf_counter() - started_at)
            except Exception:
                pass
# ----------------- Simulation routes -----------------
@app.post("/simulate", summary="Start a simulation run")
async def simulate(
    req: SimRequest,
    request: Request,
    _auth: bool = Security(require_key, scopes=["simulate"])
):
    """
    Start a simulation run (auth via app-level key; supports open_access toggle).
    Frontend should pass req.mode = "quick" (≈6m) or "deep" (≈10y).
    """
    # Normalize symbol early so logs/keys are consistent
    try:
        req.symbol = (req.symbol or "").upper().strip()
    except Exception:
        # If model is frozen, shadow it
        req = req.model_copy(update={"symbol": (req.symbol or "").upper().strip()})

    # Quota cost: deep runs count more than quick
    mode = (req.mode or "quick").lower().strip()
    cost_units = 3 if mode == "deep" else 1

    # Enforce rate+quota (simulate scope)
    await enforce_limits(REDIS, request, scope="simulate", per_min=SIM_LIMIT_PER_MIN, cost_units=cost_units)

    # Ensure Redis is available
    if not REDIS:
        log_json("error", msg="simulate_enqueue_fail", reason="redis_unavailable", symbol=req.symbol, mode=mode)
        raise HTTPException(status_code=503, detail="redis_unavailable")

    owner = getattr(request.state, "caller_id", None)
    run_id = uuid4().hex  # hex keeps keys simple (no dashes)
    rs = RunState(
        run_id=run_id,
        status="queued",
        progress=0.0,
        symbol=req.symbol,
        horizon_days=int(req.horizon_days),
        paths=int(req.paths),
        startedAt=datetime.now(timezone.utc).isoformat(),
        status_detail="Queued",
        owner=owner,
    )

    # Persist run state with configured TTL
    try:
        await REDIS.setex(f"run:{run_id}", settings.run_ttl_seconds, rs.model_dump_json())
    except Exception as e:
        log_json("error", msg="simulate_enqueue_fail", reason="redis_setex_error", symbol=req.symbol, mode=mode, error=str(e))
        raise HTTPException(status_code=503, detail="redis_error")

    # Kick off background simulation
    async def _runner() -> None:
        await run_simulation(run_id, req, REDIS)

    if SIM_DISPATCHER:
        try:
            SIM_DISPATCHER.submit(_runner)
        except Exception as exc:
            log_json("error", msg="simulate_enqueue_fail", run_id=run_id, reason="dispatcher_error", error=str(exc))
            raise HTTPException(status_code=503, detail="dispatcher_unavailable") from exc
    else:
        asyncio.create_task(_runner())
    log_json("info", msg="simulate_enqueue", run_id=run_id, symbol=req.symbol, mode=mode, horizon_days=rs.horizon_days, paths=rs.paths)

    return {"run_id": run_id}


# ------------------ Simulate: state (full) ------------------
@app.get("/simulate/{run_id}/state", summary="Get full run state")
async def get_sim_state(run_id: str, request: Request, _ok: bool = Security(require_key, scopes=["simulate"])):
    try:
        rs = await _ensure_run_owned(run_id, request)
        log_json("info", msg="simulate_state", run_id=run_id, status=rs.status, progress=float(rs.progress or 0.0))
        return rs.model_dump()
    except HTTPException as e:
        log_json("error", msg="simulate_state_err", run_id=run_id, http_status=e.status_code, detail=e.detail)
        raise


# ------------------ Simulate: status (light) ------------------
@app.get("/simulate/{run_id}/status", summary="Get run status + progress")
async def simulate_status(run_id: str, request: Request, _ok: bool = Security(require_key, scopes=["simulate"])):
    try:
        rs = await _ensure_run_owned(run_id, request)
        payload = {"status": rs.status, "progress": rs.progress}
        if rs.status_detail:
            payload["detail"] = rs.status_detail
        if rs.error:
            payload["error"] = rs.error
        log_json("info", msg="simulate_status", run_id=run_id, **payload)
        return payload
    except HTTPException as e:
        log_json("error", msg="simulate_status_err", run_id=run_id, http_status=e.status_code, detail=e.detail)
        raise


# ------------------ Simulate: artifact ------------------
@app.get("/simulate/{run_id}/artifact", summary="Get final artifact once ready")
async def get_sim_artifact(run_id: str, request: Request, _ok: bool = Security(require_key, scopes=["simulate"])):
    if not REDIS:
        log_json("error", msg="simulate_artifact_err", run_id=run_id, reason="redis_unavailable")
        raise HTTPException(status_code=503, detail="redis_unavailable")

    await _ensure_run_owned(run_id, request)
    raw = await REDIS.get(f"artifact:{run_id}")
    if not raw:
        rs = await _ensure_run_owned(run_id, request)
        log_json("info", msg="simulate_artifact_pending", run_id=run_id, status=rs.status, progress=float(rs.progress or 0.0))
        raise HTTPException(status_code=202, detail=f"Run {run_id} status={rs.status}; artifact not ready")

    try:
        payload = json.loads(raw)
    except Exception:
        payload = {"artifact": (raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw))}

    # Fire-and-forget summarization if missing
    try:
        if isinstance(payload, dict) and not payload.get("summary"):
            asyncio.create_task(_summarize_run(run_id))
    except Exception:
        pass

    log_json("info", msg="simulate_artifact_ok", run_id=run_id, has_summary=bool(isinstance(payload, dict) and payload.get("summary")))
    return JSONResponse(content=payload)


# ------------------ Simulate: SSE progress stream ------------------
@app.get("/simulate/{run_id}/stream", summary="Server-sent events for progress")
async def simulate_stream(run_id: str, request: Request, _ok: bool = Security(require_key, scopes=["simulate"])):
    initial_state = await _ensure_run_owned(run_id, request)
    expected_owner = initial_state.owner
    elevated = _has_admin_access(request)

    async def event_generator():
        stale_ticks = 0
        last = None
        while True:
            try:
                if not REDIS:
                    yield 'data: {"status":"error","progress":0,"detail":"redis_unavailable"}\n\n'
                    log_json("error", msg="simulate_stream_err", run_id=run_id, reason="redis_unavailable")
                    break

                raw = await REDIS.get(f"run:{run_id}")
                if not raw:
                    yield 'data: {"status":"error","progress":0,"detail":"run_not_found_or_expired"}\n\n'
                    log_json("error", msg="simulate_stream_err", run_id=run_id, reason="run_not_found_or_expired")
                    break

                txt = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                rs = RunState.model_validate_json(txt)
                current_owner = rs.owner
                if expected_owner and current_owner and current_owner != expected_owner and not elevated:
                    yield 'data: {"status":"error","progress":0,"detail":"forbidden"}\n\n'
                    log_json("error", msg="simulate_stream_err", run_id=run_id, reason="owner_mismatch")
                    break

                progress = float(rs.progress or 0.0)
                payload = {"status": rs.status, "progress": progress}
                if rs.status_detail:
                    payload["detail"] = rs.status_detail
                if rs.error:
                    payload["error"] = rs.error
                yield f"data: {json.dumps(payload)}\n\n"

                # Throttle logs: only emit on status/progress change
                sig = (rs.status, round(progress, 2), rs.status_detail)
                if sig != last:
                    last = sig
                    log_json(
                        "info",
                        msg="simulate_stream_tick",
                        run_id=run_id,
                        status=rs.status,
                        progress=round(progress, 4),
                        detail=rs.status_detail,
                    )

                if rs.status in ("done", "error"):
                    if rs.error:
                        yield f'data: {json.dumps({"error": rs.error})}\n\n'
                    break

                # Stall detector (~2 minutes)
                if sig == last:
                    stale_ticks += 1
                else:
                    stale_ticks = 0
                if stale_ticks > 120:
                    yield 'data: {"status":"error","progress":0,"detail":"stalled"}\n\n'
                    log_json("error", msg="simulate_stream_err", run_id=run_id, reason="stalled")
                    break

                await asyncio.sleep(1.0)

            except Exception as e:
                yield f'data: {json.dumps({"status":"error","progress":0,"detail":str(e)[:200]})}\n\n'
                log_json("error", msg="simulate_stream_err", run_id=run_id, error=str(e)[:200])
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

async def llm_shortlist(
    kind: str,
    symbols: list[str],
    top_k: int = 20,
    horizon_days: int | None = None,
) -> list[str]:
    return await service_llm_shortlist(kind, symbols, top_k=top_k, horizon_days=horizon_days)

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
async def _load_precomputed_quant(kind: str, day: str) -> list[dict]:
    return await quant_adapters.load_precomputed_quant(kind, day, redis=REDIS)


def _normalize_precomputed_result(item: dict, horizon: int) -> dict:
    return quant_adapters.normalize_precomputed_result(item, horizon)


def _combine_mc_results(prefinal: list[dict], computed: list[dict]) -> list[dict]:
    return quant_adapters.combine_mc_results(prefinal, computed)


# ===== DAILY QUANT (shortlist → MC → LLM → persist) ===========================

async def _quick_score(symbol: str) -> dict:
    return await quant_adapters.quick_score(
        symbol,
        fetch_hist=_fetch_cached_hist_prices,
        compute_features=_feat_from_prices,
        detect_regime=_detect_regime,
        redis=REDIS,
    )


async def _rank_candidates(symbols: Sequence[str], top_k: int = 8) -> list[dict]:
    return await quant_adapters.rank_candidates(symbols, top_k, quick_score_func=_quick_score)

async def _mc_for(symbol: str, horizon: int | None = None, paths: int = 6000) -> tuple[str, dict]:
    return await quant_adapters.run_mc_for(
        symbol,
        horizon=horizon,
        paths=paths,
        redis=REDIS,
        quant_allow=_quant_allow,
        quant_consume=_quant_consume,
    )


async def _mc_batch(cands: list[dict], horizon: int | None = None, paths: int = 8000) -> list[dict]:
    return await quant_adapters.run_mc_batch(
        cands,
        horizon=horizon,
        paths=paths,
        limit=3,
        redis=REDIS,
        quant_allow=_quant_allow,
        quant_consume=_quant_consume,
    )

# --- LLM adjudicator (xAI/OpenAI) --------------------------------------------
async def _llm_select_and_write(kind: str, summaries: list[dict]) -> tuple[str, str]:
    return await quant_adapters.llm_select_and_write(kind, summaries)

async def _run_daily_quant(horizon_days: int | None = None) -> dict:
    """Delegate to the shared quant scheduler service."""
    return await service_run_daily_quant(
        horizon_days=horizon_days,
        legacy_scopes=(globals(),),
    )


async def _trigger_daily_quant_from_health(reason: str = "health") -> None:
    """Trigger the daily quant run via the quant scheduler service."""

    def _register(task: asyncio.Task) -> None:
        _BG_TASKS.append(task)

    await service_trigger_daily_quant_from_health(
        reason=reason,
        register_task=_register,
        legacy_scopes=(globals(),),
    )


async def _daily_quant_scheduler() -> None:
    await service_daily_quant_scheduler(legacy_scopes=(globals(),))


def _scheduler_symbols(kind: str, limit: int = 6) -> list[str]:
    syms: list[str] = []
    if kind == "news":
        eq = getattr(settings, "equity_watch", []) or []
        cr = getattr(settings, "crypto_watch", []) or []
        syms.extend([s.strip().upper() for s in eq if s.strip()])
        syms.extend([s.strip().upper() for s in cr if s.strip()])
    return list(dict.fromkeys(syms))[:max(1, limit)]


async def _macro_scheduler_loop() -> None:
    await asyncio.sleep(random.uniform(10.0, 45.0))
    interval = max(6 * 3600, int(os.getenv("PT_MACRO_SCHED_SECONDS", str(24 * 3600))))
    while True:
        try:
            res = await _ingest_macro(log_tag="scheduler")
            rows = int(res.get("rows", 0))
            job_ok("macro_scheduler", rows=rows, source=res.get("source"))
            log_json("info", msg="macro_scheduler_ok", rows=rows, source=res.get("source"), interval_s=interval)
        except HTTPException as exc:
            job_fail("macro_scheduler", err=str(exc))
            log_json("error", msg="macro_scheduler_fail", error=exc.detail)
        except Exception as exc:
            job_fail("macro_scheduler", err=str(exc))
            log_json("error", msg="macro_scheduler_fail", error=str(exc))
        await asyncio.sleep(interval)


async def _earnings_scheduler_loop() -> None:
    await asyncio.sleep(random.uniform(20.0, 90.0))
    interval = max(24 * 3600, int(os.getenv("PT_EARNINGS_SCHED_SECONDS", str(7 * 24 * 3600))))
    symbols = getattr(settings, "equity_watch", []) or []
    symbols = [s.strip().upper() for s in symbols if s.strip()]
    default_symbols = symbols[: min(8, len(symbols))] if symbols else []
    while True:
        rows_total = 0
        errors: list[str] = []
        if not default_symbols:
            log_json("info", msg="earnings_scheduler_skip", reason="no_symbols")
        else:
            for sym in default_symbols:
                try:
                    res = await _ingest_earnings(sym, lookback_days=370, limit=16, provider=None, log_tag="scheduler")
                    rows_total += int(res.get("rows", 0))
                except HTTPException as exc:
                    errors.append(f"{sym}:{exc.detail}")
                except Exception as exc:
                    errors.append(f"{sym}:{exc}")
        if errors:
            job_fail("earnings_scheduler", err="; ".join(errors[:3]))
            log_json("error", msg="earnings_scheduler_partial", errors=errors[:3], rows=rows_total)
        else:
            job_ok("earnings_scheduler", rows=rows_total, symbols=len(default_symbols))
            log_json("info", msg="earnings_scheduler_ok", rows=rows_total, symbols=len(default_symbols), interval_s=interval)
        await asyncio.sleep(interval)


async def _news_scheduler_loop() -> None:
    await asyncio.sleep(random.uniform(15.0, 60.0))
    interval = max(1800, int(os.getenv("PT_NEWS_SCHED_SECONDS", str(6 * 3600))))
    while True:
        syms = _scheduler_symbols("news", limit=8)
        ingested_total = 0
        scored_total = 0
        errors: list[str] = []
        if syms:
            for sym in syms:
                try:
                    res_ing = await _ingest_news(sym, days=2, limit=40, provider=None, log_tag="scheduler")
                    ingested_total += int(res_ing.get("rows", 0))
                except HTTPException as exc:
                    errors.append(f"ingest:{sym}:{exc.detail}")
                except Exception as exc:
                    errors.append(f"ingest:{sym}:{exc}")
                try:
                    res_score = await _score_news(sym, days=7, batch=24, log_tag="scheduler")
                    scored_total += int(res_score.get("rows", 0))
                except HTTPException as exc:
                    errors.append(f"score:{sym}:{exc.detail}")
                except Exception as exc:
                    errors.append(f"score:{sym}:{exc}")
        else:
            log_json("info", msg="news_scheduler_skip", reason="no_symbols")

        if errors:
            job_fail("news_scheduler", err="; ".join(errors[:3]), rows=ingested_total, scored=scored_total)
            log_json("error", msg="news_scheduler_partial", errors=errors[:5], rows=ingested_total, scored=scored_total)
        else:
            job_ok("news_scheduler", rows=ingested_total, scored=scored_total, symbols=len(syms))
            log_json(
                "info",
                msg="news_scheduler_ok",
                rows=ingested_total,
                scored=scored_total,
                symbols=len(syms),
                interval_s=interval,
            )
        await asyncio.sleep(interval)

def _poly_key_present() -> bool:
    return bool(os.getenv("PT_POLYGON_KEY") or os.getenv("POLYGON_KEY"))

def _is_weekend(d) -> bool:
    return d.weekday() >= 5  # 5=Sat, 6=Sun

@app.post("/admin/cron/daily", summary="Run daily label + online learn")
async def admin_cron_daily(
    request: Request,
    n: int = 20,              # how many symbols to learn (per list; equity+crypto de-duped)
    steps: int = 50,          # SGD steps per symbol
    batch: int = 32,          # minibatch size
    _ok: bool = Depends(require_key),
):
    job = "cron_daily"
    t0 = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    await enforce_limits(REDIS, request, scope="cron", per_min=CRON_LIMIT_PER_MIN, cost_units=1)

    try:
        # -------- Start --------
        log_json("info", msg="cron_start", job=job, n=n, steps=steps, batch=batch)

        # 1) Label anything matured
        label_t0 = time.perf_counter()
        lab = await outcomes_label(limit=20000, _api_key=True)  # reuse route logic
        label_sec = round(time.perf_counter() - label_t0, 3)
        log_json("info", msg="cron_labeled", job=job, labeled=lab, duration_s=label_sec)

        # 2) Build symbol set (equity + crypto)
        syms = list(dict.fromkeys(list(WL_EQ)[:n] + list(WL_CR)[:n]))
        log_json("info", msg="cron_symbols_prepared", job=job, n_symbols=len(syms))

        # 3) Online learn with small retry loop per symbol
        learned: list[dict] = []
        ok_count = 0
        err_count = 0

        async def learn_one(sym: str) -> dict:
            sym_t0 = time.perf_counter()
            last_err = None
            for attempt in (1, 2):  # 2 attempts
                try:
                    res = await learn_online(
                        OnlineLearnRequest(symbol=sym, steps=steps, batch=batch),
                        _api_key=True,
                    )
                    dur = round(time.perf_counter() - sym_t0, 3)
                    log_json("info", msg="learn_ok", job=job, symbol=sym, attempt=attempt, duration_s=dur)
                    return {"symbol": sym, "status": res.get("status", "ok"), "attempt": attempt, "duration_s": dur}
                except Exception as e:
                    last_err = str(e)
                    log_json("error", msg="learn_err", job=job, symbol=sym, attempt=attempt, error=last_err)
                    # brief backoff on first failure
                    if attempt == 1:
                        await asyncio.sleep(0.5)
            dur = round(time.perf_counter() - sym_t0, 3)
            return {"symbol": sym, "status": "error", "error": last_err or "unknown", "attempt": 2, "duration_s": dur}

        # sequential to avoid hammering providers; flip to gather() if desired
        for s in syms:
            item = await learn_one(s)
            learned.append(item)
            if item.get("status") == "ok":
                ok_count += 1
            else:
                err_count += 1

        # -------- Finish --------
        total_sec = round(time.perf_counter() - t0, 3)
        summary = {
            "ok": True,
            "job": job,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_s": total_sec,
            "labeled": lab,
            "n_symbols": len(syms),
            "learn_ok": ok_count,
            "learn_err": err_count,
            "learned": learned,
            "params": {"n": n, "steps": steps, "batch": batch},
        }

        # increment success counter + structured log
        job_ok(job, n=n, steps=steps, batch=batch, duration_s=total_sec, learn_ok=ok_count, learn_err=err_count)
        log_json("info", msg="cron_done", **summary)
        return summary

    except Exception as e:
        # increment fail counter + structured log + HTTP 500
        err_msg = str(e)
        job_fail(job, err=err_msg, n=n, steps=steps, batch=batch)
        log_json("error", msg="cron_failed", job=job, error=err_msg)
        raise HTTPException(status_code=500, detail=f"{job} failed: {err_msg}")

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
    return service_quant_budget_key()


async def _quant_allow(redis: Redis, max_calls_per_day: int = 1) -> bool:
    return await service_quant_allow(redis, max_calls_per_day)


async def _quant_consume(redis: Redis) -> None:
    await service_quant_consume(redis)
# --- Backfill with politeness pause between days ---
#------------ADMIN -----------------------
async def _ingest_news(
    symbol: str,
    *,
    days: int,
    limit: int,
    provider: str | None = None,
    log_tag: str = "manual",
) -> dict:
    if not fs_connect or insert_news is None:
        raise HTTPException(status_code=503, detail="Feature store unavailable for news ingest.")

    symbol_norm = symbol.strip().upper()
    provider_eff = provider or _determine_news_provider()
    since = datetime.now(timezone.utc) - timedelta(days=int(days))
    try:
        rows = await _fetch_news_articles(symbol_norm, since, provider_eff, limit=limit)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("admin_fetch_news_failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"news fetch failed: {exc}") from exc

    if not rows:
        log_json("info", msg="news_ingest_empty", symbol=symbol_norm, source=provider_eff, tag=log_tag)
        return {"rows": 0, "source": provider_eff, "symbol": symbol_norm}

    try:
        con = fs_connect()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Feature store connect failed: {exc}") from exc

    t0 = time.perf_counter()
    hash_lines: list[str] = []
    for r in rows:
        ts_norm = _as_utc_datetime(r[1]) or datetime.now(timezone.utc)
        hash_lines.append(f"{r[0]}|{ts_norm.isoformat()}|{r[2]}|{r[3]}|{r[4]}")
    payload_hash = hashlib.sha256("\n".join(hash_lines).encode("utf-8", errors="ignore")).hexdigest()

    inserted = 0
    duration_ms = 0
    try:
        inserted = insert_news(con, rows)
        duration_ms = int((time.perf_counter() - t0) * 1000)
        if log_ingest_event:
            try:
                log_ingest_event(
                    con,
                    as_of=datetime.now(timezone.utc).date(),
                    source=f"news:{provider_eff}:{log_tag}",
                    row_count=int(inserted),
                    sha256=payload_hash,
                    duration_ms=duration_ms,
                    ok=True,
                )
            except Exception as exc:
                logger.debug("log_ingest_event news ok failed: %s", exc)
        if NEWS_INGESTED_COUNTER is not None:
            NEWS_INGESTED_COUNTER.inc(max(0, int(inserted)))
        log_json(
            "info",
            msg="news_ingest_ok",
            symbol=symbol_norm,
            source=provider_eff,
            rows=int(inserted),
            duration_ms=duration_ms,
            tag=log_tag,
        )
        return {"rows": int(inserted), "source": provider_eff, "symbol": symbol_norm}
    except Exception as exc:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        if log_ingest_event:
            try:
                log_ingest_event(
                    con,
                    as_of=datetime.now(timezone.utc).date(),
                    source=f"news:{provider_eff}:{log_tag}",
                    row_count=len(rows),
                    sha256=payload_hash,
                    duration_ms=duration_ms,
                    ok=False,
                    error=str(exc),
                )
            except Exception:
                pass
        log_json(
            "error",
            msg="news_ingest_fail",
            symbol=symbol_norm,
            source=provider_eff,
            error=str(exc),
            tag=log_tag,
        )
        raise HTTPException(status_code=500, detail=f"failed to insert news: {exc}") from exc
    finally:
        try:
            con.close()
        except Exception:
            pass


async def _score_news(
    symbol: str,
    *,
    days: int,
    batch: int,
    log_tag: str = "manual",
) -> dict:
    if not fs_connect:
        raise HTTPException(status_code=503, detail="Feature store unavailable for scoring.")

    symbol_norm = symbol.strip().upper()
    cutoff = datetime.now(timezone.utc) - timedelta(days=int(days))
    use_llm = _llm_background_enabled()
    try:
        con = fs_connect()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Feature store connect failed: {exc}") from exc

    try:
        rows = con.execute(
            """
            SELECT ts, source, title, url, summary
            FROM news_articles
            WHERE symbol = ? AND ts >= ? AND (sentiment IS NULL OR sentiment = 0)
            ORDER BY ts DESC
            LIMIT ?
            """,
            [symbol_norm, cutoff, int(batch)],
        ).fetchall()
    except Exception as exc:
        try:
            con.close()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"failed to load news: {exc}") from exc

    if not rows:
        try:
            con.close()
        except Exception:
            pass
        log_json("info", msg="news_score_empty", symbol=symbol_norm, tag=log_tag)
        return {"rows": 0, "symbol": symbol_norm}

    if not use_llm:
        log_json(
            "info",
            msg="news_score_skip_llm",
            symbol=symbol_norm,
            tag=log_tag,
            reason="llm_background_disabled",
        )

    oai_key = os.getenv("OPENAI_API_KEY", "").strip()
    xai_key = os.getenv("XAI_API_KEY", "").strip()
    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    prefer_xai = (provider == "xai") or (not oai_key and bool(xai_key))

    json_schema = {
        "name": "NewsSentimentScore",
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "minLength": 20, "maxLength": 240},
                "sentiment": {"type": "number", "minimum": -1.0, "maximum": 1.0},
            },
            "required": ["summary", "sentiment"],
            "additionalProperties": False,
        },
    }

    updates: list[tuple] = []
    for ts, source, title, url_item, summary in rows:
        prompt_user = {
            "role": "user",
            "content": (
                "You are a financial analyst assistant. Analyze the following news headline for symbol "
                f"{symbol_norm} and respond with JSON containing 'summary' (one sentence, <=200 chars) and "
                "'sentiment' (float in [-1,1], where -1 is very negative for the symbol and 1 is very positive).\n\n"
                f"Source: {source}\n"
                f"Title: {title}\n"
                f"URL: {url_item}\n"
                f"Existing summary: {summary or ''}\n"
            ),
        }
        if use_llm:
            try:
                out = await llm_summarize_async(
                    prompt_user,
                    prefer_xai=prefer_xai,
                    xai_key=xai_key or None,
                    oai_key=oai_key or None,
                    json_schema=json_schema,
                )
            except Exception as exc:
                logger.debug("admin_score_news llm failed for %s: %s", symbol_norm, exc)
                out = {}
        else:
            out = {}
        cleaned_summary = (out.get("summary") if isinstance(out, dict) else "") if out else ""
        if not cleaned_summary:
            cleaned_summary = summary or title or ""
        cleaned_summary = cleaned_summary.strip()
        if len(cleaned_summary) > 240:
            cleaned_summary = cleaned_summary[:237].rstrip() + "..."
        sentiment_val = _first_number([out.get("sentiment")]) if isinstance(out, dict) else None
        if sentiment_val is None:
            sentiment_val = 0.0
        sentiment_val = float(np.clip(float(sentiment_val), -1.0, 1.0))
        ts_norm = _as_utc_datetime(ts) or datetime.now(timezone.utc)
        updates.append((
            cleaned_summary.strip(),
            sentiment_val,
            symbol_norm,
            ts_norm,
            (source or "").strip(),
            (url_item or "").strip(),
        ))

    try:
        con.executemany(
            """
            UPDATE news_articles
            SET summary = ?, sentiment = ?
            WHERE symbol = ? AND ts = ? AND source = ? AND url = ?
            """,
            updates,
        )
        updated = len(updates)
        con.close()
        if NEWS_SCORED_COUNTER is not None:
            NEWS_SCORED_COUNTER.inc(max(0, updated))
        log_json("info", msg="news_score_ok", symbol=symbol_norm, rows=updated, tag=log_tag)
        return {"rows": updated, "symbol": symbol_norm}
    except Exception as exc:
        try:
            con.close()
        except Exception:
            pass
        log_json("error", msg="news_score_fail", symbol=symbol_norm, error=str(exc), tag=log_tag)
        raise HTTPException(status_code=500, detail=f"failed to update news sentiment: {exc}") from exc


async def _ingest_earnings(
    symbol: str,
    *,
    lookback_days: int,
    limit: int,
    provider: str | None = None,
    log_tag: str = "manual",
) -> dict:
    if not fs_connect or insert_earnings is None:
        raise HTTPException(status_code=503, detail="Feature store unavailable for earnings ingest.")

    symbol_norm = symbol.strip().upper()
    provider_eff = provider or _determine_earnings_provider()
    try:
        rows = await _fetch_earnings_rows(symbol_norm, int(lookback_days), provider_eff, limit=limit)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("admin_fetch_earnings_failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"earnings fetch failed: {exc}") from exc

    if not rows:
        log_json("info", msg="earnings_ingest_empty", symbol=symbol_norm, source=provider_eff, tag=log_tag)
        return {"rows": 0, "source": provider_eff, "symbol": symbol_norm}

    try:
        con = fs_connect()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Feature store connect failed: {exc}") from exc

    t0 = time.perf_counter()
    payload_hash = hashlib.sha256(
        "\n".join(f"{r[0]}|{r[1]}|{r[2]}|{r[3]}|{r[4]}|{r[5]}" for r in rows).encode("utf-8", errors="ignore")
    ).hexdigest()

    duration_ms = 0
    try:
        inserted = insert_earnings(con, rows)
        duration_ms = int((time.perf_counter() - t0) * 1000)
        if log_ingest_event:
            try:
                log_ingest_event(
                    con,
                    as_of=datetime.now(timezone.utc).date(),
                    source=f"earnings:{provider_eff}:{log_tag}",
                    row_count=int(inserted),
                    sha256=payload_hash,
                    duration_ms=duration_ms,
                    ok=True,
                )
            except Exception as exc:
                logger.debug("log_ingest_event earnings ok failed: %s", exc)
        if EARNINGS_INGESTED_COUNTER is not None:
            EARNINGS_INGESTED_COUNTER.inc(max(0, int(inserted)))
        log_json(
            "info",
            msg="earnings_ingest_ok",
            symbol=symbol_norm,
            source=provider_eff,
            rows=int(inserted),
            duration_ms=duration_ms,
            tag=log_tag,
        )
        return {"rows": int(inserted), "source": provider_eff, "symbol": symbol_norm}
    except Exception as exc:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        if log_ingest_event:
            try:
                log_ingest_event(
                    con,
                    as_of=datetime.now(timezone.utc).date(),
                    source=f"earnings:{provider_eff}:{log_tag}",
                    row_count=len(rows),
                    sha256=payload_hash,
                    duration_ms=duration_ms,
                    ok=False,
                    error=str(exc),
                )
            except Exception:
                pass
        log_json(
            "error",
            msg="earnings_ingest_fail",
            symbol=symbol_norm,
            source=provider_eff,
            error=str(exc),
            tag=log_tag,
        )
        raise HTTPException(status_code=500, detail=f"failed to insert earnings: {exc}") from exc
    finally:
        try:
            con.close()
        except Exception:
            pass


async def _ingest_macro(
    provider: Optional[str] = None,
    *,
    log_tag: str = "manual",
) -> dict:
    if not fs_connect or upsert_macro is None:
        raise HTTPException(status_code=503, detail="Feature store unavailable for macro ingest.")

    provider_eff = (provider or settings.macro_source or "fred").strip().lower()
    try:
        rows = await _fetch_macro_rows(provider_eff)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("admin_fetch_macro_failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"macro fetch failed: {exc}") from exc

    if not rows:
        log_json("info", msg="macro_ingest_empty", source=provider_eff, tag=log_tag)
        return {"rows": 0, "source": provider_eff}

    try:
        con = fs_connect()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Feature store connect failed: {exc}") from exc

    t0 = time.perf_counter()
    payload_hash = hashlib.sha256(
        "\n".join(f"{r[0]}|{r[1]}|{r[2]}|{r[3]}" for r in rows).encode("utf-8", errors="ignore")
    ).hexdigest()

    duration_ms = 0
    try:
        inserted = upsert_macro(con, rows)
        duration_ms = int((time.perf_counter() - t0) * 1000)
        if log_ingest_event:
            try:
                log_ingest_event(
                    con,
                    as_of=rows[0][0],
                    source=f"macro:{provider_eff}:{log_tag}",
                    row_count=int(inserted),
                    sha256=payload_hash,
                    duration_ms=duration_ms,
                    ok=True,
                )
            except Exception as exc:
                logger.debug("log_ingest_event macro ok failed: %s", exc)
        if MACRO_UPSERTS_COUNTER is not None:
            MACRO_UPSERTS_COUNTER.inc(max(0, int(inserted)))
        log_json(
            "info",
            msg="macro_ingest_ok",
            source=provider_eff,
            rows=int(inserted),
            duration_ms=duration_ms,
            tag=log_tag,
        )
        return {"rows": int(inserted), "source": provider_eff}
    except Exception as exc:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        if log_ingest_event:
            try:
                log_ingest_event(
                    con,
                    as_of=rows[0][0],
                    source=f"macro:{provider_eff}:{log_tag}",
                    row_count=len(rows),
                    sha256=payload_hash,
                    duration_ms=duration_ms,
                    ok=False,
                    error=str(exc),
                )
            except Exception:
                pass
        log_json("error", msg="macro_ingest_fail", source=provider_eff, error=str(exc), tag=log_tag)
        raise HTTPException(status_code=500, detail=f"failed to upsert macro snapshot: {exc}") from exc
    finally:
        try:
            con.close()
        except Exception:
            pass


@app.post("/admin/fetch/news", summary="Fetch recent news and upsert into feature store")
async def admin_fetch_news(
    request: Request,
    symbol: str = Query(..., min_length=1, description="Ticker symbol (e.g., NVDA)"),
    days: int = Query(7, ge=1, le=30),
    limit: int = Query(60, ge=1, le=200),
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    await enforce_limits(REDIS, request, scope="cron", per_min=CRON_LIMIT_PER_MIN, cost_units=1)
    result = await _ingest_news(symbol, days=int(days), limit=int(limit), provider=None, log_tag="admin")
    return {"ok": True, **result}


@app.post("/admin/score/news", summary="LLM-score recent news sentiment")
async def admin_score_news(
    request: Request,
    symbol: str = Query(..., min_length=1),
    days: int = Query(7, ge=1, le=30),
    batch: int = Query(16, ge=1, le=64),
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    await enforce_limits(REDIS, request, scope="cron", per_min=CRON_LIMIT_PER_MIN, cost_units=1)
    result = await _score_news(symbol, days=int(days), batch=int(batch), log_tag="admin")
    return {"ok": True, **result}


@app.post("/admin/fetch/earnings", summary="Fetch earnings data and upsert into feature store")
async def admin_fetch_earnings(
    request: Request,
    symbol: str = Query(..., min_length=1),
    lookback_days: int = Query(370, ge=30, le=1825),
    limit: int = Query(12, ge=1, le=40),
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    await enforce_limits(REDIS, request, scope="cron", per_min=CRON_LIMIT_PER_MIN, cost_units=1)
    result = await _ingest_earnings(
        symbol,
        lookback_days=int(lookback_days),
        limit=int(limit),
        provider=None,
        log_tag="admin",
    )
    return {"ok": True, **result}


@app.post("/admin/fetch/macro", summary="Fetch macro snapshot and upsert into feature store")
async def admin_fetch_macro(
    request: Request,
    provider: Optional[str] = Query(None, description="Override macro provider (default from settings)"),
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    await enforce_limits(REDIS, request, scope="cron", per_min=CRON_LIMIT_PER_MIN, cost_units=1)
    result = await _ingest_macro(provider, log_tag="admin")
    return {"ok": True, **result}

@app.post("/admin/ingest/backfill")
async def admin_ingest_backfill(days: int = 7, _ok: bool = Security(require_key, scopes=["admin"])):
    base = datetime.now(timezone.utc).date()
    total = 0
    pause_s = float(os.getenv("PT_BACKFILL_PAUSE_S", "2.5"))  # <— NEW
    for i in range(int(max(1, days))):
        out = await ingest_grouped_daily(base - timedelta(days=i))
        total += int(out.get("upserted", 0))
        await asyncio.sleep(pause_s)  # <— NEW
    return {"ok": True, "days": int(days), "rows": total}

@app.get("/admin/logs/latest", summary="Fetch recent service logs (in-memory buffer)")
async def admin_logs_latest(n: int = 200, _ok: bool = Security(require_key, scopes=["admin"])):
    try:
        items = get_recent_logs(n=n)
        return {"ok": True, "count": len(items), "logs": items}
    except Exception as e:
        log_json("error", msg="admin_logs_latest_fail", error=str(e))
        raise HTTPException(status_code=500, detail="failed to fetch logs")

@app.post("/admin/plan/set", summary="Set subscription plan for an API key (free|pro|inst)")
async def admin_plan_set(
    api_key: str = Body(..., embed=True),
    plan: str = Body(..., embed=True),
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    try:
        new_plan = await set_plan_for_key(REDIS, api_key, plan)
        log_json("info", msg="admin_plan_set", target_key=f"...{api_key[-6:]}", plan=new_plan)
        return {"ok": True, "api_key_tail": api_key[-6:], "plan": new_plan}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        log_json("error", msg="admin_plan_set_fail", error=str(e))
        raise HTTPException(status_code=500, detail="failed_to_set_plan")

@app.get("/admin/plan/get", summary="Get subscription plan for an API key")
async def admin_plan_get(api_key: str = Query(...), _ok: bool = Security(require_key, scopes=["admin"])):
    try:
        plan = await get_plan_for_key(REDIS, api_key)
        return {"ok": True, "api_key_tail": api_key[-6:], "plan": plan}
    except Exception as e:
        log_json("error", msg="admin_plan_get_fail", error=str(e))
        raise HTTPException(status_code=500, detail="failed_to_get_plan")

@app.get("/me/limits", summary="Return caller plan and usage/limits for key scopes")
async def me_limits(request: Request, _ok: bool = Depends(require_key)):
    try:
        used_sim, limit_sim, plan_sim, caller = await usage_today(REDIS, request, scope="simulate")
        used_cron, limit_cron, plan_cron = await usage_today_for_caller(REDIS, caller, scope="cron")
        # Both scopes use the same plan; prefer simulate’s read
        plan = plan_sim or plan_cron

        # seconds until UTC midnight (quota reset hint)
        now = datetime.now(timezone.utc)
        tomorrow = (now + timedelta(days=1)).date()
        reset_at = datetime.combine(tomorrow, datetime.min.time(), tzinfo=timezone.utc)
        seconds_to_reset = int((reset_at - now).total_seconds())

        payload = {
            "ok": True,
            "plan": plan,
            "caller": caller,
            "reset_secs": seconds_to_reset,
            "per_min_caps": {
                "base": BASE_LIMIT_PER_MIN,
                "simulate": SIM_LIMIT_PER_MIN,
                "cron": CRON_LIMIT_PER_MIN,
            },
            "daily": {
                "simulate": {"used": used_sim, "limit": limit_sim, "remaining": max(0, limit_sim - used_sim)},
                "cron": {"used": used_cron, "limit": limit_cron, "remaining": max(0, limit_cron - used_cron)},
            },
        }
        log_json("info", msg="me_limits", plan=plan, caller_tail=caller[-8:])
        return payload
    except Exception as e:
        log_json("error", msg="me_limits_fail", error=str(e))
        raise HTTPException(status_code=500, detail="failed_to_get_limits")

@app.post("/admin/ingest/daily")
async def admin_ingest_daily(
    d: Optional[str] = Query(None, description="YYYY-MM-DD (UTC)"),
    _ok: bool = Security(require_key, scopes=["admin"]),
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


@app.get("/context/{symbol}", summary="Return fused context for symbol (regime, sentiment, earnings, macro)")
async def context(symbol: str):
    sym = (symbol or "").strip().upper()
    if not sym:
        raise HTTPException(status_code=400, detail="Symbol required")

    try:
        px = await _fetch_cached_hist_prices(sym, 220, REDIS)
        reg = _detect_regime(np.asarray(px, dtype=float))
    except Exception as exc:
        logger.debug("context_regime_failed for %s: %s", sym, exc)
        reg = {"name": "neutral", "score": 0.0}

    sent = await get_sentiment_features(sym)
    earn = await get_earnings_features(sym)
    macr = await get_macro_features()

    return {
        "symbol": sym,
        "regime": reg,
        "sentiment": sent,
        "earnings": earn,
        "macro": macr,
    }


@app.post("/compare/backtest", response_model=BacktestResp, summary="Compare baseline vs enriched simulations (admin)")
async def compare_backtest(
    req: BacktestReq,
    request: Request,
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    await enforce_limits(REDIS, request, scope="simulate", per_min=SIM_LIMIT_PER_MIN, cost_units=1)
    try:
        start_date = date.fromisoformat(req.start)
        end_date = date.fromisoformat(req.end)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid start/end format; expected YYYY-MM-DD")

    if start_date >= end_date:
        raise HTTPException(status_code=400, detail="'start' must be earlier than 'end'")

    if not req.symbols:
        raise HTTPException(status_code=400, detail="At least one symbol required")

    horizon = int(req.horizon_days)
    max_per_symbol = int(req.limit_per_symbol or 300)
    norm = NormalDist()
    scale = 252.0

    baseline_apes: list[float] = []
    sim_apes: list[float] = []
    baseline_briers: list[float] = []
    sim_briers: list[float] = []
    baseline_cov_hits = 0
    sim_cov_hits = 0
    cov_total = 0

    z05 = norm.inv_cdf(0.05)
    z95 = norm.inv_cdf(0.95)

    for symbol in req.symbols:
        sym = (symbol or "").strip().upper()
        if not sym:
            continue
        series = _load_price_series_with_dates(sym)
        if not series:
            logger.debug("backtest: no price series for %s", sym)
            continue

        series.sort(key=lambda x: x[0])
        dates = [row[0] for row in series]
        prices = np.asarray([row[1] for row in series], dtype=float)
        if prices.size <= horizon + 5:
            logger.debug("backtest: insufficient price history for %s", sym)
            continue

        try:
            tune = rollforward_validation(
                prices,
                horizon_days=horizon,
                lookbacks=VALIDATION_LOOKBACKS,
                n_paths=4000,
                target_mape=VALIDATION_TARGET_MAPE,
                max_samples=VALIDATION_MAX_SAMPLES,
                bars_per_year=VALIDATION_BARS_PER_YEAR,
            )
            lookback = int(tune.best.lookback)
        except Exception as exc:
            logger.debug("backtest: rollforward validation failed for %s: %s", sym, exc)
            continue

        try:
            sent_ctx = await get_sentiment_features(sym)
        except Exception as exc:
            logger.debug("backtest: sentiment fetch failed for %s: %s", sym, exc)
            sent_ctx = {"avg_sent_7d": 0.0, "last24h": 0.0, "n_news": 0}
        try:
            earn_ctx = await get_earnings_features(sym)
        except Exception as exc:
            logger.debug("backtest: earnings fetch failed for %s: %s", sym, exc)
            earn_ctx = {"surprise_last": 0.0, "guidance_delta": 0.0, "days_since_earn": None, "days_to_next": None}

        sent_avg = float(sent_ctx.get("avg_sent_7d") or 0.0)
        sent_last24 = float(sent_ctx.get("last24h") or 0.0)
        earn_surprise = float(earn_ctx.get("surprise_last") or 0.0)

        upper = prices.size - horizon
        if upper - lookback <= 0:
            continue

        count_symbol = 0
        for start_idx in range(0, upper - lookback):
            eval_idx = start_idx + lookback
            actual_idx = eval_idx + horizon
            if actual_idx >= prices.size:
                break

            eval_date = dates[eval_idx]
            if eval_date < start_date or eval_date > end_date:
                continue

            window = prices[start_idx : eval_idx + 1]
            if window.size <= lookback:
                continue

            rets = np.diff(np.log(window))
            if rets.size == 0 or not np.all(np.isfinite(rets)):
                continue

            rets = winsorize(rets)
            sigma_bar = ewma_sigma(rets, lam=0.94)
            sigma_ann_raw = float(sigma_bar * math.sqrt(scale))
            sigma_ann = float(np.clip(sigma_ann_raw, 1e-4, SIGMA_CAP))

            mu_ann_raw = float(np.mean(rets) * scale)
            mu_cap = MU_CAP_QUICK
            mu_ann = float(np.clip(mu_ann_raw, -mu_cap, mu_cap))

            shrink = horizon_shrink(horizon)
            mu_ann *= shrink

            mu_ann_enriched = mu_ann
            sigma_ann_enriched = sigma_ann
            mu_bias = float(np.clip(sent_avg * 0.15 + earn_surprise * 0.05, -0.15, 0.15))
            mu_ann_enriched = float(np.clip(mu_ann_enriched + mu_bias, -mu_cap, mu_cap))
            if sent_last24 > 0.20:
                sigma_ann_enriched = float(np.clip(sigma_ann_enriched * 0.97, 1e-4, SIGMA_CAP))

            s0 = float(window[-1])
            actual_price = float(prices[actual_idx])

            def _forecast(mu_ann_val: float, sigma_ann_val: float) -> tuple[float, float, float, float]:
                mu_d = mu_ann_val / scale
                sigma_d = sigma_ann_val / math.sqrt(scale)
                std_log = sigma_d * math.sqrt(horizon)
                mean_log = math.log(s0) + (mu_d - 0.5 * sigma_d * sigma_d) * horizon
                median = float(math.exp(mean_log))
                if std_log <= 1e-9:
                    q05_val = q95_val = median
                    prob_up_val = 1.0 if median >= s0 else 0.0
                else:
                    q05_val = float(math.exp(mean_log + std_log * z05))
                    q95_val = float(math.exp(mean_log + std_log * z95))
                    mean_lr = (mu_d - 0.5 * sigma_d * sigma_d) * horizon
                    dist = NormalDist(mu=mean_lr, sigma=std_log)
                    prob_up_val = float(1.0 - dist.cdf(0.0))
                return median, q05_val, q95_val, prob_up_val

            median_base, q05_base, q95_base, prob_up_base = _forecast(mu_ann, sigma_ann)
            median_sim, q05_sim, q95_sim, prob_up_sim = _forecast(mu_ann_enriched, sigma_ann_enriched)

            denom = abs(actual_price) if abs(actual_price) > 1e-9 else 1.0
            baseline_apes.append(abs(median_base - actual_price) / denom * 100.0)
            sim_apes.append(abs(median_sim - actual_price) / denom * 100.0)

            hit_base = q05_base <= actual_price <= q95_base
            hit_sim = q05_sim <= actual_price <= q95_sim
            baseline_cov_hits += int(hit_base)
            sim_cov_hits += int(hit_sim)
            cov_total += 1

            outcome = 1.0 if actual_price > s0 else 0.0
            baseline_briers.append((prob_up_base - outcome) ** 2)
            sim_briers.append((prob_up_sim - outcome) ** 2)

            count_symbol += 1
            if count_symbol >= max_per_symbol:
                break

    if not baseline_apes:
        raise HTTPException(status_code=400, detail="No samples available for requested configuration")

    mdape_baseline = float(np.median(baseline_apes))
    mdape_sim = float(np.median(sim_apes)) if sim_apes else float("nan")
    cov_baseline = (baseline_cov_hits / cov_total) if cov_total else float("nan")
    cov_sim = (sim_cov_hits / cov_total) if cov_total else float("nan")
    brier_baseline = float(np.mean(baseline_briers)) if baseline_briers else None
    brier_sim = float(np.mean(sim_briers)) if sim_briers else None

    brier_payload = None
    if brier_baseline is not None or brier_sim is not None:
        brier_payload = {"baseline": brier_baseline, "sim": brier_sim}

    return BacktestResp(
        n=len(baseline_apes),
        mdape={"baseline": mdape_baseline, "sim": mdape_sim},
        coverage90={"baseline": cov_baseline, "sim": cov_sim},
        brier=brier_payload,
        crps=None,
    )


@app.post("/quant/daily/run")
async def quant_daily_run(horizon: int | None = None, _ok: bool = Security(require_key, scopes=["cron"])):
    # Avoid over-computation if multiple schedulers/clients call in the same day
    budget = int(os.getenv("PT_QUANT_BUDGET_PER_DAY", "6"))  # e.g., allow up to 6 runs/day
    if REDIS:
        try:
            if not await _quant_allow(REDIS, max_calls_per_day=budget):
                # Return whatever is cached instead of recomputing
                return await quant_daily_today()
        except Exception:
            pass

    out = await _run_daily_quant(horizon_days=int(horizon or settings.daily_quant_horizon_days))

    if REDIS:
        try: await _quant_consume(REDIS)
        except Exception: pass
    return out

@app.get(
    "/quant/daily/today",
    summary="Daily snapshot (dashboard) or minimal one-shot summary when symbol is provided",
)
async def quant_daily_today(
    symbol: str | None = Query(None, description="Ticker, e.g., NVDA or BTC-USD"),
    horizon_days: int = Query(30, ge=5, le=90),
    # If you want to require an API key for the minimal branch, uncomment:
    # _ok: bool = Security(require_key, scopes=["simulate"]),
):
    """
    - No symbol -> returns the existing daily snapshot payload (winners, finalists, etc.).
    - With ?symbol=... -> returns a minimal marketing JSON:
        { symbol, prob_up_30d, base_price, predicted_price, bullish_price }
      Prefers Redis prefill, falls back to on-demand compute.
    """
    d = datetime.now(timezone.utc).date().isoformat()

    # --- Minimal branch -------------------------------------------------
    if symbol:
        s = symbol.strip().upper()
        if not s:
            raise HTTPException(status_code=400, detail="Symbol cannot be blank")

        try:
            return await fetch_minimal_summary(s, horizon_days)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"compute failed for {s}: {exc}") from exc

    # --- Original dashboard snapshot branch -----------------------------
    out: dict[str, object] = {"as_of": d}
    if REDIS:
        try:
            base = f"quant:daily:{d}"
            eq = await REDIS.get(f"{base}:equity")
            cr = await REDIS.get(f"{base}:crypto")
            if eq:
                out["equity"] = json.loads(eq)
            if cr:
                out["crypto"] = json.loads(cr)

            eq_top = await REDIS.get(f"{base}:equity_top")
            cr_top = await REDIS.get(f"{base}:crypto_top")
            if eq_top:
                out["equity_top"] = json.loads(eq_top)
            if cr_top:
                out["crypto_top"] = json.loads(cr_top)

            eq_all = await REDIS.get(f"{base}:equity_finalists")
            cr_all = await REDIS.get(f"{base}:crypto_finalists")
            if eq_all:
                out["equity_finalists"] = json.loads(eq_all)
            if cr_all:
                out["crypto_finalists"] = json.loads(cr_all)
        except Exception:
            pass

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
async def get_run_summary(run_id: str, request: Request, refresh: int = 0, _ok: bool = Depends(require_key)):
    """
    Public: returns cached LLM summary for a completed run_id.
    - 202 while artifact isn't persisted yet
    - ?refresh=1 recomputes and overwrites cache
    """
    await _ensure_run_owned(run_id, request)
    try:
        return await _summarize_run(run_id, force=bool(refresh))
    except HTTPException as e:
        if e.status_code == 404:
            return JSONResponse({"status": "pending"}, status_code=202)
        raise

@app.get("/runs/recent", summary="Recent simulation runs")
async def runs_recent(
    limit: int = Query(8, ge=1, le=50),
    symbol: str | None = Query(None),
    _ok: bool = Depends(require_key),
) -> dict:
    try:
        rows = recent_predictions(symbol=symbol, limit=int(limit))
    except Exception as exc:
        log_json("error", msg="runs_recent_fail", error=str(exc))
        raise HTTPException(status_code=500, detail="failed_to_fetch_recent_runs") from exc

    items: list[dict] = []
    for row in rows:
        ctx_raw = row.get("user_ctx")
        ctx_obj: dict | None = None
        if isinstance(ctx_raw, str):
            try:
                ctx_obj = json.loads(ctx_raw)
            except Exception:
                ctx_obj = None
        elif isinstance(ctx_raw, dict):
            ctx_obj = ctx_raw

        n_paths: int | None = None
        if isinstance(ctx_obj, dict):
            maybe_paths = ctx_obj.get("n_paths") or ctx_obj.get("paths")
            if isinstance(maybe_paths, (int, float)) and math.isfinite(maybe_paths):
                n_paths = int(maybe_paths)

        prob_up = row.get("prob_up_next")
        if isinstance(prob_up, (int, float)) and not math.isfinite(prob_up):
            prob_up = None

        items.append(
            {
                "id": row.get("run_id") or row.get("pred_id"),
                "run_id": row.get("run_id"),
                "pred_id": row.get("pred_id"),
                "symbol": row.get("symbol"),
                "horizon_days": row.get("horizon_d"),
                "model_id": row.get("model_id"),
                "prob_up_end": prob_up,
                "prob_up_next": prob_up,
                "p05": row.get("p05"),
                "p50": row.get("p50"),
                "p95": row.get("p95"),
                "spot0": row.get("spot0"),
                "n_paths": n_paths,
                "ts": row.get("ts"),
                "finished_at": row.get("ts"),
                "finishedAt": row.get("ts"),
            }
        )
    return {"items": items}

@app.post("/outcomes/label")
async def outcomes_label(limit: int = 5000, _ok: bool = Depends(require_key)):
    """
    Label any matured predictions (ts + horizon_d <= now) that don't yet have outcomes.
    Writes to src.db.duck.outcomes and (optionally) mirrors each label into PathPanda FS
    so daily metrics rollups can see realized prices.
    """
    stats = await service_label_outcomes(limit)
    return stats

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


@app.post("/analytics/export/predictions")
async def analytics_export_predictions(
    day: Optional[str] = Query(None, description="Date YYYY-MM-DD; defaults to today"),
    symbol: Optional[str] = Query(None),
    limit: int = Query(50_000, ge=100, le=500_000),
    push_to_s3: bool = Query(False),
    _ok: bool = Depends(require_key),
):
    if export_predictions_parquet is None:
        raise HTTPException(status_code=503, detail="export_predictions_parquet unavailable")
    try:
        day_obj = date.fromisoformat(day) if day else datetime.now(timezone.utc).date()
    except ValueError:
        raise HTTPException(status_code=400, detail="day must be YYYY-MM-DD")

    dest_dir = EXPORT_ROOT / "manual" / "predictions"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"predictions_{day_obj.isoformat()}.parquet"
    try:
        path = export_predictions_parquet(dest, symbol=symbol, day=day_obj, limit=limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    s3_uri = _maybe_upload_to_s3(Path(path)) if push_to_s3 else None
    return {"path": str(path), "s3_uri": s3_uri}


@app.post("/analytics/export/outcomes")
async def analytics_export_outcomes(
    day: Optional[str] = Query(None, description="Date YYYY-MM-DD; defaults to today"),
    symbol: Optional[str] = Query(None),
    limit: int = Query(50_000, ge=100, le=500_000),
    push_to_s3: bool = Query(False),
    _ok: bool = Depends(require_key),
):
    if export_outcomes_parquet is None:
        raise HTTPException(status_code=503, detail="export_outcomes_parquet unavailable")
    try:
        day_obj = date.fromisoformat(day) if day else datetime.now(timezone.utc).date()
    except ValueError:
        raise HTTPException(status_code=400, detail="day must be YYYY-MM-DD")

    dest_dir = EXPORT_ROOT / "manual" / "outcomes"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"outcomes_{day_obj.isoformat()}.parquet"
    try:
        path = export_outcomes_parquet(dest, symbol=symbol, day=day_obj, limit=limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    s3_uri = _maybe_upload_to_s3(Path(path)) if push_to_s3 else None
    return {"path": str(path), "s3_uri": s3_uri}


@app.get("/analytics/metrics/summary")
async def analytics_metrics_summary(
    symbol: Optional[str] = Query(None),
    horizon_days: Optional[int] = Query(None),
    limit: int = Query(90, ge=1, le=365),
    _ok: bool = Depends(require_key),
):
    if fs_connect is None:
        raise HTTPException(status_code=503, detail="feature_store_unavailable")
    con = fs_connect()
    try:
        table = fetch_metrics_daily_arrow(
            con,
            symbol=symbol,
            horizon_days=horizon_days,
            limit=limit,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    finally:
        try:
            con.close()
        except Exception:
            pass

    if table.num_rows == 0:
        return {"summary": {"count": 0}, "rows": []}

    rows = table.to_pylist()
    total_n = sum(int(r.get("n") or 0) for r in rows)
    avg_brier = None
    avg_rmse = None
    avg_mape = None
    coverage = None
    if rows:
        briers = [float(r["brier"]) for r in rows if r.get("brier") is not None]
        rmses = [float(r["rmse"]) for r in rows if r.get("rmse") is not None]
        mapes = [float(r["mape"]) for r in rows if r.get("mape") is not None]
        if briers:
            avg_brier = float(np.mean(briers))
        if rmses:
            avg_rmse = float(np.mean(rmses))
        if mapes:
            avg_mape = float(np.mean(mapes))
    if total_n:
        coverage = sum(
            float(r.get("p90_cov") or 0.0) * int(r.get("n") or 0) for r in rows
        ) / total_n

    summary = {
        "count": len(rows),
        "total_samples": total_n,
        "avg_brier": avg_brier,
        "avg_rmse": avg_rmse,
        "avg_mape": avg_mape,
        "weighted_p90_cov": coverage,
    }
    return {"summary": summary, "rows": rows}


@app.get("/admin/reports/system-health")
async def admin_report_system_health(
    request: Request,
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    uptime = datetime.now(timezone.utc) - START_TIME
    result = {
        "service": "simetrix-api",
        "version": os.getenv("PT_RELEASE", "dev"),
        "uptime_seconds": int(uptime.total_seconds()),
        "start_time": START_TIME.isoformat(),
        "redis": bool(REDIS),
        "duckdb_path": str(FS_DB_PATH) if "FS_DB_PATH" in globals() else None,
        "arrow_root": str(ARROW_ROOT),
        "watch_equities": len(WL_EQ),
        "watch_cryptos": len(WL_CR),
    }
    log_json(
        "info",
        msg="admin_report_system_health",
        actor=_admin_actor(request),
        uptime_seconds=result["uptime_seconds"],
    )
    return result


@app.get("/admin/reports/simulations")
async def admin_report_simulations(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    symbol: Optional[str] = Query(None),
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    try:
        rows = recent_predictions(symbol=symbol, limit=limit)
    except Exception as exc:
        log_json(
            "error",
            msg="admin_report_simulations_fail",
            actor=_admin_actor(request),
            error=str(exc),
            symbol=symbol,
        )
        raise HTTPException(status_code=500, detail=f"failed to fetch simulations: {exc}") from exc
    payload = {"count": len(rows), "items": rows}
    log_json(
        "info",
        msg="admin_report_simulations",
        actor=_admin_actor(request),
        count=len(rows),
        symbol=symbol or "*",
    )
    return payload


async def _scan_model_meta(pattern: str = "model_meta:*", limit: int = 500) -> list[dict]:
    if not REDIS:
        return []
    results: list[dict] = []
    try:
        async for key in REDIS.scan_iter(match=pattern, count=200):
            try:
                raw = await REDIS.get(key)
                if not raw:
                    continue
                if isinstance(raw, (bytes, bytearray)):
                    raw = raw.decode("utf-8", errors="replace")
                doc = json.loads(raw)
                doc["key"] = key
                results.append(doc)
            except Exception:
                continue
            if len(results) >= limit:
                break
    except Exception:
        return []
    return results


@app.get("/admin/reports/model-training")
async def admin_report_model_training(
    request: Request,
    limit: int = Query(200, ge=10, le=1000),
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    meta = await _scan_model_meta(limit=limit)
    meta.sort(key=lambda x: x.get("trained_at", ""), reverse=True)
    payload = {"count": len(meta), "items": meta[:limit]}
    log_json(
        "info",
        msg="admin_report_model_training",
        actor=_admin_actor(request),
        count=payload["count"],
    )
    return payload


@app.get("/admin/reports/usage")
async def admin_report_usage(
    request: Request,
    scope: str = Query("simulate"),
    sample: int = Query(20, ge=1, le=200),
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    records: list[dict] = []
    now = datetime.now(timezone.utc)
    if not REDIS:
        payload = {"records": records, "note": "Redis unavailable; usage data not captured."}
        log_json(
            "warning",
            msg="admin_report_usage_unavailable",
            actor=_admin_actor(request),
            scope=scope,
        )
        return payload
    try:
        async for key in REDIS.scan_iter(match=f"qt:{scope}:*", count=500):
            raw = await REDIS.get(key)
            if not raw:
                continue
            parts = key.split(":")
            if len(parts) < 4:
                continue
            _, _, caller, ymd = parts
            used = int(raw)
            records.append({"caller": caller, "date": ymd, "used": used})
            if len(records) >= sample:
                break
    except Exception as exc:
        log_json(
            "error",
            msg="admin_report_usage_fail",
            actor=_admin_actor(request),
            scope=scope,
            error=str(exc),
        )
        return {"records": [], "note": f"Usage fetch failed: {exc}"}
    payload = {"generated_at": now.isoformat(), "records": records}
    log_json(
        "info",
        msg="admin_report_usage",
        actor=_admin_actor(request),
        scope=scope,
        count=len(records),
    )
    return payload


@app.get("/admin/reports/telemetry")
async def admin_report_telemetry(
    request: Request,
    limit: int = Query(200, ge=10, le=1000),
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    headers = ["timestamp", "level", "message"]
    timestamp = datetime.now(timezone.utc).isoformat()
    rows = [
        [timestamp, "INFO", "Telemetry export not yet implemented; hook into real store to enrich."],
    ]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(headers)
    writer.writerows(rows[:limit])
    buffer.seek(0)
    log_json(
        "info",
        msg="admin_report_telemetry",
        actor=_admin_actor(request),
        rows=min(len(rows), limit),
    )
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=telemetry_{timestamp[:10]}.csv"
        },
    )

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
    try:
        cols_now = {
            row[1]
            for row in con.execute("PRAGMA table_info('predictions')").fetchall()
        }
    except Exception:
        cols_now = set()
    issued_expr = "p.issued_at" if "issued_at" in cols_now else "p.ts"
    rows = con.execute(
        f"""
        SELECT
            p.model_id, p.symbol, {issued_expr} AS issued_at, p.horizon_days,
            p.features_ref, p.q50, p.yhat_mean,
            o.realized_at, o.y AS realized_price
        FROM predictions p
        JOIN outcomes o USING (run_id)
        WHERE p.symbol = ?
        ORDER BY o.realized_at DESC
        LIMIT ?
    """,
        [symbol.upper(), limit],
    ).fetchall()
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


def _build_training_dataset(
    prices: Sequence[float],
    *,
    horizon: int = 1,
    min_history: int = 60,
) -> tuple[list[str], np.ndarray, np.ndarray, str]:
    arr = np.asarray(
        [float(p) for p in prices if isinstance(p, (int, float)) and math.isfinite(p)],
        dtype=float,
    )
    if arr.size <= max(min_history + horizon, 10):
        raise ValueError("Not enough price history to build training dataset")

    feature_list = ["mom_20", "rvol_20", "autocorr_5"]
    rows: list[list[float]] = []
    targets: list[float] = []

    upper = arr.size - horizon
    start_idx = max(min_history, 1)

    for idx in range(start_idx, upper):
        window = arr[: idx + 1]
        feats = _basic_features_from_array(window)
        rows.append([float(feats.get(f, 0.0)) for f in feature_list])
        future_price = arr[idx + horizon]
        current_price = arr[idx]
        targets.append(1.0 if future_price > current_price else 0.0)

    if not rows:
        raise ValueError("No feature rows generated for training dataset")

    X = np.asarray(rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    dataset_hash = hashlib.sha1(arr.tobytes()).hexdigest()
    return feature_list, X, y, dataset_hash

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
    polygon_key: Optional[str] = Header(None, alias="X-Polygon-Key"),  
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

#---------Train helpers--------------------
def _parse_trained_at(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _onnx_supported() -> bool:
    return (
        onnx is not None
        and onnx_helper is not None
        and TensorProto is not None
        and ort is not None
    )


def _onnx_model_path(symbol: str) -> Path:
    return Path("models") / f"{symbol.upper()}_linear.onnx"


def _export_linear_onnx(symbol: str, weights: Sequence[float], feature_list: Sequence[str]) -> Path | None:
    if not _onnx_supported():
        return None
    try:
        sym_up = symbol.upper()
        n_features = len(feature_list)
        if n_features == 0 or len(weights) != n_features + 1:
            return None

        w = np.asarray(weights[1:], dtype=np.float32).reshape(1, n_features)
        b = np.asarray([weights[0]], dtype=np.float32)

        input_tensor = onnx_helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, n_features])
        output_tensor = onnx_helper.make_tensor_value_info("prob", TensorProto.FLOAT, [None, 1])
        weight_init = onnx_helper.make_tensor("W", TensorProto.FLOAT, w.shape, w.flatten())
        bias_init = onnx_helper.make_tensor("B", TensorProto.FLOAT, b.shape, b.flatten())

        nodes = [
            onnx_helper.make_node("MatMul", ["features", "W"], ["matmul_out"], name="linear_matmul"),
            onnx_helper.make_node("Add", ["matmul_out", "B"], ["logits"], name="linear_bias"),
            onnx_helper.make_node("Sigmoid", ["logits"], ["prob"], name="linear_sigmoid"),
        ]

        graph = onnx_helper.make_graph(
            nodes,
            name="simetrix_linear",
            inputs=[input_tensor],
            outputs=[output_tensor],
            initializer=[weight_init, bias_init],
        )
        model = onnx_helper.make_model(graph, producer_name="simetrix-linear")
        onnx.checker.check_model(model)

        path = _onnx_model_path(sym_up)
        path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, path)
        path = path.resolve()
        _ONNX_SESSION_CACHE.pop(sym_up, None)
        return path
    except Exception as exc:
        logger.debug("ONNX export skipped for %s: %s", symbol, exc)
        return None


def _load_linear_onnx(symbol: str):
    if ort is None:
        return None
    sym = symbol.upper()
    session = _ONNX_SESSION_CACHE.get(sym)
    if session is not None:
        return session
    candidate_paths: list[Path] = []
    try:
        entry = get_active_model_version("linear", sym)
    except Exception as exc:
        logger.debug("Model registry lookup failed for linear %s: %s", sym, exc)
        entry = None
    if entry and entry.get("artifact_path"):
        try:
            candidate_paths.append(_resolve_artifact_path(entry["artifact_path"]))
        except Exception as exc:
            logger.debug("Unable to resolve registry artifact for %s: %s", sym, exc)
    candidate_paths.append(_onnx_model_path(sym))
    seen: set[Path] = set()
    for path in candidate_paths:
        try:
            candidate = Path(path)
        except TypeError:
            continue
        try:
            candidate = _resolve_artifact_path(candidate)
        except Exception:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.exists():
            continue
        try:
            session = ort.InferenceSession(candidate.as_posix(), providers=["CPUExecutionProvider"])
            _ONNX_SESSION_CACHE[sym] = session
            return session
        except Exception as exc:
            logger.debug("Failed to load ONNX model for %s from %s: %s", sym, candidate, exc)
    return None


def _run_linear_onnx(symbol: str, feature_vector: Sequence[float]) -> float | None:
    session = _load_linear_onnx(symbol)
    if session is None:
        return None
    try:
        X = np.asarray([feature_vector], dtype=np.float32)
        output = session.run(None, {"features": X})
        prob = float(output[0][0][0])
        return float(np.clip(prob, 0.0, 1.0))
    except Exception as exc:
        logger.debug("ONNX inference failed for %s: %s", symbol, exc)
        return None


def _lstm_saved_model_dir(symbol: str) -> Path:
    return Path("models") / f"{symbol.upper()}_lstm.savedmodel"


def _export_lstm_savedmodel(symbol: str, model: Any) -> Path | None:
    if not TF_AVAILABLE or model is None:
        return None
    sym = symbol.upper()
    path = _lstm_saved_model_dir(sym)
    try:
        if path.exists():
            shutil.rmtree(path)
    except Exception as exc:
        logger.debug("Unable to prune previous SavedModel for %s: %s", sym, exc)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        exporter = getattr(model, "export", None)
        if callable(exporter):
            exporter(path.as_posix())
        else:
            import tensorflow as tf  # type: ignore

            tf.saved_model.save(model, path.as_posix())
        _LSTM_INFER_CACHE.pop(sym, None)
        return path.resolve()
    except Exception as exc:
        logger.debug("LSTM SavedModel export skipped for %s: %s", sym, exc)
        return None


def _load_lstm_predictor(symbol: str) -> Callable[[np.ndarray], float] | None:
    sym = symbol.upper()
    cached = _LSTM_INFER_CACHE.get(sym)
    if cached is not None:
        return cached
    try:
        model = load_lstm_model(sym)
    except HTTPException:
        return None
    if model is None:
        return None
    try:
        import tensorflow as tf  # type: ignore
    except Exception as exc:
        logger.debug("TensorFlow unavailable for LSTM inference: %s", exc)
        return None

    if hasattr(model, "signatures"):
        signature = getattr(model, "signatures", {}).get("serving_default")
        if signature is not None:

            def infer_fn(inputs: np.ndarray) -> float:
                tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
                outputs = signature(tensor)
                first = next(iter(outputs.values()))
                return float(np.asarray(first)[0][0])

            _LSTM_INFER_CACHE[sym] = infer_fn
            return infer_fn

    if hasattr(model, "predict"):

        def infer_fn(inputs: np.ndarray) -> float:
            preds = model.predict(inputs, verbose=0)
            return float(np.asarray(preds)[0][0])

        _LSTM_INFER_CACHE[sym] = infer_fn
        return infer_fn

    if callable(model):

        def infer_fn(inputs: np.ndarray) -> float:
            tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)
            preds = model(tensor, training=False)
            return float(np.asarray(preds)[0][0])

        _LSTM_INFER_CACHE[sym] = infer_fn
        return infer_fn

    return None


def _run_lstm_savedmodel(symbol: str, feature_vector: Sequence[float]) -> float | None:
    infer_fn = _load_lstm_predictor(symbol)
    if infer_fn is None:
        return None
    arr = np.asarray(feature_vector, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    inputs = arr[np.newaxis, np.newaxis, :]
    sym = symbol.upper()
    try:
        prob = float(infer_fn(inputs))
        return float(np.clip(prob, 0.0, 1.0))
    except Exception as exc:
        logger.debug("LSTM inference failed for %s: %s", sym, exc)
        _LSTM_INFER_CACHE.pop(sym, None)
        return None


def _train_lstm_blocking(
    sym: str,
    feature_list: Sequence[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    model_lstm = Sequential(
        [
            LSTM(64, input_shape=(1, len(feature_list)), return_sequences=True),
            GRU(32),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model_lstm.compile(optimizer="adam", loss="binary_crossentropy")
    X_lstm_train = np.expand_dims(X_train.astype(np.float32), axis=1)
    y_lstm_train = y_train.astype(np.float32)
    X_lstm_val = np.expand_dims(X_val.astype(np.float32), axis=1)
    y_lstm_val = y_val.astype(np.float32)
    model_lstm.fit(X_lstm_train, y_lstm_train, epochs=5, batch_size=32, verbose=0)
    val_probs_lstm = model_lstm.predict(X_lstm_val, verbose=0).ravel()
    val_preds_lstm = (val_probs_lstm >= 0.5).astype(float)
    lstm_acc = float(np.mean(val_preds_lstm == y_lstm_val))
    lstm_brier = float(np.mean((val_probs_lstm - y_lstm_val) ** 2))
    keras_path = Path("models") / f"{sym}_lstm.keras"
    model_lstm.save(keras_path.as_posix())
    savedmodel_dir = _export_lstm_savedmodel(sym, model_lstm)
    lstm_meta = {
        "val_accuracy": lstm_acc,
        "val_brier": lstm_brier,
        "epochs": 5,
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "keras_path": keras_path.as_posix(),
    }
    return lstm_meta, keras_path.as_posix(), savedmodel_dir.as_posix() if savedmodel_dir else None


def _train_arima_blocking(
    sym: str,
    arr: np.ndarray,
    train_idx: int,
    val_size: int,
) -> dict[str, Any]:
    order = (5, 1, 0) if len(arr) >= 60 else (1, 1, 0)
    train_prices = arr[: train_idx + 1]
    arima_model = ARIMA(train_prices, order=order).fit()
    forecast_steps = min(val_size, max(1, len(arr) - (train_idx + 1)))
    mae = None
    if forecast_steps > 0:
        forecast_vals = arima_model.forecast(steps=forecast_steps)
        actual_vals = arr[train_idx + 1 : train_idx + 1 + forecast_steps]
        if actual_vals.size == forecast_vals.shape[0]:
            mae = float(np.mean(np.abs(forecast_vals - actual_vals)))
    final_arima = ARIMA(arr, order=order).fit()
    with open(f"models/{sym}_arima.pkl", "wb") as file:
        pickle.dump(final_arima, file)
    return {
        "order": order,
        "aic": float(final_arima.aic) if hasattr(final_arima, "aic") else None,
        "mae": mae,
        "n_obs": int(len(arr)),
    }


def _train_rl_blocking(sym: str, prices: Sequence[float]) -> dict[str, Any]:
    env = StockEnv(prices, window_len=RL_WINDOW)
    try:
        model_rl = DQN("MlpPolicy", env, verbose=0)
        model_rl.learn(total_timesteps=10_000)
        model_rl.save(f"models/{sym}_rl.zip")
    finally:
        env.close()
    return {"total_timesteps": 10_000}


def _predict_lstm_prob_blocking(symbol: str, feature_vector: Sequence[float]) -> float:
    lstm_model = load_lstm_model(symbol)
    X = np.array(feature_vector, dtype=np.float32)
    X_lstm = np.expand_dims(np.expand_dims(X, axis=0), axis=0)
    p_lstm = float(lstm_model.predict(X_lstm, verbose=0)[0][0])
    return float(np.clip(p_lstm, 0.0, 1.0))


def _predict_arima_direction_blocking(symbol: str, prices: Sequence[float], horizon_days: int) -> float | None:
    arima_model = load_arima_model(symbol)
    fc = arima_model.forecast(steps=max(1, horizon_days))
    if hasattr(fc, "iloc"):
        last_fc = float(fc.iloc[-1])
    else:
        last_fc = float(fc[-1])
    last_price = float(prices[-1])
    return 1.0 if last_fc > last_price else 0.0


def _predict_rl_adjust_blocking(symbol: str, prices: Sequence[float], window_len: int) -> float:
    rl_model = load_rl_model(symbol)
    env = StockEnv(prices, window_len=window_len)
    try:
        obs, _ = env.reset()
        action, _ = rl_model.predict(obs, deterministic=True)
        a_map = {-1: -0.01, 0: 0.0, 1: 0.01}
        a_idx = int(action)
        a_val = a_map.get(a_idx - 1, 0.0)
        return float(np.clip(a_val, -0.05, 0.05))
    finally:
        env.close()


async def _train_models(symbol: str, *, lookback_days: int, min_samples: int = 80) -> dict[str, Any]:
    if not REDIS:
        raise HTTPException(status_code=503, detail="redis_unavailable")

    sym = (symbol or "").upper().strip()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol_required")

    lookback_days = int(max(30, min(lookback_days, settings.lookback_days_max)))
    px = await _fetch_hist_prices(sym, window_days=lookback_days)
    if not px or len(px) < 10:
        raise HTTPException(status_code=400, detail="Not enough price history")

    os.makedirs("models", exist_ok=True)

    min_samples = max(30, int(min_samples))

    trained_at = datetime.now(timezone.utc).isoformat()
    arr = np.asarray(px, dtype=float)
    try:
        feature_list, X_all, y_all, dataset_hash = _build_training_dataset(arr, horizon=1)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    n_samples = X_all.shape[0]
    if n_samples < min_samples:
        logger.warning(
            "training_samples_low",
            extra={"symbol": sym, "n_samples": n_samples, "min_required": min_samples},
        )
        if n_samples < 30:
            raise HTTPException(status_code=400, detail="Insufficient samples for model training")

    min_train = 20
    min_val = 8
    val_size = max(min_val, int(n_samples * 0.2))
    if val_size >= n_samples - min_train:
        val_size = max(min_val, min(n_samples - min_train, max(min_val, n_samples // 4)))
    train_idx = n_samples - val_size
    if train_idx < min_train or val_size < min_val:
        raise HTTPException(status_code=400, detail="Insufficient samples for validation split")

    X_train = np.nan_to_num(X_all[:train_idx], nan=0.0, posinf=0.0, neginf=0.0)
    y_train = y_all[:train_idx]
    X_val = np.nan_to_num(X_all[train_idx:], nan=0.0, posinf=0.0, neginf=0.0)
    y_val = y_all[train_idx:]

    model_linear = SGDOnline(lr=0.05, l2=1e-4)
    model_linear.init(len(feature_list))
    epochs = 3
    for _ in range(epochs):
        for x_row, yi in zip(X_train, y_train):
            model_linear.update(x_row, yi)

    val_probs = np.array([model_linear.proba(row) for row in X_val], dtype=float)
    val_preds = (val_probs >= 0.5).astype(float)
    linear_val_accuracy = float(np.mean(val_preds == y_val))
    linear_val_brier = float(np.mean((val_probs - y_val) ** 2))

    weights = model_linear.w.tolist()
    linear_data = {
        "coef": weights,
        "features": feature_list,
        "trained_at": trained_at,
        "n_train": int(train_idx),
        "n_val": int(val_size),
        "lookback_days": int(lookback_days),
        "symbol": sym,
        "val_accuracy": linear_val_accuracy,
        "val_brier": linear_val_brier,
        "dataset_hash": dataset_hash,
        "epochs": epochs,
    }
    version_suffix = (dataset_hash or uuid4().hex)[:8]
    model_version = f"{trained_at}-{version_suffix}"
    linear_data["version"] = model_version
    onnx_path = await asyncio.to_thread(_export_linear_onnx, sym, weights, feature_list)
    if onnx_path:
        onnx_path = Path(onnx_path)
        linear_data["onnx_path"] = onnx_path.as_posix()
        try:
            await asyncio.to_thread(
                register_model_version,
                "linear",
                sym,
                model_version,
                onnx_path.as_posix(),
                "onnx",
                metadata={
                    "trained_at": trained_at,
                    "dataset_hash": dataset_hash,
                    "val_accuracy": linear_val_accuracy,
                    "val_brier": linear_val_brier,
                    "n_train": int(train_idx),
                    "n_val": int(val_size),
                    "lookback_days": int(lookback_days),
                },
                status="active",
            )
        except Exception as exc:
            logger.debug("Model registry update failed for linear %s: %s", sym, exc)
    await REDIS.set(await _model_key(sym + "_linear"), json.dumps(linear_data))

    models_trained = 1

    lstm_meta: dict[str, Any] | None = None
    if Sequential is not None and X_train.shape[0] >= 100:
        try:
            lstm_meta, _, savedmodel_dir = await asyncio.to_thread(
                _train_lstm_blocking,
                sym,
                feature_list,
                X_train,
                y_train,
                X_val,
                y_val,
            )
            if lstm_meta:
                if savedmodel_dir:
                    lstm_meta["saved_model_dir"] = savedmodel_dir
                models_trained += 1
                if savedmodel_dir:
                    try:
                        await asyncio.to_thread(
                            register_model_version,
                            "lstm",
                            sym,
                            model_version,
                            savedmodel_dir,
                            "tf-savedmodel",
                            metadata={
                                "trained_at": trained_at,
                                "dataset_hash": dataset_hash,
                                "val_accuracy": float(lstm_meta.get("val_accuracy", 0)),
                                "val_brier": float(lstm_meta.get("val_brier", 0)),
                                "n_train": int(X_train.shape[0]),
                                "n_val": int(X_val.shape[0]),
                                "lookback_days": int(lookback_days),
                            },
                            status="active",
                        )
                    except Exception as exc:
                        logger.debug("Model registry update failed for lstm %s: %s", sym, exc)
        except Exception as e:
            logger.warning(f"LSTM skipped: {e}")
            lstm_meta = None
        finally:
            _LSTM_MODEL_CACHE.pop(sym, None)
            _LSTM_INFER_CACHE.pop(sym, None)

    arima_meta: dict[str, Any] | None = None
    try:
        arima_meta = await asyncio.to_thread(
            _train_arima_blocking,
            sym,
            arr,
            train_idx,
            val_size,
        )
        models_trained += 1
    except Exception as e:
        logger.warning(f"ARIMA skipped: {e}")
        arima_meta = None
    finally:
        _ARIMA_MODEL_CACHE.pop(sym, None)

    rl_meta: dict[str, Any] | None = None
    try:
        if gym is None or DQN is None:
            raise RuntimeError("RL libs not available")
        rl_meta = await asyncio.to_thread(_train_rl_blocking, sym, px)
        models_trained += 1
    except Exception as e:
        logger.warning(f"RL skipped: {e}")
        rl_meta = None
    finally:
        _RL_MODEL_CACHE.pop(sym, None)

    meta_payload = {
        "symbol": sym,
        "trained_at": trained_at,
        "version": model_version,
        "dataset_hash": dataset_hash,
        "n_samples": int(n_samples),
        "lookback_days": int(lookback_days),
        "linear": {
            "val_accuracy": linear_val_accuracy,
            "val_brier": linear_val_brier,
            "n_train": int(train_idx),
            "n_val": int(val_size),
        },
    }
    if lstm_meta:
        meta_payload["lstm"] = lstm_meta
    if arima_meta:
        meta_payload["arima"] = arima_meta
    if rl_meta:
        meta_payload["rl"] = rl_meta

    try:
        await REDIS.set(f"model_meta:{sym}", json.dumps(meta_payload))
    except Exception:
        pass
    _MODEL_META_CACHE[sym] = meta_payload

    return {
        "status": "ok",
        "symbol": sym,
        "models_trained": int(models_trained),
        "lookback_days": int(lookback_days),
        "trained_at": trained_at,
        "model_version": model_version,
        "n_samples": int(n_samples),
        "val_accuracy": linear_val_accuracy,
        "val_brier": linear_val_brier,
        "onnx_path": str(onnx_path) if onnx_path else None,
    }


async def _ensure_trained_models(symbol: str, *, required_lookback: int) -> None:
    if not REDIS:
        raise HTTPException(status_code=503, detail="redis_unavailable")

    sym = (symbol or "").upper().strip()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol_required")

    required_lookback = int(max(30, min(required_lookback, settings.lookback_days_max)))

    try:
        model_key = await _model_key(sym + "_linear")
    except TypeError:
        model_key = _model_key(sym + "_linear")

    raw = await REDIS.get(model_key)
    if raw:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except Exception:
            data = {}

        stored_lookback = int(data.get("lookback_days") or 0)
        trained_at = _parse_trained_at(data.get("trained_at"))

        is_fresh = True
        if TRAIN_REFRESH_DELTA:
            if trained_at is None:
                is_fresh = False
            else:
                age = datetime.now(timezone.utc) - trained_at
                is_fresh = age <= TRAIN_REFRESH_DELTA

        if stored_lookback >= required_lookback and is_fresh:
            return

    min_samples = 80 if required_lookback >= 365 else max(40, required_lookback // 3)
    await _train_models(sym, lookback_days=required_lookback, min_samples=min_samples)


@app.post("/train")
async def train(req: TrainRequest, _api_key: str = Depends(require_key)):
    lookback = int(req.lookback_days)
    min_samples = 80 if lookback >= 365 else max(40, lookback // 3)
    return await _train_models(req.symbol, lookback_days=lookback, min_samples=min_samples)


@app.get("/models/{model_group}/{symbol}/active")
async def model_active_version(
    model_group: str,
    symbol: str,
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    entry = get_active_model_version(model_group, symbol)
    if not entry:
        raise HTTPException(status_code=404, detail="active_model_not_found")
    return entry


@app.get("/models/{model_group}/{symbol}/versions")
async def model_versions(
    model_group: str,
    symbol: str,
    limit: int = Query(20, ge=1, le=200),
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    versions = list_model_versions(model_group, symbol, limit=limit)
    return {"versions": versions}


@app.post("/models/{model_group}/{symbol}/promote")
async def promote_model_endpoint(
    model_group: str,
    symbol: str,
    req: PromoteModelRequest,
    _ok: bool = Security(require_key, scopes=["admin"]),
):
    try:
        promote_model_version(model_group, symbol, req.version)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    entry = get_active_model_version(model_group, symbol)
    return {"status": "ok", "active": entry}


if gym is not None:
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
else:
    StockEnv = None


@app.post("/learn/online")
async def learn_online(req: OnlineLearnRequest, _api_key: str = Depends(require_key)):
    return await service_learn_online(req)

@app.post("/predict")
async def predict(req: PredictRequest, _api_key: str = Depends(require_key)):
    load_learners()
    load_tensorflow()
    """
    Returns an ensemble probability that the next move is up, logs the prediction to DuckDB,
    and mirrors the row into the PathPanda Feature Store so dashboards/metrics see it.
    """
    start_time = time.perf_counter()
    horizon_days = max(1, int(getattr(req, "horizon_days", 1)))
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
    spot0 = float(px[-1])

    # --- 3) Build input vector for linear head ---
    feature_list = model_linear.get("features", ["mom_20", "rvol_20", "autocorr_5"])
    feature_vector = [float(f.get(feat, 0.0)) for feat in feature_list]
    X = np.array(feature_vector, dtype=float)

    preds: List[float] = []
    used_models: set[str] = set()

    # --- Linear probability via logistic(dot(w, [1, X])) ---
    linear_prob = _run_linear_onnx(symbol, feature_vector)
    if linear_prob is None:
        try:
            w = np.array(model_linear.get("coef", []), dtype=float)
            if w.size:
                xb = np.concatenate([[1.0], X])            # prepend bias
                k = min(w.shape[0], xb.shape[0])
                score = float(np.dot(xb[:k], w[:k]))
                score = float(np.clip(score, -60.0, 60.0)) # numerical safety
                preds.append(1.0 / (1.0 + np.exp(-score)))
                used_models.add("linear")
        except Exception as e:
            logger.info(f"Linear skipped: {e}")
    else:
        preds.append(linear_prob)
        used_models.add("linear")

    # --- LSTM probability (optional/lenient) ---
    lstm_prob = _run_lstm_savedmodel(symbol, feature_vector)
    if lstm_prob is not None:
        preds.append(lstm_prob)
        used_models.add("lstm")
    else:
        try:
            lstm_prob = await asyncio.to_thread(_predict_lstm_prob_blocking, symbol, feature_vector)
            preds.append(lstm_prob)
            used_models.add("lstm")
        except Exception as e:
            logger.info(f"LSTM skipped: {e}")

    # --- ARIMA direction (0/1) ---
    try:
        arima_direction = await asyncio.to_thread(
            _predict_arima_direction_blocking,
            symbol,
            px,
            horizon_days,
        )
        if arima_direction is not None:
            preds.append(arima_direction)
            used_models.add("arima")
    except Exception as e:
        logger.info(f"ARIMA skipped: {e}")

    if not preds:
        raise HTTPException(status_code=500, detail="No model produced a prediction")

    # --- RL tiny tilt (optional) ---
    rl_adjust = 0.0
    if StockEnv is not None:
        try:
            rl_adjust = await asyncio.to_thread(
                _predict_rl_adjust_blocking,
                symbol,
                px,
                RL_WINDOW,
            )
            used_models.add("rl")
        except Exception as e:
            logger.info(f"RL skipped: {e}")
    # --- Price bands via conformal calibration ---
    calibration_hint = await _get_calibration_params(symbol, horizon_days)
    q05 = q50 = q95 = None
    if calibration_hint:
        try:
            q05 = float(spot0 * math.exp(float(calibration_hint["q05"])))
            q50 = float(spot0 * math.exp(float(calibration_hint["q50"])))
            q95 = float(spot0 * math.exp(float(calibration_hint["q95"])))
        except Exception as exc:
            logger.debug("Calibration quantile conversion failed for %s: %s", symbol, exc)
            q05 = q50 = q95 = None
    fallback_quantiles = _compute_fallback_quantiles(px, prob_up, horizon_days)
    if (q05 is None or q95 is None) and fallback_quantiles:
        q05, q50, q95 = fallback_quantiles
    elif q05 is None or q95 is None:
        logger.debug("Fallback quantile calibration unavailable for %s", symbol)

    if CALIBRATION_ERROR_GAUGE and fallback_quantiles:
        try:
            horizon_label = str(horizon_days)
            if calibration_hint and q05 is not None and q95 is not None:
                cal_error = max(
                    abs(q05 - float(fallback_quantiles[0])),
                    abs(q95 - float(fallback_quantiles[2])),
                )
                CALIBRATION_ERROR_GAUGE.labels(symbol=symbol, horizon_days=horizon_label).set(cal_error)
            else:
                CALIBRATION_ERROR_GAUGE.labels(symbol=symbol, horizon_days=horizon_label).set(-1.0)
        except Exception:
            pass

    # --- Log to Predictions/Outcomes store (DuckDB via src.db.duck) ---
    pred_id = str(uuid4())
    try:
        insert_prediction(
            {
                "pred_id": pred_id,
                "ts": datetime.utcnow(),
                "symbol": symbol,
                "horizon_d": horizon_days,
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
        features_payload = {
            "mom_20": float(f.get("mom_20", 0.0)),
            "rvol_20": float(f.get("rvol_20", 0.0)),
            "autocorr_5": float(f.get("autocorr_5", 0.0)),
            "spot0": spot0,
        }
        if calibration_hint:
            features_payload["calibration"] = {
                "sigma_ann": float(calibration_hint.get("sigma_ann")) if calibration_hint.get("sigma_ann") is not None else None,
                "sample_n": int(calibration_hint.get("sample_n", 0)),
            }
        fs_log_prediction(
            con_fs,
            {
                # mirror using pred_id as run_id for joinability
                "run_id": pred_id,
                "model_id": "ensemble-v1",
                "symbol": symbol,
                "issued_at": datetime.now(timezone.utc).isoformat(),
                "horizon_days": horizon_days,
                "yhat_mean": None,                # fill with price mid later if available
                "prob_up": float(prob_up),        # note: FS expects 'prob_up'
                "q05": q05,
                "q50": q50,
                "q95": q95,
                "uncertainty": None,
                "features_ref": json.dumps(features_payload),
            },
        )
        con_fs.close()
    except Exception as e:
        logger.warning(f"Feature store mirror failed: {e}")

    if MODEL_USAGE_COUNTER:
        try:
            for model_name in used_models:
                MODEL_USAGE_COUNTER.labels(model=model_name).inc()
        except Exception:
            pass
    if INFERENCE_LATENCY_SECONDS:
        try:
            INFERENCE_LATENCY_SECONDS.labels(endpoint="predict").observe(time.perf_counter() - start_time)
        except Exception:
            pass

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
@app.get("/ui/fan-chart.tsx")
async def _train_models(
    symbol: str,
    *,
    lookback_days: int,
    min_samples: int = 80,
    profile: str | None = None,
) -> dict[str, Any]:
    return await service_train_models(
        symbol,
        lookback_days=lookback_days,
        profile=profile,
    )


async def _ensure_trained_models(
    symbol: str,
    *,
    required_lookback: int,
    profile: str | None = None,
) -> None:
    await service_ensure_trained_models(
        symbol,
        required_lookback=required_lookback,
        profile=profile,
    )


async def get_ensemble_prob(
    symbol: str,
    redis: "Redis",
    horizon_days: int = 1,
    profile: str | None = None,
) -> float:
    return await service_get_ensemble_prob(
        symbol,
        redis,
        horizon_days,
        profile=profile,
    )


async def get_ensemble_prob_light(
    symbol: str,
    redis: "Redis",
    horizon_days: int = 1,
    profile: str | None = None,
) -> float:
    try:
        return await service_get_ensemble_prob_light(
            symbol,
            redis,
            horizon_days,
            profile=profile,
        )
    except HTTPException:
        return 0.5


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




