from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from redis.asyncio import Redis

from src.task_queue import TaskDispatcher

__all__ = [
    "settings",
    "REDIS",
    "SIM_DISPATCHER",
    "DATA_ROOT",
    "BACKUP_DIR",
    "EXPORT_ROOT",
    "ARROW_ROOT",
    "ARROW_PREDICTIONS_DIR",
    "ARROW_OUTCOMES_DIR",
    "START_TIME",
    "CALIBRATION_SAMPLE_LIMIT",
    "CALIBRATION_SAMPLE_MIN",
    "CALIBRATION_TTL_SECONDS",
    "CALIBRATION_DB_MAX_AGE",
    "CALIBRATION_GRID_SIZE",
    "IV_CACHE_TTL",
    "_ONNX_SESSION_CACHE",
    "_LSTM_MODEL_CACHE",
    "_LSTM_INFER_CACHE",
    "_ARIMA_MODEL_CACHE",
    "_RL_MODEL_CACHE",
    "_MODEL_META_CACHE",
    "_IV_CACHE",
    "JWT_ALGORITHM",
    "resolve_artifact_path",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("simetrix.core")

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _default_data_root() -> Path:
    env_root = (os.getenv("PT_DATA_ROOT") or "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    container_root = Path("/data")
    if container_root.exists() and os.access(container_root, os.W_OK):
        return container_root
    fallback = (_REPO_ROOT / "data").resolve()
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def resolve_artifact_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = (_REPO_ROOT / p).resolve()
    return p


DATA_ROOT = _default_data_root()
os.environ.setdefault("PT_DATA_ROOT", str(DATA_ROOT))

START_TIME = datetime.now(timezone.utc)
BACKUP_DIR = DATA_ROOT / "backups"
EXPORT_ROOT = DATA_ROOT / "exports"
ARROW_ROOT = DATA_ROOT / "arrow_store"
ARROW_PREDICTIONS_DIR = ARROW_ROOT / "predictions"
ARROW_OUTCOMES_DIR = ARROW_ROOT / "outcomes"
for directory in (BACKUP_DIR, EXPORT_ROOT, ARROW_ROOT, ARROW_PREDICTIONS_DIR, ARROW_OUTCOMES_DIR):
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


JWT_ALGORITHM = "HS256"

def _poly_key() -> str:
    env_k = (os.getenv("PT_POLYGON_KEY") or os.getenv("POLYGON_KEY") or "").strip()
    if env_k:
        return env_k
    try:
        return (settings.polygon_key or "").strip()
    except Exception:
        return ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PT_", case_sensitive=False)

    polygon_key: Optional[str] = Field(default_factory=_poly_key)
    news_api_key: Optional[str] = None
    macro_source: Optional[str] = Field(default=None)
    earnings_source: Optional[str] = Field(default=None)

    redis_url: str = Field(default_factory=lambda: os.getenv("PT_REDIS_URL", "redis://localhost:6379/0"))
    cors_origins_raw: Optional[str] = None

    backup_dir: str = Field(default_factory=lambda: os.getenv("PT_BACKUP_DIR", str(BACKUP_DIR)))
    backup_keep: int = Field(default_factory=lambda: int(os.getenv("PT_BACKUP_KEEP", "7")), ge=1, le=90)

    labeling_interval_seconds: int = Field(default_factory=lambda: int(os.getenv("PT_LABEL_INTERVAL_SECONDS", "900")),
                                           ge=60, le=86400)
    labeling_batch_limit: int = Field(default_factory=lambda: int(os.getenv("PT_LABEL_BATCH_LIMIT", "5000")),
                                      ge=100, le=50000)
    metrics_rollup_interval_seconds: int = Field(default_factory=lambda: int(os.getenv("PT_METRICS_ROLLUP_SECONDS", "3600")),
                                                 ge=300, le=86400)
    model_retrain_poll_seconds: int = Field(default_factory=lambda: int(os.getenv("PT_MODEL_RETRAIN_POLL_SECONDS", "1800")),
                                            ge=300, le=86400)
    model_retrain_daily_hours: int = Field(default_factory=lambda: int(os.getenv("PT_MODEL_RETRAIN_DAILY_HOURS", "24")),
                                           ge=1, le=168)
    model_retrain_weekly_hours: int = Field(default_factory=lambda: int(os.getenv("PT_MODEL_RETRAIN_WEEKLY_HOURS", str(24 * 7))),
                                            ge=24, le=24 * 30)
    data_retention_days: int = Field(default_factory=lambda: int(os.getenv("PT_DATA_RETENTION_DAYS", "180")),
                                     ge=30, le=3650)
    data_retention_poll_seconds: int = Field(default_factory=lambda: int(os.getenv("PT_DATA_RETENTION_POLL_SECONDS", str(24 * 3600))),
                                             ge=3600, le=7 * 24 * 3600)

    retrain_daily_symbols: str = Field(default="", validation_alias="PT_RETRAIN_DAILY_SYMBOLS")
    retrain_weekly_symbols: str = Field(default="", validation_alias="PT_RETRAIN_WEEKLY_SYMBOLS")

    backup_interval_minutes: int = Field(default_factory=lambda: int(os.getenv("PT_BACKUP_INTERVAL_MINUTES", "1440")),
                                         ge=60)
    daily_quant_horizon_days: int = Field(default_factory=lambda: int(os.getenv("PT_DAILY_QUANT_HORIZON", "14")),
                                          ge=5, le=90)
    daily_quant_hhmm: str = Field(default_factory=lambda: os.getenv("PT_DAILY_QUANT_HHMM", "08:45"))
    daily_quant_timezone: str = Field(default_factory=lambda: os.getenv("PT_DAILY_QUANT_TZ", "America/New_York"))
    sim_queue_concurrency: int = Field(default_factory=lambda: int(os.getenv("PT_SIM_QUEUE_CONCURRENCY", "0")),
                                       ge=0, le=16)

    cookie_name: str = Field("pt_app", validation_alias="PT_COOKIE_NAME")
    cookie_max_age: int = Field(60 * 60 * 24, validation_alias="PT_COOKIE_MAX_AGE")
    cookie_secure: bool = Field(default_factory=lambda: bool(os.getenv("RENDER")) or os.getenv("ENV", "dev") == "prod")
    admin_username: Optional[str] = Field(default_factory=lambda: os.getenv("PT_ADMIN_USERNAME"))
    admin_password_hash: Optional[str] = Field(default_factory=lambda: os.getenv("PT_ADMIN_PASSWORD_HASH"))
    admin_session_secret: str = Field(default_factory=lambda: os.getenv("PT_ADMIN_SESSION_SECRET") or os.getenv("PT_JWT_SECRET", "dev-secret"))
    admin_session_cookie: str = Field("pt_admin_session", validation_alias="PT_ADMIN_SESSION_COOKIE")
    admin_session_ttl_minutes: int = Field(default_factory=lambda: int(os.getenv("PT_ADMIN_SESSION_TTL_MINUTES", "30")),
                                           ge=5, le=24 * 60)
    admin_login_rpm: int = Field(default_factory=lambda: int(os.getenv("PT_ADMIN_LOGIN_RPM", "6")), ge=1, le=120)
    admin_cookie_secure: bool = Field(default_factory=lambda: _env_flag(
        "PT_ADMIN_COOKIE_SECURE",
        bool(os.getenv("RENDER")) or os.getenv("ENV", "dev") == "prod",
    ))

    n_paths_max: int = Field(default_factory=lambda: int(os.getenv("PT_N_PATHS_MAX", "10000")))
    horizon_days_max: int = Field(default_factory=lambda: int(os.getenv("PT_HORIZON_DAYS_MAX", "3650")))
    lookback_days_max: int = Field(default_factory=lambda: int(os.getenv("PT_LOOKBACK_DAYS_MAX", str(365 * 10))))
    pathday_budget_max: int = Field(default_factory=lambda: int(os.getenv("PT_PATHDAY_BUDGET_MAX", "500000")))
    max_active_runs: int = Field(default_factory=lambda: int(os.getenv("PT_MAX_ACTIVE_RUNS", "2")))
    run_ttl_seconds: int = Field(default_factory=lambda: int(os.getenv("PT_RUN_TTL_SECONDS", "3600")))
    validation_target_mape: float = Field(default=5.0,
                                          validation_alias="PT_VALIDATION_TARGET_MAPE",
                                          ge=0.1, le=50.0)
    validation_lookbacks: str = Field(default="20,60,120", validation_alias="PT_VALIDATION_LOOKBACKS")
    validation_max_samples: int = Field(default=120,
                                        validation_alias="PT_VALIDATION_MAX_SAMPLES",
                                        ge=20, le=500)

    predictive_defaults: dict[str, Any] = Field(
        default_factory=lambda: {
            "X:BTCUSD": {"horizon_days": 365, "n_paths": 10000, "lookback_preset": "3y"},
            "NVDA": {"horizon_days": 30, "n_paths": 5000, "lookback_preset": "180d"},
        }
    )

    watchlist_equities: str = Field("", validation_alias="PT_WATCHLIST_EQUITIES")
    watchlist_cryptos: str = Field("", validation_alias="PT_WATCHLIST_CRYPTOS")
    equity_watch: list[str] = Field(default_factory=list, validation_alias="PT_EQUITY_WATCH")
    crypto_watch: list[str] = Field(default_factory=list, validation_alias="PT_CRYPTO_WATCH")

    shortlist_disable: bool = Field(False, validation_alias="PT_SHORTLIST_DISABLE")
    open_access: bool = Field(default_factory=lambda: os.getenv("PT_OPEN_ACCESS", "1") == "1")
    pt_api_key: Optional[str] = None

    jwt_secret: str = Field(default_factory=lambda: os.getenv("PT_JWT_SECRET", "dev-secret"))
    jwt_exp_minutes: int = Field(default_factory=lambda: int(os.getenv("PT_JWT_EXP_MINUTES", "60")))

    @model_validator(mode="after")
    def _validate_required(self) -> "Settings":
        if not (self.polygon_key or "").strip():
            raise ValueError("Polygon API key is required; set PT_POLYGON_KEY or POLYGON_KEY.")
        if not self.open_access and not (self.pt_api_key or "").strip():
            raise ValueError("PT_API_KEY must be set when PT_OPEN_ACCESS is disabled.")
        return self


settings = Settings()
BACKUP_DIR = Path(settings.backup_dir).expanduser()

SIM_DISPATCHER: TaskDispatcher | None = None
if settings.sim_queue_concurrency > 0:
    SIM_DISPATCHER = TaskDispatcher("simulate", concurrency=settings.sim_queue_concurrency)

try:
    REDIS: Redis | None = Redis.from_url(settings.redis_url, decode_responses=True)
except Exception as exc:
    logger.warning("redis_init_failed: %s", exc)
    REDIS = None


CALIBRATION_SAMPLE_LIMIT = 500
CALIBRATION_SAMPLE_MIN = 40
CALIBRATION_TTL_SECONDS = int(os.getenv("PT_CALIBRATION_TTL", "21600"))
CALIBRATION_DB_MAX_AGE = timedelta(hours=int(os.getenv("PT_CALIBRATION_MAX_AGE_HOURS", "24")))
CALIBRATION_GRID_SIZE = 2048
IV_CACHE_TTL = timedelta(hours=6)

_ONNX_SESSION_CACHE: dict[str, Any] = {}
_LSTM_MODEL_CACHE: dict[str, Any] = {}
_LSTM_INFER_CACHE: dict[str, Callable[..., float]] = {}
_ARIMA_MODEL_CACHE: dict[str, Any] = {}
_RL_MODEL_CACHE: dict[str, Any] = {}
_MODEL_META_CACHE: dict[str, dict[str, Any]] = {}
_IV_CACHE: dict[str, tuple[float, datetime]] = {}

ONNX_SESSION_CACHE = _ONNX_SESSION_CACHE
LSTM_MODEL_CACHE = _LSTM_MODEL_CACHE
LSTM_INFER_CACHE = _LSTM_INFER_CACHE
ARIMA_MODEL_CACHE = _ARIMA_MODEL_CACHE
RL_MODEL_CACHE = _RL_MODEL_CACHE
MODEL_META_CACHE = _MODEL_META_CACHE
IV_CACHE = _IV_CACHE
