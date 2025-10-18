from __future__ import annotations

"""
Shared heavy dependency loaders (TensorFlow, ARIMA, Gymnasium, Stable-Baselines3).
"""

import asyncio
import logging
from datetime import datetime, timezone

logger = logging.getLogger("simetrix.services.common")

# Globals tracking optional dependencies
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

_loader_lock = asyncio.Lock()


def load_arima() -> None:
    """Load statsmodels ARIMA lazily."""
    global ARIMA_AVAILABLE, ARIMA
    if ARIMA_AVAILABLE:
        return
    try:
        from statsmodels.tsa.arima.model import ARIMA as _ARIMA

        ARIMA = _ARIMA
        ARIMA_AVAILABLE = True
        logger.info("ARIMA ready")
    except Exception as exc:
        ARIMA_AVAILABLE = False
        logger.warning("ARIMA unavailable: %s", exc)


def load_tensorflow() -> None:
    """Load TensorFlow (heavy import)."""
    global TF_AVAILABLE, load_model, Sequential, LSTM, GRU, Dense
    if TF_AVAILABLE:
        return
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
    except Exception as exc:
        TF_AVAILABLE = False
        logger.warning("TensorFlow unavailable: %s", exc)


def load_gymnasium() -> None:
    """Load Gymnasium."""
    global gym
    if gym is not None:
        return
    try:
        import gymnasium as _gym

        gym = _gym
        logger.info("Gymnasium ready")
    except Exception as exc:
        gym = None
        logger.warning("Gymnasium unavailable: %s", exc)


def load_stable_baselines3() -> None:
    """Load Stable-Baselines3 (for RL)."""
    global SB3_AVAILABLE, DQN
    if SB3_AVAILABLE:
        return
    try:
        from stable_baselines3 import DQN as _DQN

        DQN = _DQN
        SB3_AVAILABLE = True
        logger.info("Stable-Baselines3 ready")
    except Exception as exc:
        SB3_AVAILABLE = False
        logger.warning("Stable-Baselines3 unavailable: %s", exc)


async def ensure_arima() -> None:
    if ARIMA_AVAILABLE:
        return
    async with _loader_lock:
        if ARIMA_AVAILABLE:
            return
        await asyncio.to_thread(load_arima)


async def ensure_tf() -> None:
    if TF_AVAILABLE:
        return
    async with _loader_lock:
        if TF_AVAILABLE:
            return
        await asyncio.to_thread(load_tensorflow)


async def ensure_sb3() -> None:
    if SB3_AVAILABLE:
        return
    async with _loader_lock:
        if SB3_AVAILABLE:
            return
        await asyncio.to_thread(load_stable_baselines3)


__all__ = [
    "ARIMA",
    "ARIMA_AVAILABLE",
    "DQN",
    "Dense",
    "GRU",
    "LSTM",
    "SB3_AVAILABLE",
    "Sequential",
    "TF_AVAILABLE",
    "gym",
    "load_arima",
    "load_tensorflow",
    "load_gymnasium",
    "load_stable_baselines3",
    "ensure_arima",
    "ensure_tf",
    "ensure_sb3",
    "model_key",
    "parse_trained_at",
]


def model_key(symbol: str) -> str:
    """Return the canonical Redis key for a model symbol."""
    return f"model:{(symbol or '').upper().strip()}"


def parse_trained_at(raw: str | None) -> datetime | None:
    """Parse an ISO8601 timestamp into an aware UTC datetime."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw))
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None
