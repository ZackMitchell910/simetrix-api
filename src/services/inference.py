from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone, date
from typing import Any, Callable, List, Mapping, Sequence, Set

import numpy as np
from scipy import stats
from fastapi import HTTPException

from src.core import (
    CALIBRATION_DB_MAX_AGE,
    CALIBRATION_GRID_SIZE,
    CALIBRATION_SAMPLE_LIMIT,
    CALIBRATION_SAMPLE_MIN,
    CALIBRATION_TTL_SECONDS,
    resolve_artifact_path,
    _ARIMA_MODEL_CACHE,
    _LSTM_INFER_CACHE,
    _LSTM_MODEL_CACHE,
    _ONNX_SESSION_CACHE,
    _RL_MODEL_CACHE,
)
from src.model_registry import get_active_model_version
from src.feature_store import (
    connect as fs_connect,
    get_calibration_params as fs_get_calibration_params,
    upsert_calibration_params,
)
from src.services.common import (
    ARIMA,
    ARIMA_AVAILABLE,
    DQN,
    Dense,
    GRU,
    LSTM,
    SB3_AVAILABLE,
    Sequential,
    TF_AVAILABLE,
    gym,
    load_arima,
    load_gymnasium,
    load_stable_baselines3,
    load_tensorflow,
)
from src.services.training import (
    RL_WINDOW,
    StockEnv,
    _feat_from_prices,
    _onnx_model_path,
    _onnx_supported,
    DEFAULT_TRAIN_PROFILE,
    linear_model_key_for_profile,
    resolve_training_profile,
)
from src.services import ingestion as ingestion_service

logger = logging.getLogger("simetrix.services.inference")


@dataclass
class PreparedFeatures:
    symbol: str
    model_linear: dict[str, Any]
    feature_list: List[str]
    feature_vector: List[float]
    feature_map: dict[str, float]
    prices: List[float]
    spot: float
    profile: str


@dataclass
class ModelEvaluation:
    prob_up: float
    used_models: Set[str]
    preds: List[float]
    rl_adjust: float


def _ensure_tf_sync() -> None:
    if not TF_AVAILABLE:
        load_tensorflow()
    if not TF_AVAILABLE:
        raise HTTPException(status_code=503, detail="TensorFlow unavailable")


def _ensure_arima_sync() -> None:
    if not ARIMA_AVAILABLE or ARIMA is None:
        load_arima()
    if not ARIMA_AVAILABLE or ARIMA is None:
        raise HTTPException(status_code=503, detail="arima_unavailable")


def _ensure_rl_sync() -> None:
    if gym is None:
        load_gymnasium()
    if DQN is None or not SB3_AVAILABLE:
        load_stable_baselines3()
    if DQN is None:
        raise HTTPException(status_code=503, detail="rl_unavailable")


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


def sigma_from_calibration(cal: Mapping[str, Any]) -> float | None:
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


def _calibration_from_store(symbol: str, horizon_days: int) -> dict[str, Any] | None:
    if fs_connect is None or fs_get_calibration_params is None:
        return None
    try:
        con = fs_connect()
        row = fs_get_calibration_params(con, symbol, horizon_days)
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


async def get_calibration_params(symbol: str, horizon_days: int, redis) -> dict[str, Any] | None:
    symbol_norm = symbol.upper()
    key = _calibration_key(symbol_norm, horizon_days)
    if redis:
        try:
            cached = await redis.get(key)
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
    if cal and redis:
        try:
            await redis.set(key, json.dumps(cal), ex=int(CALIBRATION_TTL_SECONDS))
        except Exception:
            pass
    return cal


def compute_fallback_quantiles(
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
        return tuple(float(spot0 * math.exp(rv.ppf(p))) for p in (0.05, 0.5, 0.95))
    except Exception as exc:
        logger.debug("fallback quantile computation failed: %s", exc)
        return None


def _candidate_linear_paths(symbol: str) -> list[Path]:
    sym = symbol.upper()
    candidates: list[Path] = []
    try:
        entry = get_active_model_version("linear", sym)
    except Exception as exc:
        logger.debug("Model registry lookup failed for linear %s: %s", sym, exc)
        entry = None
    if entry and entry.get("artifact_path"):
        try:
            candidates.append(resolve_artifact_path(entry["artifact_path"]))
        except Exception as exc:
            logger.debug("Unable to resolve linear registry artifact for %s: %s", sym, exc)
    candidates.append(_onnx_model_path(sym))
    return candidates


def _load_linear_onnx(symbol: str):
    if not _onnx_supported():
        return None
    sym = symbol.upper()
    cached = _ONNX_SESSION_CACHE.get(sym)
    if cached is not None:
        return cached
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.debug("onnxruntime unavailable for %s: %s", sym, exc)
        return None

    for path in _candidate_linear_paths(sym):
        if not path.exists():
            continue
        try:
            session = ort.InferenceSession(path.as_posix(), providers=["CPUExecutionProvider"])
            _ONNX_SESSION_CACHE[sym] = session
            return session
        except Exception as exc:
            logger.debug("Failed to load ONNX model for %s from %s: %s", sym, path, exc)
            continue
    return None


def _run_linear_onnx(symbol: str, feature_vector: Sequence[float]) -> float | None:
    session = _load_linear_onnx(symbol)
    if session is None:
        return None
    try:
        X = np.asarray([feature_vector], dtype=np.float32)
        outputs = session.run(None, {"features": X})
        prob = float(outputs[0][0][0])
        return float(np.clip(prob, 0.0, 1.0))
    except Exception as exc:
        logger.debug("ONNX inference failed for %s: %s", symbol, exc)
        return None


def _load_lstm_model(symbol: str):
    sym = (symbol or "").upper().strip()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol_required")
    cached = _LSTM_MODEL_CACHE.get(sym)
    if cached is not None:
        return cached

    _ensure_tf_sync()

    candidates: list[Path] = []
    try:
        entry = get_active_model_version("lstm", sym)
    except Exception as exc:
        logger.debug("Model registry lookup failed for lstm %s: %s", sym, exc)
        entry = None
    if entry and entry.get("artifact_path"):
        try:
            candidates.append(resolve_artifact_path(entry["artifact_path"]))
        except Exception as exc:
            logger.debug("Unable to resolve LSTM registry artifact for %s: %s", sym, exc)

    candidates.extend(
        [
            Path("models") / f"{sym}_lstm_savedmodel",
            Path("models") / f"{sym}_lstm.keras",
            Path("models") / f"{sym}_lstm.h5",
        ]
    )

    for path in candidates:
        try:
            if not path.exists():
                continue
            if path.is_dir():
                from tensorflow.keras.models import load_model as tf_load_model  # type: ignore

                model = tf_load_model(path.as_posix())
            else:
                if "_tf_load_model" in globals() and globals().get("_tf_load_model"):
                    model = globals()["_tf_load_model"](path.as_posix())  # type: ignore
                else:
                    from tensorflow.keras.models import load_model as tf_load_model  # type: ignore

                    model = tf_load_model(path.as_posix())
            _LSTM_MODEL_CACHE[sym] = model
            return model
        except Exception as exc:
            logger.debug("LSTM load candidate %s failed for %s: %s", path, sym, exc)
            continue
    raise HTTPException(status_code=404, detail="LSTM model not found; train first")


def _load_lstm_predictor(symbol: str) -> Callable[[np.ndarray], float] | None:
    sym = symbol.upper()
    cached = _LSTM_INFER_CACHE.get(sym)
    if cached is not None:
        return cached

    try:
        model = _load_lstm_model(symbol)
    except HTTPException:
        return None
    except Exception as exc:
        logger.debug("LSTM predictor load failed for %s: %s", sym, exc)
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
    try:
        prob = float(infer_fn(inputs))
        return float(np.clip(prob, 0.0, 1.0))
    except Exception as exc:
        logger.debug("LSTM inference failed for %s: %s", symbol, exc)
        _LSTM_INFER_CACHE.pop(symbol.upper(), None)
        return None


def _predict_lstm_prob_blocking(symbol: str, feature_vector: Sequence[float]) -> float:
    prob = _run_lstm_savedmodel(symbol, feature_vector)
    if prob is None:
        raise RuntimeError("lstm_unavailable")
    return prob


def load_arima_model(symbol: str):
    sym = (symbol or "").upper().strip()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol_required")

    cached = _ARIMA_MODEL_CACHE.get(sym)
    if cached is not None:
        return cached

    _ensure_arima_sync()

    candidates: list[Path] = []
    try:
        entry = get_active_model_version("arima", sym)
    except Exception as exc:
        logger.debug("Model registry lookup failed for arima %s: %s", sym, exc)
        entry = None
    if entry and entry.get("artifact_path"):
        try:
            candidates.append(resolve_artifact_path(entry["artifact_path"]))
        except Exception as exc:
            logger.debug("Unable to resolve ARIMA registry artifact for %s: %s", sym, exc)
    candidates.append(Path("models") / f"{sym}_arima.pkl")

    for path in candidates:
        try:
            if not path.exists():
                continue
            with open(path, "rb") as fh:
                model = pickle.load(fh)
                _ARIMA_MODEL_CACHE[sym] = model
                return model
        except Exception as exc:
            logger.debug("ARIMA load candidate %s failed for %s: %s", path, sym, exc)
            continue
    raise HTTPException(status_code=404, detail="ARIMA model not found; train first")


def _predict_arima_direction_blocking(symbol: str, prices: Sequence[float], horizon_days: int) -> float | None:
    try:
        model = load_arima_model(symbol)
    except HTTPException:
        return None
    fc = model.forecast(steps=max(1, horizon_days))
    if hasattr(fc, "iloc"):
        last_fc = float(fc.iloc[-1])
    else:
        last_fc = float(fc[-1])
    last_price = float(prices[-1])
    return 1.0 if last_fc > last_price else 0.0


def load_rl_model(symbol: str):
    sym = (symbol or "").upper().strip()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol_required")

    cached = _RL_MODEL_CACHE.get(sym)
    if cached is not None:
        return cached

    _ensure_rl_sync()

    path = Path("models") / f"{sym}_rl.zip"
    if not path.exists():
        raise HTTPException(status_code=404, detail="RL model not found; train first")
    try:
        model = DQN.load(path.as_posix(), print_system_info=False)  # type: ignore[attr-defined]
        _RL_MODEL_CACHE[sym] = model
        return model
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"failed to load rl model: {exc}") from exc


def _predict_rl_adjust_blocking(symbol: str, prices: Sequence[float], window_len: int) -> float:
    model = load_rl_model(symbol)
    if StockEnv is None:
        raise RuntimeError("rl_env_unavailable")
    env = StockEnv(prices, window_len=window_len)
    try:
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)  # type: ignore[attr-defined]
        a_map = {-1: -0.01, 0: 0.0, 1: 0.01}
        a_idx = int(action)
        a_val = a_map.get(a_idx - 1, 0.0)
        return float(np.clip(a_val, -0.05, 0.05))
    finally:
        env.close()


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(np.clip(z, -60.0, 60.0))))


async def prepare_features(
    symbol: str,
    redis,
    horizon_days: int,
    *,
    profile: str | None = None,
) -> PreparedFeatures:
    sym = (symbol or "").upper().strip()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol_required")

    if redis is None:
        raise HTTPException(status_code=503, detail="redis_unavailable")

    try:
        resolved_profile = resolve_training_profile(profile)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    raw = await redis.get(linear_model_key_for_profile(sym, resolved_profile))
    if not raw:
        detail = (
            f"linear model ({resolved_profile}) not found; train first"
            if resolved_profile != DEFAULT_TRAIN_PROFILE
            else "Linear model not found; train first"
        )
        raise HTTPException(status_code=404, detail=detail)
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    try:
        model_linear = json.loads(raw)
    except Exception as exc:
        logger.debug("Failed to parse linear model payload for %s: %s", sym, exc)
        raise HTTPException(status_code=500, detail="Corrupt linear model payload") from exc

    prices = await ingestion_service.fetch_hist_prices(sym, window_days=horizon_days or None)
    if not prices or len(prices) < 10:
        raise HTTPException(status_code=400, detail="Not enough price history")

    feature_map = await _feat_from_prices(sym, prices)
    feature_list = list(model_linear.get("features") or ["mom_20", "rvol_20", "autocorr_5"])
    feature_vector = [float(feature_map.get(name, 0.0)) for name in feature_list]
    spot = float(prices[-1])
    return PreparedFeatures(
        symbol=sym,
        model_linear=model_linear,
        feature_list=feature_list,
        feature_vector=feature_vector,
        feature_map=feature_map,
        prices=list(prices),
        spot=spot,
        profile=resolved_profile,
    )


def _linear_probability(prep: PreparedFeatures) -> float | None:
    prob = _run_linear_onnx(prep.symbol, prep.feature_vector)
    if prob is not None:
        return prob

    coef = np.asarray(prep.model_linear.get("coef", []), dtype=float)
    if coef.size == 0:
        return None
    xb = np.concatenate([[1.0], np.asarray(prep.feature_vector, dtype=float)])
    k = min(coef.shape[0], xb.shape[0])
    if k <= 0:
        return None
    score = float(np.dot(coef[:k], xb[:k]))
    return _sigmoid(score)


async def evaluate_models(prep: PreparedFeatures, horizon_days: int) -> ModelEvaluation:
    preds: List[float] = []
    used: Set[str] = set()

    linear_prob = _linear_probability(prep)
    if linear_prob is not None:
        preds.append(linear_prob)
        used.add("linear")

    lstm_prob = _run_lstm_savedmodel(prep.symbol, prep.feature_vector)
    if lstm_prob is not None:
        preds.append(lstm_prob)
        used.add("lstm")
    else:
        try:
            prob = await asyncio.to_thread(_predict_lstm_prob_blocking, prep.symbol, prep.feature_vector)
            preds.append(prob)
            used.add("lstm")
        except Exception as exc:
            logger.info("LSTM skipped: %s", exc)

    try:
        arima_prob = await asyncio.to_thread(
            _predict_arima_direction_blocking,
            prep.symbol,
            prep.prices,
            horizon_days,
        )
        if arima_prob is not None:
            preds.append(arima_prob)
            used.add("arima")
    except Exception as exc:
        logger.info("ARIMA skipped: %s", exc)

    rl_adjust = 0.0
    if StockEnv is not None:
        try:
            rl_adjust = await asyncio.to_thread(
                _predict_rl_adjust_blocking,
                prep.symbol,
                prep.prices,
                RL_WINDOW,
            )
            used.add("rl")
        except Exception as exc:
            logger.info("RL skipped: %s", exc)

    if not preds:
        raise HTTPException(status_code=500, detail="No model produced a prediction")

    mean_prob = float(np.clip(np.mean(preds), 0.0, 1.0))
    prob_up = float(np.clip(mean_prob + rl_adjust, 0.0, 1.0))
    return ModelEvaluation(prob_up=prob_up, used_models=used, preds=preds, rl_adjust=rl_adjust)


async def get_ensemble_prob(
    symbol: str,
    redis,
    horizon_days: int = 1,
    *,
    profile: str | None = None,
) -> float:
    try:
        prep = await prepare_features(symbol, redis, horizon_days, profile=profile)
        evaluation = await evaluate_models(prep, horizon_days)
        return evaluation.prob_up
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("get_ensemble_prob failed for %s: %s", symbol, exc)
        return 0.5


async def get_ensemble_prob_light(
    symbol: str,
    redis,
    horizon_days: int = 1,
    *,
    profile: str | None = None,
) -> float:
    try:
        prep = await prepare_features(symbol, redis, horizon_days, profile=profile)
        prob = _linear_probability(prep)
        if prob is not None:
            return prob
        return 0.5
    except HTTPException:
        raise
    except Exception as exc:
        logger.info("get_ensemble_prob_light fallback (0.5) due to: %s", exc)
        return 0.5


def winsorize(arr: np.ndarray, p_lo: float = 0.005, p_hi: float = 0.995) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo, hi = np.quantile(arr, [p_lo, p_hi])
    return np.clip(arr, lo, hi)


__all__ = [
    "PreparedFeatures",
    "ModelEvaluation",
    "prepare_features",
    "evaluate_models",
    "get_ensemble_prob",
    "get_ensemble_prob_light",
    "winsorize",
    "_run_linear_onnx",
    "_run_lstm_savedmodel",
    "_predict_lstm_prob_blocking",
    "_predict_arima_direction_blocking",
    "_predict_rl_adjust_blocking",
    "load_arima_model",
    "load_rl_model",
    "get_calibration_params",
    "compute_fallback_quantiles",
    "sigma_from_calibration",
]
