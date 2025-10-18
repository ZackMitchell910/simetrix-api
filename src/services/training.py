from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from uuid import uuid4

import numpy as np
from fastapi import HTTPException

from src.core import (
    settings,
    _ARIMA_MODEL_CACHE,
    _LSTM_INFER_CACHE,
    _LSTM_MODEL_CACHE,
    _MODEL_META_CACHE,
    _ONNX_SESSION_CACHE,
    _RL_MODEL_CACHE,
)
from src.model_registry import register_model_version
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
    ensure_arima,
    ensure_sb3,
    ensure_tf,
    gym,
    load_arima,
    load_stable_baselines3,
    load_tensorflow,
    model_key,
    parse_trained_at,
)
from src.services import ingestion as ingestion_service

try:
    from src.feature_store import connect as fs_connect
except Exception:  # pragma: no cover - fallback when running as script
    from feature_store import connect as fs_connect  # type: ignore

logger = logging.getLogger("simetrix.services.training")

try:  # Optional deps
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


def _poly_key() -> str:
    env_k = (os.getenv("PT_POLYGON_KEY") or os.getenv("POLYGON_KEY") or "").strip()
    if env_k:
        return env_k
    try:
        return (settings.polygon_key or "").strip()
    except Exception:
        return ""


def _get_redis():
    """Fetch the Redis client from predictive_service for compatibility."""
    try:
        from src import predictive_service as svc  # type: ignore

        return getattr(svc, "REDIS", None)
    except Exception:
        return None


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
        x = rets[-w - 1 : -1]
        y = rets[-w:]
        sx, sy = np.std(x), np.std(y)
        if sx > 0 and sy > 0:
            out["autocorr_5"] = float(np.corrcoef(x, y)[0, 1])
        else:
            out["autocorr_5"] = 0.0

    return out


async def _feat_from_prices(symbol: str, px: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in px if isinstance(v, (int, float)) and math.isfinite(v)], dtype=float)
    return _basic_features_from_array(arr)


def _load_labeled_samples(symbol: str, limit: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    con = fs_connect()
    rows = con.execute(
        """
        SELECT
            p.model_id, p.symbol, p.issued_at, p.horizon_days,
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

    X: List[List[float]] = []
    y: List[float] = []
    for _, _, _, _, features_ref, q50, yhat_mean, _, realized_price in rows:
        try:
            feats = {}
            if features_ref:
                js = json.loads(features_ref)
                feats = js.get("features", js) if isinstance(js, dict) else {}

            mom = float(feats.get("mom_20", 0.0))
            rvol = float(feats.get("rvol_20", 0.0))
            ac5 = float(feats.get("autocorr_5", 0.0))

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


class SGDOnline:
    """Simple logistic-regression learner with SGD + L2."""

    def __init__(self, lr: float = 0.05, l2: float = 1e-4):
        self.lr = float(lr)
        self.l2 = float(l2)
        self.w: np.ndarray | None = None

    def init(self, d: int) -> None:
        self.w = np.zeros(int(d) + 1, dtype=float)

    @staticmethod
    def _sigmoid(z: float) -> float:
        z = float(np.clip(z, -60.0, 60.0))
        return 1.0 / (1.0 + math.exp(-z))

    def proba(self, x: Iterable[float]) -> float:
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

    def update(self, x: Iterable[float], y: float) -> None:
        if self.w is None:
            raise RuntimeError("Call init(d) before update()")
        xb = np.concatenate([[1.0], np.asarray(x, dtype=float)])
        y = float(y)
        z = float(np.dot(self.w, xb))
        p = self._sigmoid(z)
        grad = (p - y) * xb
        grad[1:] += self.l2 * self.w[1:]
        self.w -= self.lr * grad


class EW:
    """Exponential weights combiner for ensembling probabilities."""

    def __init__(self, eta: float = 2.0):
        self.eta = float(eta)
        self.w: np.ndarray | None = None

    def init(self, n: int) -> None:
        n = int(n)
        if n <= 0:
            raise ValueError("n must be > 0")
        self.w = np.ones(n, dtype=float) / float(n)

    def update(self, losses: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("Call init(n) before update()")
        L = np.asarray(losses, dtype=float)
        w_new = self.w * np.exp(-self.eta * L)
        s = float(w_new.sum())
        self.w = (w_new / s) if s > 0 else (np.ones_like(w_new) / w_new.size)
        return self.w


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
        onnx_input = onnx_helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, n_features])
        onnx_output = onnx_helper.make_tensor_value_info("prob", TensorProto.FLOAT, [1, 1])

        coef = np.asarray(weights, dtype=np.float32)
        if coef.size != n_features + 1:
            pad_width = (n_features + 1) - coef.size
            coef = np.pad(coef, (0, max(0, pad_width)))
        bias = float(coef[0])
        linear_weights = coef[1 : n_features + 1].reshape(1, n_features)

        initializers = [
            onnx_helper.make_tensor("linear_weight", TensorProto.FLOAT, [1, n_features], linear_weights.flatten()),
            onnx_helper.make_tensor("linear_bias", TensorProto.FLOAT, [1], [bias]),
        ]

        nodes = [
            onnx_helper.make_node("Gemm", inputs=["input", "linear_weight", "linear_bias"], outputs=["logits"]),
            onnx_helper.make_node("Sigmoid", inputs=["logits"], outputs=["prob"]),
        ]

        graph = onnx_helper.make_graph(nodes, f"{sym_up}_linear", [onnx_input], [onnx_output], initializers)
        model = onnx_helper.make_model(graph, producer_name="simetrix")
        path = _onnx_model_path(sym_up)
        path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save_model(model, path.as_posix())  # type: ignore[attr-defined]
        _ONNX_SESSION_CACHE.pop(sym_up, None)
        return path
    except Exception as exc:
        logger.debug("ONNX export failed for %s: %s", symbol, exc)
        return None


def _lstm_saved_model_dir(symbol: str) -> Path:
    return Path("models") / f"{symbol.upper()}_lstm_savedmodel"


def _export_lstm_savedmodel(sym: str, model: Any) -> Path | None:
    path = _lstm_saved_model_dir(sym)
    try:
        if path.exists():
            for child in path.iterdir():
                if child.is_dir():
                    for sub in child.rglob("*"):
                        sub.unlink(missing_ok=True)  # type: ignore[attr-defined]
                else:
                    child.unlink(missing_ok=True)  # type: ignore[attr-defined]
            path.rmdir()
    except Exception:
        pass
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


def _train_lstm_blocking(
    sym: str,
    feature_list: Sequence[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    ensure_tf()
    if Sequential is None or LSTM is None or GRU is None or Dense is None:
        raise RuntimeError("TensorFlow not available")

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


def _train_arima_blocking(sym: str, arr: np.ndarray, train_idx: int, val_size: int) -> dict[str, Any]:
    ensure_arima()
    if not ARIMA_AVAILABLE or ARIMA is None:
        raise RuntimeError("ARIMA unavailable")
    history = arr[:train_idx]
    model = ARIMA(history, order=(2, 1, 2)).fit()
    preds = model.forecast(steps=val_size)
    actual = arr[train_idx : train_idx + val_size]
    preds = np.asarray(preds, dtype=float)
    actual = np.asarray(actual, dtype=float)
    mape = float(np.mean(np.abs((actual - preds) / np.clip(actual, 1e-9, None))) * 100.0)
    mae = float(np.mean(np.abs(actual - preds)))
    with open(f"models/{sym}_arima.pkl", "wb") as file:
        import pickle

        pickle.dump(model, file)
    return {"mape": mape, "mae": mae, "n_obs": int(len(arr))}


def _train_rl_blocking(sym: str, prices: Sequence[float]) -> dict[str, Any]:
    if StockEnv is None:
        raise RuntimeError("Gymnasium unavailable")
    if DQN is None:
        load_stable_baselines3()
    if DQN is None:
        raise RuntimeError("Stable-Baselines3 unavailable")
    env = StockEnv(prices, window_len=RL_WINDOW)  # type: ignore[arg-type]
    try:
        model_rl = DQN("MlpPolicy", env, verbose=0)
        model_rl.learn(total_timesteps=10_000)
        model_rl.save(f"models/{sym}_rl.zip")
        return {"timesteps": 10_000}
    finally:
        env.close()


TRAIN_REFRESH_HOURS = float(os.getenv("PT_TRAIN_REFRESH_HOURS", "12") or 0)
TRAIN_REFRESH_DELTA = timedelta(hours=TRAIN_REFRESH_HOURS) if TRAIN_REFRESH_HOURS > 0 else None
QUICK_TRAIN_LOOKBACK_DAYS = max(
    30, min(settings.lookback_days_max, int(os.getenv("PT_TRAIN_QUICK_LOOKBACK_DAYS", "180")))
)
DEEP_TRAIN_LOOKBACK_DAYS = max(
    30, min(settings.lookback_days_max, int(os.getenv("PT_TRAIN_DEEP_LOOKBACK_DAYS", "3650")))
)

RL_WINDOW = int(os.getenv("PT_RL_WINDOW", "100"))


async def _train_models(symbol: str, *, lookback_days: int) -> dict[str, Any]:
    redis = _get_redis()
    if not redis:
        raise HTTPException(status_code=503, detail="redis_unavailable")

    sym = (symbol or "").upper().strip()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol_required")

    lookback_days = int(max(30, min(lookback_days, settings.lookback_days_max)))
    px = await ingestion_service.fetch_hist_prices(sym, window_days=lookback_days)
    if not px or len(px) < 10:
        raise HTTPException(status_code=400, detail="Not enough price history")

    os.makedirs("models", exist_ok=True)

    trained_at = datetime.now(timezone.utc).isoformat()
    arr = np.asarray(px, dtype=float)
    try:
        feature_list, X_all, y_all, dataset_hash = _build_training_dataset(arr, horizon=1)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    n_samples = X_all.shape[0]
    if n_samples < 80:
        raise HTTPException(status_code=400, detail="Insufficient samples for model training")

    val_size = max(30, int(n_samples * 0.2))
    if val_size >= n_samples - 20:
        val_size = max(10, min(50, n_samples // 5))
    train_idx = n_samples - val_size
    if train_idx <= 20 or val_size <= 5:
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

    weights = model_linear.w.tolist() if model_linear.w is not None else []
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
    await redis.set(model_key(sym + "_linear"), json.dumps(linear_data))

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
        except Exception as exc:
            logger.warning("LSTM skipped: %s", exc)
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
    except Exception as exc:
        logger.warning("ARIMA skipped: %s", exc)
        arima_meta = None
    finally:
        _ARIMA_MODEL_CACHE.pop(sym, None)

    rl_meta: dict[str, Any] | None = None
    try:
        if gym is None or DQN is None:
            raise RuntimeError("RL libs not available")
        rl_meta = await asyncio.to_thread(_train_rl_blocking, sym, px)
        models_trained += 1
    except Exception as exc:
        logger.warning("RL skipped: %s", exc)
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
        await redis.set(f"model_meta:{sym}", json.dumps(meta_payload))
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
    redis = _get_redis()
    if not redis:
        raise HTTPException(status_code=503, detail="redis_unavailable")

    sym = (symbol or "").upper().strip()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol_required")

    required_lookback = int(max(30, min(required_lookback, settings.lookback_days_max)))

    redis_key = model_key(sym + "_linear")

    raw = await redis.get(redis_key)
    if raw:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except Exception:
            data = {}

        stored_lookback = int(data.get("lookback_days") or 0)
        trained_at = parse_trained_at(data.get("trained_at"))

        is_fresh = True
        if TRAIN_REFRESH_DELTA:
            if trained_at is None:
                is_fresh = False
            else:
                age = datetime.now(timezone.utc) - trained_at
                is_fresh = age <= TRAIN_REFRESH_DELTA

        if stored_lookback >= required_lookback and is_fresh:
            return

    try:
        from src import predictive_service as svc  # type: ignore

        train_fn = getattr(svc, "_train_models", _train_models)
    except Exception:
        train_fn = _train_models

    await train_fn(sym, lookback_days=required_lookback)


if gym is not None:
    class StockEnv(gym.Env):  # type: ignore[attr-defined]
        """Simple price-following environment for RL."""

        metadata = {"render_modes": []}

        def __init__(self, prices: Sequence[float], window_len: int = 100):
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

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            self.current_step = 0
            return self._get_obs(), {}

        def _get_obs(self) -> np.ndarray:
            end = self.current_step + 1
            start = max(0, end - self.window_len)
            window = self.prices[start:end]
            if window.size < self.window_len:
                pad = np.full(self.window_len - window.size, window[0], dtype=np.float32)
                window = np.concatenate([pad, window])
            base = window[0] if window[0] != 0 else 1.0
            return (window / base).astype(np.float32)

        def step(self, action: int):
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
            info: dict[str, Any] = {}
            return obs, reward, terminated, truncated, info
else:
    StockEnv = None  # type: ignore


__all__ = [
    "_basic_features_from_array",
    "_build_training_dataset",
    "_ensure_trained_models",
    "_export_linear_onnx",
    "_feat_from_prices",
    "_load_labeled_samples",
    "_onnx_model_path",
    "_onnx_supported",
    "_train_arima_blocking",
    "_train_lstm_blocking",
    "_train_models",
    "_train_rl_blocking",
    "DEEP_TRAIN_LOOKBACK_DAYS",
    "EW",
    "QUICK_TRAIN_LOOKBACK_DAYS",
    "RL_WINDOW",
    "SGDOnline",
    "StockEnv",
    "TRAIN_REFRESH_DELTA",
    "TRAIN_REFRESH_HOURS",
]
