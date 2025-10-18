from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any


def _to_float(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _ensure_median(pick: Mapping[str, Any], spot: float | None, base: float | None) -> float | None:
    median = _to_float(pick.get("median_return_pct"))
    if median is not None:
        return median
    if base is None or spot is None or spot <= 0:
        return None
    try:
        return ((base / spot) - 1.0) * 100.0
    except ZeroDivisionError:
        return None


def _format_blurb(symbol: str, horizon: int, prob: float | None, median: float | None) -> str:
    sym = symbol.upper().strip()
    pieces: list[str] = []
    if prob is not None:
        pieces.append(f"P(up)≈{prob * 100.0:.0f}%")
    if median is not None:
        pieces.append(f"median ≈{median:.1f}%")
    if pieces:
        details = ", ".join(pieces)
        return f"{sym}: {horizon}d outlook {details}"
    return f"{sym}: {horizon}d outlook pending signals"


def ensure_daily_pick_fields(pick: Mapping[str, Any], *, horizon_default: int | None = None) -> dict[str, Any]:
    data = dict(pick or {})
    symbol = str(data.get("symbol") or "").upper()
    horizon = data.get("horizon_days")
    if horizon is None:
        horizon = horizon_default if horizon_default is not None else 30
    try:
        horizon = int(horizon)
    except Exception:
        horizon = 30
    prob = _to_float(data.get("prob_up_end") or data.get("prob_up_30d") or data.get("prob_up"))
    base = _to_float(data.get("base_price"))
    spot = _to_float(data.get("spot_price") or data.get("spot0") or data.get("spot"))
    if spot is None and base is not None and data.get("median_return_pct") is not None:
        median_candidate = _to_float(data.get("median_return_pct"))
        if median_candidate is not None:
            spot = base / (1.0 + median_candidate / 100.0)
    if spot is None and base is not None:
        spot = base
    median = _ensure_median(data, spot, base)
    blurb = str(data.get("blurb") or "").strip()
    if not blurb:
        blurb = _format_blurb(symbol, horizon, prob, median)

    data.update(
        {
            "symbol": symbol,
            "horizon_days": horizon,
            "prob_up_end": prob,
            "median_return_pct": median,
            "spot_price": spot,
            "blurb": blurb,
        }
    )
    return data


def ensure_daily_snapshot(payload: MutableMapping[str, Any], *, default_horizon: int | None = None) -> MutableMapping[str, Any]:
    for key in ("equity", "crypto"):
        pick = payload.get(key)
        if isinstance(pick, Mapping):
            payload[key] = ensure_daily_pick_fields(pick, horizon_default=default_horizon)

    for list_key in ("equity_top", "crypto_top", "equity_finalists", "crypto_finalists"):
        items = payload.get(list_key)
        if isinstance(items, Sequence):
            payload[list_key] = [
                ensure_daily_pick_fields(item, horizon_default=default_horizon)
                if isinstance(item, Mapping)
                else item
                for item in items
            ]
    return payload


__all__ = ["ensure_daily_pick_fields", "ensure_daily_snapshot"]
