from __future__ import annotations

import math
from datetime import datetime
from typing import Iterable, Tuple

import numpy as np

from .types import Artifact, StateVector


def anchor_to_implied_vol(artifact: Artifact, state: StateVector, year_basis: float = 252.0) -> None:
    if not state.iv_by_expiry:
        return
    target_sigma = _interpolate_iv(state.iv_by_expiry, state.asof, artifact.times[-1], year_basis)
    if target_sigma is None or target_sigma <= 0:
        return
    horizon_years = artifact.times[-1] / year_basis
    if horizon_years <= 0:
        return
    target_var = (target_sigma ** 2) * horizon_years
    terminal = artifact.paths[:, -1]
    start = artifact.paths[:, 0]
    log_returns = np.log(np.maximum(terminal, 1e-12) / np.maximum(start, 1e-12))
    current_var = float(np.var(log_returns, ddof=1))
    if current_var <= 0:
        return
    scale = math.sqrt(target_var / current_var)
    if not math.isfinite(scale) or scale <= 0:
        return

    rel = artifact.paths / artifact.paths[:, [0]]
    log_rel = np.log(np.maximum(rel, 1e-12))
    scaled_log_rel = log_rel * scale
    artifact.paths = artifact.paths[:, [0]] * np.exp(scaled_log_rel)
    artifact.metadata["iv_anchor"] = {
        "target_sigma": target_sigma,
        "scale": scale,
        "horizon_years": horizon_years,
    }


def _interpolate_iv(
    curve: Iterable[Tuple[datetime, float]] | dict[datetime, float],
    asof: datetime,
    horizon_days: float,
    year_basis: float,
) -> float | None:
    pairs = list(curve.items() if isinstance(curve, dict) else curve)
    if not pairs:
        return None
    tenors = []
    vols = []
    for expiry, vol in sorted(pairs, key=lambda kv: kv[0]):
        tenor_years = max(0.0, (expiry - asof).days / year_basis)
        tenors.append(tenor_years)
        vols.append(float(vol))
    target_years = horizon_days / year_basis
    if target_years <= tenors[0]:
        return vols[0]
    for idx in range(1, len(tenors)):
        if target_years <= tenors[idx]:
            t0, t1 = tenors[idx - 1], tenors[idx]
            v0, v1 = vols[idx - 1], vols[idx]
            weight = 0.0 if t1 == t0 else (target_years - t0) / (t1 - t0)
            return (1 - weight) * v0 + weight * v1
    return vols[-1]
