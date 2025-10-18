from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.scenarios.schema import EventShock
from src.state.contracts import StateVector

__all__ = ["DynamicFactorInputs", "apply_dynamic_factors"]


@dataclass(slots=True)
class DynamicFactorInputs:
    drift_prior: float | None = None
    drift_confidence: float = 0.0
    vol_prior: float | None = None
    vol_confidence: float = 0.0
    jump_intensity: float | None = None
    jump_mean: float | None = None
    jump_vol: float | None = None
    news_bias: float = 0.0
    sentiment_bias: float = 0.0
    macro: Mapping[str, Any] = field(default_factory=dict)
    sentiment: Mapping[str, Any] = field(default_factory=dict)
    cross: Mapping[str, Any] = field(default_factory=dict)
    events: Sequence[EventShock] = field(default_factory=list)
    provenance: Mapping[str, Any] = field(default_factory=dict)


def _kalman_blend(prior: float, signal: float, confidence: float) -> float:
    weight = max(0.0, min(1.0, float(confidence)))
    return float(prior * (1.0 - weight) + signal * weight)


def _bounded(value: float, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def apply_dynamic_factors(prior: StateVector, inputs: DynamicFactorInputs) -> StateVector:
    """Fuse exogenous signals into a new :class:`StateVector`.

    The function intentionally implements a lightweight Kalman-style update so
    that downstream code has sane defaults even without the heavy statistical
    stack in place.  News and sentiment enter as small tilts on the drift,
    options-implied volatility flows through ``vol_prior`` and macro/credit
    spreads flow via the ``macro`` and ``cross`` payloads.
    """

    drift = prior.drift_annual
    if inputs.drift_prior is not None:
        drift = _kalman_blend(drift, float(inputs.drift_prior), inputs.drift_confidence)
    drift += 0.05 * float(inputs.news_bias) + 0.03 * float(inputs.sentiment_bias)
    drift = _bounded(drift, -1.5, 1.5)

    vol = prior.vol_annual
    if inputs.vol_prior is not None:
        vol = _kalman_blend(vol, float(inputs.vol_prior), inputs.vol_confidence)
    vol = _bounded(vol, 1e-4, 5.0)

    jump_intensity = inputs.jump_intensity if inputs.jump_intensity is not None else prior.jump_intensity
    jump_mean = inputs.jump_mean if inputs.jump_mean is not None else prior.jump_mean
    jump_vol = inputs.jump_vol if inputs.jump_vol is not None else prior.jump_vol
    jump_intensity = _bounded(float(jump_intensity), 0.0, 5.0)
    jump_vol = _bounded(float(jump_vol), 0.0, 5.0)

    macro = {**dict(prior.macro), **dict(inputs.macro)}
    sentiment = {**dict(prior.sentiment), **dict(inputs.sentiment)}
    cross = {**dict(prior.cross), **dict(inputs.cross)}
    provenance = {**dict(prior.provenance), **dict(inputs.provenance)}

    events = list(inputs.events or prior.events)

    return StateVector(
        t=prior.t,
        spot=prior.spot,
        drift_annual=drift,
        vol_annual=vol,
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_vol=jump_vol,
        regime=prior.regime,
        macro=macro,
        sentiment=sentiment,
        events=events,
        cross=cross,
        provenance=provenance,
    )
