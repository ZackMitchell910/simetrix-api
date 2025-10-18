from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

from .models import EventShock, ShockOverride

_POSITIVE_TOKENS = {
    "beat",
    "beats",
    "surge",
    "strong",
    "raises",
    "upgraded",
    "record",
    "accelerating",
    "expands",
    "rebound",
}
_NEGATIVE_TOKENS = {
    "miss",
    "cuts",
    "guidance",
    "downgrade",
    "slows",
    "weak",
    "lawsuit",
    "recall",
    "probe",
    "delays",
    "pressure",
}


def _score_headlines(headlines: Sequence[str]) -> float:
    score = 0.0
    for line in headlines:
        text = line.lower()
        for token in _POSITIVE_TOKENS:
            if token in text:
                score += 1.0
        for token in _NEGATIVE_TOKENS:
            if token in text:
                score -= 1.0
    return float(score)


def _normalise_priors(shocks: Iterable[EventShock]) -> list[EventShock]:
    shocks = list(shocks)
    if not shocks:
        return shocks
    priors = [max(0.0, min(1.0, s.prior)) for s in shocks]
    total = float(sum(priors))
    if total <= 0:
        priors = [1.0 / len(shocks)] * len(shocks)
        total = 1.0
    inv_total = 1.0 / total
    return [replace(shock, prior=min(1.0, p * inv_total)) for shock, p in zip(shocks, priors)]


class ScenarioGenerator:
    """Generate :class:`EventShock` objects from qualitative inputs."""

    def __init__(self, prompt_path: str | Path | None = None):
        path = Path(prompt_path or Path(__file__).with_name("prompt.txt"))
        self.prompt_template = path.read_text(encoding="utf-8")

    def build_prompt(
        self,
        symbol: str,
        event_time: datetime,
        headlines: Sequence[str],
    ) -> str:
        lines = [
            "Ticker: {symbol}".format(symbol=symbol),
            "Event: Earnings {ts}".format(ts=event_time.isoformat()),
            "Headlines:",
        ]
        lines.extend(f"{idx}. {line}" for idx, line in enumerate(headlines, start=1))
        return f"{self.prompt_template}\n\n" + "\n".join(lines)

    def _base_scenarios(
        self,
        symbol: str,
        event_time: datetime,
        sentiment: float,
        earnings_item: dict,
    ) -> list[EventShock]:
        volatility_hint = float(earnings_item.get("implied_vol", 0.35) or 0.35)
        vol_scale = max(0.2, min(1.5, volatility_hint))
        beat_prior = 0.35 + 0.05 * sentiment
        miss_prior = 0.25 - 0.05 * sentiment
        inline_prior = 1.0 - beat_prior - miss_prior
        priors = _normalise_priors(
            [
                EventShock(
                    symbol=symbol,
                    name="Earnings",
                    variant="Beat",
                    prior=max(0.05, beat_prior),
                    window_start=event_time,
                    window_end=event_time,
                    override=ShockOverride(
                        drift_bump=0.012 + 0.002 * sentiment,
                        vol_multiplier=1.0 + 0.15 * vol_scale,
                        jump_intensity=0.20 * vol_scale,
                        jump_mean=0.04,
                        jump_std=0.08,
                    ),
                    metadata={"rationale": "Positive momentum into the print."},
                ),
                EventShock(
                    symbol=symbol,
                    name="Earnings",
                    variant="Inline",
                    prior=max(0.1, inline_prior),
                    window_start=event_time,
                    window_end=event_time,
                    override=ShockOverride(
                        drift_bump=0.001 * (1.0 + sentiment),
                        vol_multiplier=0.9 + 0.05 * vol_scale,
                        jump_intensity=0.05 * vol_scale,
                        jump_mean=0.0,
                        jump_std=0.04,
                    ),
                    metadata={"rationale": "Consensus expectations realised."},
                ),
                EventShock(
                    symbol=symbol,
                    name="Earnings",
                    variant="Miss",
                    prior=max(0.05, miss_prior),
                    window_start=event_time,
                    window_end=event_time,
                    override=ShockOverride(
                        drift_bump=-0.015 + 0.002 * sentiment,
                        vol_multiplier=1.25 + 0.10 * vol_scale,
                        jump_intensity=0.25 * vol_scale,
                        jump_mean=-0.05,
                        jump_std=0.09,
                    ),
                    metadata={"rationale": "Downside surprise risk from execution."},
                ),
            ]
        )
        return priors

    def generate(
        self,
        symbol: str,
        event_time: datetime,
        headlines: Sequence[str],
        earnings_item: dict,
    ) -> list[EventShock]:
        sentiment = _score_headlines(headlines)
        base = self._base_scenarios(symbol, event_time, sentiment, earnings_item)
        prompt = self.build_prompt(symbol, event_time, headlines)
        return [shock.with_metadata(prompt=prompt, sentiment=sentiment) for shock in base]
