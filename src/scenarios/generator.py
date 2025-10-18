"""Scenario generator producing EventShock objects from textual inputs."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Iterable, Sequence

from .types import CalendarItem, EventShock, Headline

_POSITIVE_TERMS = {
    "beat",
    "beats",
    "growth",
    "strong",
    "record",
    "surge",
    "improve",
    "rebound",
    "raise",
    "upgrade",
}

_NEGATIVE_TERMS = {
    "miss",
    "cuts",
    "weak",
    "down",
    "guidance",
    "slump",
    "warning",
    "downgrade",
    "probe",
    "delay",
    "slow",
}


class ScenarioGenerator:
    """Deterministic fallback generator backed by a prompt template."""

    def __init__(self, template_path: str | Path | None = None) -> None:
        if template_path is None:
            template_path = Path(__file__).with_name("prompt.txt")
        self.template_path = Path(template_path)
        self.template = self.template_path.read_text(encoding="utf-8")

    def build_prompt(
        self, headlines: Sequence[Headline], calendar_item: CalendarItem
    ) -> str:
        """Render the few-shot prompt for an LLM request."""

        headline_lines = [
            f"- {h.published.isoformat()} :: {h.text}" + (f" ({h.source})" if h.source else "")
            for h in sorted(headlines, key=lambda h: h.published)
        ]
        calendar_repr = json.dumps(
            {
                "symbol": calendar_item.symbol,
                "event_date": calendar_item.event_date.isoformat(),
                "kind": calendar_item.kind,
                "metadata": calendar_item.metadata,
            },
            separators=(",", ":"),
        )
        return self.template.format(
            headlines="\n".join(headline_lines) if headline_lines else "(none)",
            calendar_item=calendar_repr,
        )

    # ------------------------------------------------------------------
    # Heuristic fallback implementation
    # ------------------------------------------------------------------
    def generate(
        self,
        headlines: Sequence[Headline],
        calendar_item: CalendarItem,
        event_id: str | None = None,
    ) -> list[EventShock]:
        """Produce 1-5 EventShock objects with calibrated priors."""

        if not event_id:
            event_id = f"{calendar_item.symbol}-{calendar_item.event_date.date()}"

        if not headlines:
            sentiment_score = 0.0
        else:
            sentiment_score = self._score_headlines(headlines)

        base_prior = 0.55 if sentiment_score > 0 else 0.45 if sentiment_score < 0 else 0.5
        neutral_prior = max(0.1, 1.0 - base_prior)

        start = calendar_item.event_date - timedelta(hours=2)
        end = calendar_item.event_date + timedelta(days=1)

        scenarios: list[EventShock] = []
        variant_prefix = calendar_item.kind.replace(" ", "").lower()

        if sentiment_score > 0:
            scenarios.append(
                EventShock(
                    event_id=event_id,
                    variant=f"{variant_prefix}-upside",
                    window_start=start,
                    window_end=end,
                    prior=min(0.6, base_prior),
                    drift_bump=0.12,
                    vol_multiplier=1.25,
                    jump_intensity=0.4,
                    jump_mean=0.05,
                    jump_std=0.12,
                    description="Upside surprise aligned with positive headlines.",
                    metadata={"sentiment_score": sentiment_score},
                )
            )
        elif sentiment_score < 0:
            scenarios.append(
                EventShock(
                    event_id=event_id,
                    variant=f"{variant_prefix}-downside",
                    window_start=start,
                    window_end=end,
                    prior=min(0.6, base_prior),
                    drift_bump=-0.15,
                    vol_multiplier=1.35,
                    jump_intensity=0.45,
                    jump_mean=-0.06,
                    jump_std=0.16,
                    description="Downside risk prompted by negative coverage.",
                    metadata={"sentiment_score": sentiment_score},
                )
            )

        scenarios.append(
            EventShock(
                event_id=event_id,
                variant=f"{variant_prefix}-base",
                window_start=start,
                window_end=end,
                prior=min(0.4, neutral_prior),
                drift_bump=0.0,
                vol_multiplier=1.1,
                jump_intensity=0.25,
                jump_mean=-0.01,
                jump_std=0.08,
                description="Baseline outcome consistent with consensus.",
                metadata={"sentiment_score": sentiment_score},
            )
        )

        if len(scenarios) < 3:
            # Add tail scenario to maintain diversity.
            tail_direction = "up" if sentiment_score >= 0 else "down"
            scenarios.append(
                EventShock(
                    event_id=event_id,
                    variant=f"{variant_prefix}-tail-{tail_direction}",
                    window_start=start,
                    window_end=end,
                    prior=0.1,
                    drift_bump=0.25 if tail_direction == "up" else -0.3,
                    vol_multiplier=1.5,
                    jump_intensity=0.6,
                    jump_mean=0.08 if tail_direction == "up" else -0.1,
                    jump_std=0.2,
                    description="Extremal scenario covering fat-tail risk.",
                    metadata={"sentiment_score": sentiment_score},
                )
            )

        self._normalize_priors(scenarios)
        return scenarios

    # ------------------------------------------------------------------
    @staticmethod
    def _score_headlines(headlines: Sequence[Headline]) -> float:
        """Compute a crude sentiment score based on keyword tallies."""

        ordered = sorted(headlines, key=lambda h: h.published, reverse=True)
        if not ordered:
            return 0.0
        counts = Counter()
        decay = 0.0
        anchor = ordered[0].published
        for headline in ordered:
            tokens = headline.key_terms()
            score = sum(token in _POSITIVE_TERMS for token in tokens) - sum(
                token in _NEGATIVE_TERMS for token in tokens
            )
            age_days = max((anchor - headline.published).days, 0)
            weight = math.exp(-age_days / 14)
            counts["weighted_score"] += score * weight
            decay += weight
        if decay == 0:
            return 0.0
        return counts["weighted_score"] / decay

    @staticmethod
    def _normalize_priors(scenarios: Iterable[EventShock]) -> None:
        """Normalize priors within each event group so they sum to ≤1."""

        by_event: dict[str, list[EventShock]] = defaultdict(list)
        for shock in scenarios:
            by_event[shock.event_id].append(shock)

        for shocks in by_event.values():
            total = sum(max(shock.prior, 0.0) for shock in shocks)
            if total <= 1.0 or total == 0:
                continue
            scale = min(1.0 / total, 1.0)
            for shock in shocks:
                shock.prior = max(shock.prior, 0.0) * scale
