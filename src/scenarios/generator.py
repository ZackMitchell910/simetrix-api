from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

from .models import EventShock

DEFAULT_TEMPLATE_PATH = Path(__file__).with_name("prompt.txt")


@dataclass(slots=True)
class ScenarioPromptBuilder:
    """Utility to render the LLM prompt used for manual review/testing."""

    template_path: Path = DEFAULT_TEMPLATE_PATH

    def load_template(self) -> str:
        return self.template_path.read_text(encoding="utf-8")

    def build(
        self,
        symbol: str,
        as_of: datetime,
        headlines: Sequence[str],
        calendar_item: Mapping[str, object],
    ) -> str:
        template = self.load_template()
        calendar_text = "{} {} {}".format(
            calendar_item.get("symbol", symbol),
            calendar_item.get("date", calendar_item.get("datetime", "")),
            calendar_item.get("descriptor", calendar_item.get("event", "Earnings")),
        ).strip()
        payload = {
            "symbol": symbol,
            "as_of": as_of.isoformat(),
            "headlines": list(headlines),
            "calendar": calendar_text,
        }
        return f"{template}\n\nCurrent request:\n{json.dumps(payload, indent=2)}"


class ScenarioGenerator:
    """Heuristic scenario generator backed by a few-shot prompt template.

    The generator is designed to operate without requiring an actual LLM during
    automated tests. It uses lightweight sentiment heuristics to seed
    ``EventShock`` objects while ensuring probability mass is properly
    normalised across mutually exclusive variants.
    """

    POSITIVE_TERMS = {
        "beat",
        "strong",
        "growth",
        "upgrade",
        "record",
        "expand",
        "accelerat",
        "surge",
        "raise",
        "positive",
        "wins",
        "contract",
    }
    NEGATIVE_TERMS = {
        "miss",
        "downgrade",
        "weak",
        "recall",
        "delay",
        "negative",
        "lawsuit",
        "fraud",
        "probe",
        "cut",
        "guidance",
        "investigat",
    }

    def __init__(self, prompt_builder: Optional[ScenarioPromptBuilder] = None):
        self.prompt_builder = prompt_builder or ScenarioPromptBuilder()

    def generate(
        self,
        symbol: str,
        as_of: datetime,
        headlines: Sequence[str],
        calendar_item: Mapping[str, object],
    ) -> List[EventShock]:
        if not headlines:
            raise ValueError("At least one headline is required to seed scenarios")

        event_offset_days = self._event_offset_days(as_of, calendar_item)
        group = "earnings-direction"
        sentiment = self._sentiment_score(headlines)

        shocks: List[EventShock] = []
        positive_weight = max(sentiment, 0)
        negative_weight = max(-sentiment, 0)
        baseline_weight = 0.2 if sentiment else 0.4

        # Cap heuristics to maintain stability
        positive_weight = min(positive_weight, 1.0)
        negative_weight = min(negative_weight, 1.0)

        window_start = max(event_offset_days - 1.0, 0.0)
        window_end = event_offset_days + 1.0

        if positive_weight > 0:
            shocks.append(
                EventShock(
                    shock_id="earnings-beat",
                    label="Upside surprise",
                    prior=0.4 + 0.3 * positive_weight,
                    window_start=window_start,
                    window_end=window_end,
                    drift=0.05 + 0.05 * positive_weight,
                    vol_multiplier=1.2 + 0.3 * positive_weight,
                    jump_intensity=0.5 + 0.4 * positive_weight,
                    jump_mean=0.05 + 0.03 * positive_weight,
                    jump_std=0.03 + 0.02 * positive_weight,
                    mutually_exclusive_group=group,
                    metadata={
                        "sentiment_score": sentiment,
                        "evidence": self._top_headlines(headlines, positive=True),
                        "calendar": dict(calendar_item),
                    },
                )
            )

        if negative_weight > 0:
            shocks.append(
                EventShock(
                    shock_id="earnings-miss",
                    label="Downside surprise",
                    prior=0.4 + 0.3 * negative_weight,
                    window_start=window_start,
                    window_end=window_end,
                    drift=-0.06 - 0.04 * negative_weight,
                    vol_multiplier=1.3 + 0.4 * negative_weight,
                    jump_intensity=0.6 + 0.5 * negative_weight,
                    jump_mean=-0.07 - 0.05 * negative_weight,
                    jump_std=0.04 + 0.03 * negative_weight,
                    mutually_exclusive_group=group,
                    metadata={
                        "sentiment_score": sentiment,
                        "evidence": self._top_headlines(headlines, positive=False),
                        "calendar": dict(calendar_item),
                    },
                )
            )

        shocks.append(
            EventShock(
                shock_id="earnings-inline",
                label="Inline quarter",
                prior=baseline_weight,
                window_start=window_start,
                window_end=window_end,
                drift=0.0,
                vol_multiplier=1.05,
                jump_intensity=0.2,
                jump_mean=0.0,
                jump_std=0.02,
                mutually_exclusive_group=group,
                metadata={
                    "sentiment_score": sentiment,
                    "evidence": self._top_headlines(headlines),
                    "calendar": dict(calendar_item),
                },
            )
        )

        shocks = self._normalise_priors(shocks)
        return shocks[:5]

    @staticmethod
    def _event_offset_days(as_of: datetime, calendar_item: Mapping[str, object]) -> float:
        raw = calendar_item.get("event_time") or calendar_item.get("datetime") or calendar_item.get("date")
        if raw is None:
            return 0.0
        if isinstance(raw, (int, float)):
            event_time = datetime.utcfromtimestamp(float(raw))
        elif isinstance(raw, datetime):
            event_time = raw
        else:
            try:
                event_time = datetime.fromisoformat(str(raw))
            except ValueError:
                event_time = as_of
        delta = max(event_time - as_of, timedelta(0))
        return delta.days + delta.seconds / 86400.0

    @classmethod
    def _sentiment_score(cls, headlines: Sequence[str]) -> float:
        score = 0.0
        for headline in headlines:
            lowered = headline.lower()
            if any(term in lowered for term in cls.POSITIVE_TERMS):
                score += 1.0
            if any(term in lowered for term in cls.NEGATIVE_TERMS):
                score -= 1.0
        return max(min(score / max(len(headlines), 1), 1.0), -1.0)

    @staticmethod
    def _top_headlines(headlines: Sequence[str], positive: Optional[bool] = None) -> Sequence[str]:
        if positive is None:
            return list(headlines[:3])
        filtered = []
        for headline in headlines:
            head_lower = headline.lower()
            if positive and any(term in head_lower for term in ScenarioGenerator.POSITIVE_TERMS):
                filtered.append(headline)
            elif positive is False and any(term in head_lower for term in ScenarioGenerator.NEGATIVE_TERMS):
                filtered.append(headline)
            if len(filtered) == 3:
                break
        return filtered or list(headlines[:3])

    @staticmethod
    def _normalise_priors(shocks: Sequence[EventShock]) -> List[EventShock]:
        by_group: dict[Optional[str], List[EventShock]] = {}
        for shock in shocks:
            by_group.setdefault(shock.mutually_exclusive_group, []).append(shock)

        normalised: List[EventShock] = []
        for group, members in by_group.items():
            total = sum(max(member.prior, 0.0) for member in members)
            if total <= 1.0:
                normalised.extend(members)
                continue
            scale = 1.0 / total
            for member in members:
                normalised.append(member.copy_with_prior(prior=member.prior * scale))
        return normalised

    def build_prompt(
        self,
        symbol: str,
        as_of: datetime,
        headlines: Sequence[str],
        calendar_item: Mapping[str, object],
    ) -> str:
        """Expose prompt builder for diagnostics and interactive workflows."""

        return self.prompt_builder.build(symbol=symbol, as_of=as_of, headlines=headlines, calendar_item=calendar_item)


__all__ = ["ScenarioGenerator", "ScenarioPromptBuilder"]
