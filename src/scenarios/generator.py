"""Rule-based scenario generator backed by a prompt template."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Sequence

from .models import EventShock, normalise_priors


DEFAULT_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompt.txt")

_POSITIVE_KEYWORDS = {
    "beat", "beats", "record", "upbeat", "raises", "lift", "surge",
    "strong", "growth", "expands", "accelerates", "positive", "bullish",
}
_NEGATIVE_KEYWORDS = {
    "miss", "warning", "cuts", "lawsuit", "probe", "delay", "down",
    "weaker", "soft", "decline", "slump", "negative", "bearish",
}
_VOLATILITY_KEYWORDS = {
    "investigation", "regulator", "strike", "walkout", "emergency",
    "guidance", "update", "halts", "disruption", "volatility",
}


def _parse_dt(value: str | datetime | None, default: datetime) -> datetime:
    if value is None:
        return default
    if isinstance(value, datetime):
        return value
    # Support both "Z" suffix and explicit offsets.
    if isinstance(value, str):
        val = value.replace("Z", "+00:00") if value.endswith("Z") else value
        try:
            dt = datetime.fromisoformat(val)
        except ValueError:
            return default
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    return default


def _window(calendar_item: dict[str, object], fallback: datetime) -> tuple[datetime, datetime]:
    start = calendar_item.get("window_start") if calendar_item else None
    end = calendar_item.get("window_end") if calendar_item else None
    if start and end:
        start_dt = _parse_dt(start, fallback)  # type: ignore[arg-type]
        end_dt = _parse_dt(end, fallback)  # type: ignore[arg-type]
        if end_dt < start_dt:
            end_dt = start_dt
        return start_dt, end_dt
    event_dt = _parse_dt(calendar_item.get("datetime") if calendar_item else None, fallback)  # type: ignore[arg-type]
    return event_dt - timedelta(days=1), event_dt + timedelta(days=1)


@dataclass
class ScenarioGenerator:
    """Deterministic scenario generator leveraging heuristic few-shot patterns."""

    prompt_path: str = DEFAULT_PROMPT_PATH

    def __post_init__(self) -> None:
        if os.path.exists(self.prompt_path):
            with open(self.prompt_path, "r", encoding="utf-8") as handle:
                self.prompt_template = handle.read()
        else:
            self.prompt_template = ""

    def generate(
        self,
        headlines: Sequence[str],
        calendar_item: dict[str, object] | None,
        *,
        symbol: str | None = None,
        asof: datetime | None = None,
    ) -> list[EventShock]:
        """Generate EventShock instances from textual evidence.

        Parameters
        ----------
        headlines:
            Iterable of strings describing recent company news.
        calendar_item:
            Dictionary describing the scheduled event.  Expected keys include
            ``symbol`` and ``datetime`` (ISO 8601).  Optional ``window_start`` and
            ``window_end`` override the default +/- 1 day window.
        symbol:
            Optional ticker.  If omitted we fall back to the calendar symbol or
            ``"UNKNOWN"``.
        asof:
            Reference timestamp used when parsing missing calendar fields.
        """

        if asof is None:
            asof = datetime.now(tz=timezone.utc)

        calendar_item = calendar_item or {}
        symbol = symbol or str(calendar_item.get("symbol", "UNKNOWN"))
        event_dt = _parse_dt(calendar_item.get("datetime"), asof)
        window_start, window_end = _window(calendar_item, event_dt)

        positive_score, negative_score, vol_score = self._score_headlines(headlines)

        earnings_group = "earnings_outcome"
        base_prior = 0.35 + 0.05 * math.tanh((positive_score - negative_score) / 2)
        base_prior = float(min(max(base_prior, 0.15), 0.55))

        shocks_by_group: dict[str | None, list[EventShock]] = {earnings_group: []}

        # Inline/base case
        shocks_by_group[earnings_group].append(
            EventShock(
                symbol=symbol,
                label=f"{symbol} earnings inline",
                window_start=window_start,
                window_end=window_end,
                prior=base_prior,
                variant="Inline",
                description="Neutral base case anchored on current guidance.",
                drift=0.0001,
                volatility_scale=1.0 + 0.05 * vol_score,
                jump_intensity=0.05 + 0.02 * vol_score,
                jump_mean=0.0,
                jump_std=0.15,
                mutually_exclusive_group=earnings_group,
                evidence=self._select_evidence(headlines, limit=2),
                metadata={"prompt_template": bool(self.prompt_template)},
            )
        )

        if positive_score > 0:
            shocks_by_group.setdefault(earnings_group, []).append(
                EventShock(
                    symbol=symbol,
                    label=f"{symbol} earnings beat",
                    window_start=window_start,
                    window_end=window_end,
                    prior=min(0.6, 0.2 + 0.15 * positive_score),
                    variant="Beat",
                    description="Upside signals in recent headlines.",
                    drift=0.0005 + 0.0002 * positive_score,
                    volatility_scale=1.1 + 0.05 * vol_score,
                    jump_intensity=0.08 + 0.03 * positive_score,
                    jump_mean=0.08,
                    jump_std=0.18,
                    mutually_exclusive_group=earnings_group,
                    evidence=self._select_evidence(headlines, positive=True),
                    metadata={"prompt_template": bool(self.prompt_template)},
                )
            )

        if negative_score > 0:
            shocks_by_group.setdefault(earnings_group, []).append(
                EventShock(
                    symbol=symbol,
                    label=f"{symbol} earnings miss",
                    window_start=window_start,
                    window_end=window_end,
                    prior=min(0.6, 0.2 + 0.15 * negative_score),
                    variant="Miss",
                    description="Downside skew implied by coverage.",
                    drift=-0.0006 - 0.0002 * negative_score,
                    volatility_scale=1.2 + 0.08 * (vol_score + negative_score / 2),
                    jump_intensity=0.10 + 0.04 * negative_score,
                    jump_mean=-0.09,
                    jump_std=0.22,
                    mutually_exclusive_group=earnings_group,
                    evidence=self._select_evidence(headlines, positive=False),
                    metadata={"prompt_template": bool(self.prompt_template)},
                )
            )

        if vol_score > 0:
            # Non-exclusive volatility regime shock
            shocks_by_group.setdefault(None, []).append(
                EventShock(
                    symbol=symbol,
                    label=f"{symbol} volatility watch",
                    window_start=window_start - timedelta(days=1),
                    window_end=window_end + timedelta(days=1),
                    prior=min(0.4, 0.1 + 0.1 * vol_score),
                    variant="Volatility",
                    description="Idiosyncratic catalysts elevate realised volatility.",
                    drift=0.0,
                    volatility_scale=1.3 + 0.1 * vol_score,
                    jump_intensity=0.12 + 0.03 * vol_score,
                    jump_mean=0.0,
                    jump_std=0.30,
                    mutually_exclusive_group=None,
                    evidence=self._select_evidence(headlines, keyword_pool=_VOLATILITY_KEYWORDS),
                    metadata={"prompt_template": bool(self.prompt_template)},
                )
            )

        # Normalise priors per mutually exclusive bucket and trim to maximum of 5 shocks.
        grouped = normalise_priors({group: variants for group, variants in shocks_by_group.items() if variants})
        grouped.sort(key=lambda shock: shock.prior, reverse=True)
        return grouped[:5]

    def _score_headlines(self, headlines: Sequence[str]) -> tuple[int, int, int]:
        pos = neg = vol = 0
        for headline in headlines:
            text = headline.lower()
            if any(keyword in text for keyword in _POSITIVE_KEYWORDS):
                pos += 1
            if any(keyword in text for keyword in _NEGATIVE_KEYWORDS):
                neg += 1
            if any(keyword in text for keyword in _VOLATILITY_KEYWORDS):
                vol += 1
        return pos, neg, vol

    def _select_evidence(
        self,
        headlines: Sequence[str],
        *,
        positive: bool | None = None,
        keyword_pool: Iterable[str] | None = None,
        limit: int = 2,
    ) -> list[str]:
        if not headlines:
            return []
        selected: list[str] = []
        pool = set(keyword_pool or [])
        for headline in headlines:
            text = headline.lower()
            if positive is True and not any(k in text for k in _POSITIVE_KEYWORDS):
                continue
            if positive is False and not any(k in text for k in _NEGATIVE_KEYWORDS):
                continue
            if pool and not any(k in text for k in pool):
                continue
            selected.append(headline)
            if len(selected) >= limit:
                break
        if not selected:
            selected = list(headlines[:limit])
        return selected
