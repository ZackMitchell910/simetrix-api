"""Scenario book assembly and calibration utilities."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Iterable, Mapping, Sequence

from .generator import ScenarioGenerator
from .types import CalendarItem, EventShock, Headline, HistoricalOutcome
from ..engine.state import StateVector


class ScenarioBook:
    """Coordinate prompt driven scenarios with historical calibration."""

    def __init__(self, generator: ScenarioGenerator | None = None) -> None:
        self.generator = generator or ScenarioGenerator()
        self.diagnostics: dict[str, object] = {}

    def build(
        self,
        symbol: str,
        asof: datetime,
        feeds: Mapping[str, object],
        state: StateVector,
    ) -> list[EventShock]:
        """Assemble scenarios for the provided symbol."""

        headlines = self._extract_headlines(feeds.get("headlines"), symbol)
        calendar_item = self._find_calendar_item(feeds.get("calendar"), symbol, asof)
        if calendar_item is None:
            raise ValueError(f"No calendar item found for {symbol}")

        shocks = self.generator.generate(headlines, calendar_item)
        history = feeds.get("history")
        shocks = self.calibrate_priors(shocks, history)
        return shocks

    def calibrate_priors(
        self, shocks: Sequence[EventShock], history: object | None
    ) -> list[EventShock]:
        """Re-weight priors using hit rates derived from history."""

        if not history:
            self.diagnostics["confusion_matrix"] = {}
            return list(shocks)

        outcomes = list(self._coerce_history(history))
        lookback_cutoff = datetime.utcnow() - timedelta(days=365)
        filtered = [o for o in outcomes if o.timestamp >= lookback_cutoff]

        confusion: dict[str, Counter[str]] = defaultdict(Counter)
        hit_rate: dict[str, float] = {}
        for outcome in filtered:
            confusion[outcome.predicted_variant][outcome.actual_variant] += outcome.weight

        for variant, counts in confusion.items():
            total = sum(counts.values())
            hits = counts.get(variant, 0.0)
            if total == 0:
                hit_rate[variant] = 0.5
            else:
                hit_rate[variant] = (hits + 1.0) / (total + 2.0)

        self.diagnostics["confusion_matrix"] = {
            variant: dict(counter) for variant, counter in confusion.items()
        }

        avg_hit_rate = sum(hit_rate.values()) / len(hit_rate) if hit_rate else 1.0
        reweighted: list[EventShock] = []
        grouped: dict[str, list[EventShock]] = defaultdict(list)
        for shock in shocks:
            grouped[shock.event_id].append(shock)

        for event_id, event_shocks in grouped.items():
            adjusted = []
            for shock in event_shocks:
                prior = max(shock.prior, 0.0)
                multiplier = hit_rate.get(shock.variant, avg_hit_rate)
                scale = multiplier / avg_hit_rate if avg_hit_rate else 1.0
                adjusted.append(shock.copy_with(prior=prior * scale))
            total = sum(s.prior for s in adjusted)
            if total > 1.0 and total > 0:
                scale = 1.0 / total
                adjusted = [s.copy_with(prior=s.prior * scale) for s in adjusted]
            reweighted.extend(adjusted)

        return reweighted

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_headlines(feed: object, symbol: str) -> list[Headline]:
        if not feed:
            return []
        headlines: list[Headline] = []
        for item in feed:  # type: ignore[assignment]
            if isinstance(item, Headline):
                headline = item
            else:
                headline = Headline(
                    published=_coerce_datetime(item.get("published")),
                    text=item.get("text", ""),
                    source=item.get("source"),
                )
            if symbol.upper() in headline.text.upper():
                headlines.append(headline)
        return sorted(headlines, key=lambda h: h.published, reverse=True)

    @staticmethod
    def _find_calendar_item(
        feed: object, symbol: str, asof: datetime
    ) -> CalendarItem | None:
        if not feed:
            return None
        symbol = symbol.upper()
        closest: CalendarItem | None = None
        min_delta = timedelta.max
        for item in feed:  # type: ignore[assignment]
            if isinstance(item, CalendarItem):
                calendar_item = item
            else:
                calendar_item = CalendarItem(
                    symbol=item.get("symbol", ""),
                    event_date=_coerce_datetime(item.get("event_date")),
                    kind=item.get("kind", "earnings"),
                    metadata=item.get("metadata", {}),
                )
            if calendar_item.symbol.upper() != symbol:
                continue
            delta = abs(calendar_item.event_date - asof)
            if delta < min_delta:
                closest, min_delta = calendar_item, delta
        return closest

    @staticmethod
    def _coerce_history(history: Iterable[object]) -> Iterable[HistoricalOutcome]:
        for item in history:
            if isinstance(item, HistoricalOutcome):
                yield item
            else:
                yield HistoricalOutcome(
                    timestamp=_coerce_datetime(item.get("timestamp")),
                    event_id=item.get("event_id", ""),
                    predicted_variant=item.get("predicted_variant", ""),
                    actual_variant=item.get("actual_variant", ""),
                    weight=float(item.get("weight", 1.0)),
                )


def _coerce_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    raise TypeError(f"Unsupported datetime value: {value!r}")
