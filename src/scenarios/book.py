from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence

from .generator import ScenarioGenerator
from .models import EventShock


class ScenarioBook:
    """Assemble a calibrated set of :class:`EventShock` objects for simulation."""

    def __init__(self, generator: ScenarioGenerator | None = None, lookback_days: int = 365):
        self.generator = generator or ScenarioGenerator()
        self.lookback_days = lookback_days

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
        feeds: Mapping[str, Any],
        state: Any,
    ) -> list[EventShock]:
        earnings = self._select_earnings(symbol, feeds)
        if not earnings:
            return []
        event_time = earnings.get("window_start") or earnings.get("datetime")
        if not isinstance(event_time, datetime):
            event_time = asof
        headlines = list(feeds.get("headlines", []))
        shocks = self.generator.generate(symbol, event_time, headlines, earnings)
        history = feeds.get("history") or feeds.get("scenario_history")
        if history:
            shocks = self.calibrate_priors(shocks, history)
        return shocks

    def calibrate_priors(
        self,
        shocks: Sequence[EventShock],
        history: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ) -> list[EventShock]:
        shocks = list(shocks)
        if not shocks:
            return []
        records, asof = self._extract_history(history)
        if not records:
            return shocks
        cutoff = asof - timedelta(days=self.lookback_days)
        filtered = [rec for rec in records if rec.get("timestamp") and rec["timestamp"] >= cutoff]
        if not filtered:
            return shocks

        variants = {shock.variant for shock in shocks}
        stats: dict[str, dict[str, int]] = {
            variant: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for variant in variants
        }
        for rec in filtered:
            predicted = rec.get("predicted") or rec.get("variant")
            realised = rec.get("realised") or rec.get("actual")
            for variant in variants:
                if predicted == variant and realised == variant:
                    stats[variant]["tp"] += 1
                elif predicted == variant and realised != variant:
                    stats[variant]["fp"] += 1
                elif predicted != variant and realised == variant:
                    stats[variant]["fn"] += 1
                else:
                    stats[variant]["tn"] += 1

        adjustments: dict[str, float] = {}
        for variant, cm in stats.items():
            tp = cm["tp"]
            fp = cm["fp"]
            denom = tp + fp
            hit_rate = tp / denom if denom > 0 else 0.5
            adjustments[variant] = 0.5 + 0.5 * hit_rate

        adjusted = [
            replace(shock, prior=shock.prior * adjustments.get(shock.variant, 1.0))
            for shock in shocks
        ]
        normalised = self._normalise(adjusted)
        diagnostics = {
            "asof": asof,
            "confusion_matrix": stats,
            "records": len(filtered),
        }
        return [shock.with_metadata(diagnostics=diagnostics) for shock in normalised]

    def _normalise(self, shocks: Iterable[EventShock]) -> list[EventShock]:
        shocks = list(shocks)
        total = sum(shock.prior for shock in shocks)
        if total <= 0:
            equal = 1.0 / len(shocks)
            return [replace(shock, prior=equal) for shock in shocks]
        inv_total = 1.0 / total
        return [replace(shock, prior=min(1.0, shock.prior * inv_total)) for shock in shocks]

    @staticmethod
    def _select_earnings(symbol: str, feeds: Mapping[str, Any]) -> Mapping[str, Any] | None:
        items = feeds.get("earnings_calendar") or []
        for item in items:
            if item.get("symbol", symbol) == symbol:
                return item
        return None

    @staticmethod
    def _extract_history(
        history: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ) -> tuple[list[Mapping[str, Any]], datetime]:
        if isinstance(history, Mapping):
            asof = history.get("asof") or datetime.utcnow()
            records = history.get("records") or []
        else:
            asof = max((rec.get("timestamp") for rec in history if rec.get("timestamp")), default=datetime.utcnow())
            records = list(history)
        return list(records), asof
