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
