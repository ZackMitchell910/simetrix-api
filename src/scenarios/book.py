from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Mapping, MutableMapping, Sequence

from .generator import ScenarioGenerator
from .models import EventShock, ScenarioDiagnostics
from ..engine.types import StateVector


class ScenarioBook:
    """Assembles a curated list of ``EventShock`` instances for a symbol."""

    def __init__(self, generator: ScenarioGenerator | None = None):
        self.generator = generator or ScenarioGenerator()

    def build(
        self,
        symbol: str,
        asof: datetime,
        feeds: Mapping[str, object],
        state: StateVector,
    ) -> List[EventShock]:
        headlines: Sequence[str] = feeds.get("headlines", [])  # type: ignore[assignment]
        calendar_item: Mapping[str, object] = feeds.get("calendar", {})  # type: ignore[assignment]
        history: Sequence[Mapping[str, object]] = feeds.get("history", [])  # type: ignore[assignment]

        shocks = self.generator.generate(symbol=symbol, as_of=asof, headlines=headlines, calendar_item=calendar_item)
        return self.calibrate_priors(shocks, history)

    def calibrate_priors(
        self,
        shocks: Sequence[EventShock],
        history: Sequence[Mapping[str, object]],
    ) -> List[EventShock]:
        if not history:
            return list(shocks)

        diagnostics = ScenarioDiagnostics()
        hit_rates: MutableMapping[str, float] = defaultdict(float)
        counts: MutableMapping[str, int] = defaultdict(int)

        cutoff = datetime.utcnow() - timedelta(days=365)
        for record in history:
            timestamp = record.get("timestamp") or record.get("as_of")
            if isinstance(timestamp, datetime) and timestamp < cutoff:
                continue
            elif isinstance(timestamp, str):
                try:
                    parsed = datetime.fromisoformat(timestamp)
                except ValueError:
                    parsed = None
                if parsed and parsed < cutoff:
                    continue
            predicted = str(record.get("predicted"))
            actual = str(record.get("actual"))
            diagnostics.add(predicted, actual)
            counts[predicted] += 1
            if predicted == actual:
                hit_rates[predicted] += 1

        normalised_hit_rates: dict[str, float] = {}
        for key, total in counts.items():
            hits = hit_rates.get(key, 0.0)
            normalised_hit_rates[key] = hits / total if total else 0.5

        by_group: MutableMapping[str | None, List[EventShock]] = defaultdict(list)
        for shock in shocks:
            by_group[shock.mutually_exclusive_group].append(shock)

        calibrated: List[EventShock] = []
        for group, members in by_group.items():
            weighted_members: List[EventShock] = []
            for member in members:
                rate = normalised_hit_rates.get(member.shock_id, normalised_hit_rates.get(member.label, 0.5))
                adjusted_prior = member.prior * (0.5 + 0.5 * rate)
                enriched = member.copy_with_prior(adjusted_prior)
                enriched.metadata.setdefault("diagnostics", {})
                enriched.metadata["diagnostics"].update(diagnostics.as_dict())
                enriched.metadata["diagnostics"]["hit_rate"] = rate
                weighted_members.append(enriched)

            total = sum(max(member.prior, 0.0) for member in weighted_members)
            if total > 1.0:
                scale = 1.0 / total
            else:
                scale = 1.0
            calibrated.extend(member.copy_with_prior(member.prior * scale) for member in weighted_members)

        return calibrated


__all__ = ["ScenarioBook"]
