"""Scenario book orchestration and prior calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Sequence

from .generator import ScenarioGenerator
from .models import EventShock, normalise_priors
from src.engine.base import StateVector


@dataclass
class ScenarioBook:
    """High level helper to assemble and calibrate scenario shocks."""

    generator: ScenarioGenerator = field(default_factory=ScenarioGenerator)
    lookback_days: int = 365

    diagnostics: dict[str, object] = field(default_factory=dict, init=False)

    def build(
        self,
        symbol: str,
        asof: datetime,
        feeds: dict[str, object],
        state: StateVector,
    ) -> list[EventShock]:
        """Construct a list of scenarios for a given symbol."""

        headlines: Sequence[str] = tuple(feeds.get("headlines", []))  # type: ignore[arg-type]
        calendar_item: dict[str, object] | None = feeds.get("calendar")  # type: ignore[assignment]
        history: Sequence[dict[str, object]] = tuple(feeds.get("history", []))  # type: ignore[arg-type]

        shocks = self.generator.generate(headlines, calendar_item, symbol=symbol, asof=asof)
        calibrated = self.calibrate_priors(shocks, history)
        self.diagnostics["last_generated"] = [shock.to_dict() for shock in calibrated]
        self.diagnostics["state_symbol"] = state.symbol
        self.diagnostics["asof"] = asof.isoformat()
        return calibrated

    def calibrate_priors(
        self, shocks: Sequence[EventShock], history: Sequence[dict[str, object]]
    ) -> list[EventShock]:
        """Re-weight priors using historical hit rates."""

        calibrator = _PriorCalibrator(history, self.lookback_days)
        adjusted = calibrator.apply(shocks)
        self.diagnostics["confusion_matrix"] = calibrator.confusion_matrix
        self.diagnostics["hit_rates"] = calibrator.hit_rates
        return adjusted


@dataclass
class _HistoryRecord:
    variant: str
    timestamp: datetime
    actual: bool
    predicted: bool
    group: str | None


class _PriorCalibrator:
    def __init__(self, history: Sequence[dict[str, object]], lookback_days: int) -> None:
        self.records = self._parse_history(history, lookback_days)
        self.hit_rates = self._compute_hit_rates()
        self.confusion_matrix = self._compute_confusion_matrix()
        self.baseline_rate = self._baseline_rate()

    def apply(self, shocks: Sequence[EventShock]) -> list[EventShock]:
        groups: dict[str | None, list[EventShock]] = {}
        for shock in shocks:
            rate = self.hit_rates.get(shock.variant, self.baseline_rate)
            if self.baseline_rate > 0:
                scale = rate / self.baseline_rate
            else:
                scale = 1.0
            scaled_prior = max(0.0, min(shock.prior * scale, 0.95))
            groups.setdefault(shock.mutually_exclusive_group, []).append(shock.with_prior(scaled_prior))
        return normalise_priors(groups)

    def _parse_history(
        self, history: Sequence[dict[str, object]], lookback_days: int
    ) -> list[_HistoryRecord]:
        if not history:
            return []
        asof = datetime.now(tz=timezone.utc)
        cutoff = asof - timedelta(days=lookback_days)
        records: list[_HistoryRecord] = []
        for entry in history:
            variant = str(entry.get("variant") or entry.get("label") or "")
            if not variant:
                continue
            timestamp = entry.get("timestamp") or entry.get("asof")
            ts = self._parse_timestamp(timestamp, asof)
            if ts < cutoff:
                continue
            actual = bool(entry.get("actual") or entry.get("realised") or entry.get("realized") or False)
            predicted = bool(entry.get("predicted", True))
            group = entry.get("group") or entry.get("mutually_exclusive_group")
            group_str = str(group) if group is not None else None
            records.append(_HistoryRecord(variant=variant, timestamp=ts, actual=actual, predicted=predicted, group=group_str))
        return records

    def _parse_timestamp(self, value: object, default: datetime) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, str):
            try:
                converted = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return default
            if converted.tzinfo is None:
                converted = converted.replace(tzinfo=timezone.utc)
            return converted
        return default

    def _compute_hit_rates(self) -> dict[str, float]:
        if not self.records:
            return {}
        rates: dict[str, float] = {}
        counts: dict[str, tuple[int, int]] = {}
        for record in self.records:
            hit, total = counts.get(record.variant, (0, 0))
            counts[record.variant] = (hit + int(record.actual), total + 1)
        for variant, (hits, total) in counts.items():
            # Add Laplace smoothing to avoid zero probabilities.
            rates[variant] = (hits + 1.0) / (total + 2.0)
        return rates

    def _compute_confusion_matrix(self) -> dict[str, int]:
        matrix = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for record in self.records:
            if record.predicted and record.actual:
                matrix["tp"] += 1
            elif record.predicted and not record.actual:
                matrix["fp"] += 1
            elif (not record.predicted) and record.actual:
                matrix["fn"] += 1
            else:
                matrix["tn"] += 1
        return matrix

    def _baseline_rate(self) -> float:
        if not self.records:
            return 0.5
        total_hits = sum(int(record.actual) for record in self.records)
        return (total_hits + 1.0) / (len(self.records) + 2.0)
