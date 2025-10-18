"""Shared primitives for feature adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence


@dataclass(slots=True)
class FeedRecord:
    symbol: str
    asof: datetime
    source: str
    confidence: float
    payload: Dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "asof": self.asof.isoformat(),
            "source": self.source,
            "confidence": float(self.confidence),
            "payload": dict(self.payload),
            "tags": list(self.tags),
        }


class FeedFrame(Sequence[FeedRecord]):
    """In-memory container with helper aggregations."""

    def __init__(self, rows: Iterable[FeedRecord] | None = None) -> None:
        self._rows: List[FeedRecord] = list(rows or [])

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, item: int) -> FeedRecord:
        return self._rows[item]

    def __iter__(self) -> Iterator[FeedRecord]:
        return iter(self._rows)

    def append(self, record: FeedRecord) -> None:
        self._rows.append(record)

    def extend(self, records: Iterable[FeedRecord]) -> None:
        for record in records:
            self.append(record)

    def weighted_mean(self, key: str, *, default: float = 0.0, weight_key: str = "confidence") -> float:
        total = 0.0
        weight = 0.0
        for row in self._rows:
            value = row.payload.get(key)
            if value is None:
                continue
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                continue
            w = getattr(row, weight_key, None)
            try:
                w_f = float(w) if w is not None else 1.0
            except (TypeError, ValueError):
                w_f = 1.0
            total += value_f * w_f
            weight += w_f
        if weight <= 0:
            return default
        return total / weight

    def latest(self) -> Optional[FeedRecord]:
        if not self._rows:
            return None
        return max(self._rows, key=lambda r: r.asof)

    @classmethod
    def from_dicts(cls, rows: Iterable[Mapping[str, Any]]) -> "FeedFrame":
        parsed: List[FeedRecord] = []
        for row in rows:
            asof_value = row.get("asof")
            if isinstance(asof_value, str):
                asof = datetime.fromisoformat(asof_value)
            elif isinstance(asof_value, datetime):
                asof = asof_value
            else:  # pragma: no cover - defensive
                continue
            parsed.append(
                FeedRecord(
                    symbol=str(row.get("symbol", "")),
                    asof=asof,
                    source=str(row.get("source", "unknown")),
                    confidence=float(row.get("confidence", 0.0)),
                    payload=dict(row.get("payload", {})),
                    tags=tuple(row.get("tags", ())),
                )
            )
        return cls(parsed)

    def to_pandas(self):  # pragma: no cover - optional dependency
        try:
            import pandas as pd
        except Exception as exc:  # pragma: no cover - import may fail on slim env
            raise RuntimeError("pandas is required to materialize FeedFrame") from exc
        return pd.DataFrame([row.to_dict() for row in self._rows])


__all__ = ["FeedFrame", "FeedRecord"]
