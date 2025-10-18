from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Iterable, List

from src.services.feature_store import feature_store_cache

try:  # pragma: no cover
    from src.feature_store import connect as feature_store_connect  # type: ignore
except Exception:  # pragma: no cover
    feature_store_connect = None  # type: ignore

from .base import AdapterRow, normalize_asof


def _rows_from_payload(payload: Iterable[dict[str, Any]]) -> List[AdapterRow]:
    rows: List[AdapterRow] = []
    for item in payload:
        try:
            asof = datetime.fromisoformat(str(item.get("asof")))
        except Exception:
            asof = normalize_asof(None)
        rows.append(
            AdapterRow(
                symbol=str(item.get("symbol", "")).upper(),
                asof=normalize_asof(asof),
                source=str(item.get("source") or "futures"),
                confidence=float(item.get("confidence", 0.5)),
                payload={k: v for k, v in item.items() if k not in {"symbol", "asof", "source", "confidence"}},
            )
        )
    return rows


async def fetch(symbol: str, asof: datetime | None = None, window: int = 10) -> List[AdapterRow]:
    symbol_norm = (symbol or "").strip().upper()
    if not symbol_norm:
        return []

    asof_norm = normalize_asof(asof)
    start = asof_norm - timedelta(days=max(1, int(window)))

    async def loader() -> List[dict[str, Any]]:
        if not callable(feature_store_connect):
            return []
        try:
            con = feature_store_connect()
        except Exception:
            return []

        queries = [
            (
                """
                SELECT as_of, provider, basis, carry, confidence
                FROM futures_basis
                WHERE symbol = ? AND as_of BETWEEN ? AND ?
                ORDER BY as_of DESC
                LIMIT 60
                """,
                [symbol_norm, start.date(), asof_norm.date()],
            ),
            (
                """
                SELECT collected_at, provider, implied_drift, carry
                FROM futures_quotes
                WHERE symbol = ? AND collected_at BETWEEN ? AND ?
                ORDER BY collected_at DESC
                LIMIT 60
                """,
                [symbol_norm, start, asof_norm],
            ),
        ]

        rows: List[dict[str, Any]] = []
        for sql, params in queries:
            try:
                data = con.execute(sql, params).fetchall()
            except Exception:
                continue
            for record in data:
                ts_val = record[0]
                src_val = record[1] if len(record) > 1 else "futures"
                basis_val = record[2] if len(record) > 2 else None
                carry_val = record[3] if len(record) > 3 else None
                conf_val = record[4] if len(record) > 4 else None
                payload = {
                    "basis": float(basis_val) if basis_val is not None else None,
                    "carry": float(carry_val) if carry_val is not None else None,
                }
                confidence = float(conf_val) if isinstance(conf_val, (int, float)) else 0.5
                if isinstance(ts_val, str):
                    try:
                        ts_val = datetime.fromisoformat(ts_val)
                    except Exception:
                        ts_val = asof_norm
                rows.append(
                    AdapterRow(
                        symbol=symbol_norm,
                        asof=normalize_asof(ts_val if isinstance(ts_val, datetime) else asof_norm),
                        source=str(src_val or "futures"),
                        confidence=confidence,
                        payload=payload,
                    ).as_dict()
                )
            if rows:
                break

        try:
            con.close()
        except Exception:
            pass

        return rows

    cached = await feature_store_cache.fetch(
        key=f"futures:{symbol_norm}:{asof_norm.date().isoformat()}:{int(window)}",
        loader=loader,
        ttl=3600,
        namespace="futures",
    )
    return _rows_from_payload(cached if isinstance(cached, list) else [])


__all__ = ["fetch"]
