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
                source=str(item.get("source") or "social"),
                confidence=float(item.get("confidence", 0.5)),
                payload={k: v for k, v in item.items() if k not in {"symbol", "asof", "source", "confidence"}},
            )
        )
    return rows


async def fetch(symbol: str, asof: datetime | None = None, window: int = 7) -> List[AdapterRow]:
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
        try:
            rows = con.execute(
                """
                SELECT collected_at, provider, score, volume, confidence
                FROM social_sentiment
                WHERE symbol = ? AND collected_at BETWEEN ? AND ?
                ORDER BY collected_at DESC
                LIMIT 90
                """,
                [symbol_norm, start, asof_norm],
            ).fetchall()
        except Exception:
            rows = []
        finally:
            try:
                con.close()
            except Exception:
                pass

        payload: List[dict[str, Any]] = []
        for ts_val, provider, score, volume, conf in rows:
            if isinstance(ts_val, str):
                try:
                    ts_val = datetime.fromisoformat(ts_val)
                except Exception:
                    ts_val = asof_norm
            payload.append(
                AdapterRow(
                    symbol=symbol_norm,
                    asof=normalize_asof(ts_val if isinstance(ts_val, datetime) else asof_norm),
                    source=str(provider or "social"),
                    confidence=float(conf) if isinstance(conf, (int, float)) else 0.5,
                    payload={
                        "score": float(score) if isinstance(score, (int, float)) else score,
                        "volume": float(volume) if isinstance(volume, (int, float)) else volume,
                    },
                ).as_dict()
            )
        return payload

    cached = await feature_store_cache.fetch(
        key=f"social:{symbol_norm}:{asof_norm.date().isoformat()}:{int(window)}",
        loader=loader,
        ttl=7200,
        namespace="social",
    )
    return _rows_from_payload(cached if isinstance(cached, list) else [])


__all__ = ["fetch"]
