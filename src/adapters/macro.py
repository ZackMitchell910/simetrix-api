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
                symbol=str(item.get("symbol", "MACRO")),
                asof=normalize_asof(asof),
                source=str(item.get("source") or "macro"),
                confidence=float(item.get("confidence", 0.5)),
                payload={k: v for k, v in item.items() if k not in {"symbol", "asof", "source", "confidence"}},
            )
        )
    return rows


async def fetch(symbol: str, asof: datetime | None = None, window: int = 30) -> List[AdapterRow]:
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
                SELECT as_of, source, rff, cpi_yoy, u_rate
                FROM macro_daily
                WHERE as_of BETWEEN ? AND ?
                ORDER BY as_of DESC
                LIMIT 90
                """,
                [start.date(), asof_norm.date()],
            ).fetchall()
        except Exception:
            rows = []
        finally:
            try:
                con.close()
            except Exception:
                pass

        payload: List[dict[str, Any]] = []
        for ts_val, src_val, rff, cpi_yoy, u_rate in rows:
            if isinstance(ts_val, str):
                try:
                    ts_val = datetime.fromisoformat(ts_val)
                except Exception:
                    ts_val = asof_norm
            payload.append(
                AdapterRow(
                    symbol="MACRO",
                    asof=normalize_asof(ts_val if isinstance(ts_val, datetime) else asof_norm),
                    source=str(src_val or "macro"),
                    confidence=0.7,
                    payload={"rff": rff, "cpi_yoy": cpi_yoy, "u_rate": u_rate},
                ).as_dict()
            )
        return payload

    cached = await feature_store_cache.fetch(
        key=f"macro:{asof_norm.date().isoformat()}:{int(window)}",
        loader=loader,
        ttl=6 * 3600,
        namespace="macro",
    )
    return _rows_from_payload(cached if isinstance(cached, list) else [])


__all__ = ["fetch"]
