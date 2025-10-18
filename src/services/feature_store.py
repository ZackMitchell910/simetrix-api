"""Async feature store facade with Redis + SQLite caching."""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

from redis.asyncio import Redis

from src.core import REDIS


def _default_sqlite_path() -> Path:
    root = os.getenv("PT_DATA_ROOT")
    if root:
        base = Path(root).expanduser().resolve()
    else:
        base = Path(__file__).resolve().parents[2] / "data"
    base.mkdir(parents=True, exist_ok=True)
    return (base / "feature_store.sqlite").resolve()


class FeatureStore:
    """Hybrid caching layer used by adapters and model fusion."""

    def __init__(
        self,
        *,
        redis: Redis | None = None,
        sqlite_path: str | os.PathLike[str] | None = None,
        namespace: str = "feature_store",
        default_ttl: int = 6 * 3600,
    ) -> None:
        self.redis = redis or REDIS
        self.sqlite_path = Path(sqlite_path or _default_sqlite_path())
        self.namespace = namespace
        self.default_ttl = int(default_ttl)
        self._ensure_sqlite()

    # -- SQLite helpers -----------------------------------------------------------------

    def _ensure_sqlite(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    expires_at REAL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_meta (
                    key TEXT PRIMARY KEY,
                    cold_latency_ms REAL
                );
                """
            )
            conn.commit()

    def _sqlite_get(self, key: str) -> tuple[Any | None, float | None]:
        now = time.time()
        with sqlite3.connect(self.sqlite_path) as conn:
            row = conn.execute("SELECT value, expires_at FROM cache WHERE key=?", (key,)).fetchone()
            if not row:
                return None, None
            value_raw, expires_at = row
            if expires_at is not None and expires_at < now:
                conn.execute("DELETE FROM cache WHERE key=?", (key,))
                conn.commit()
                return None, None
            meta_row = conn.execute("SELECT cold_latency_ms FROM cache_meta WHERE key=?", (key,)).fetchone()
            latency = meta_row[0] if meta_row else None
            try:
                return json.loads(value_raw), latency
            except Exception:
                return None, latency

    def _sqlite_set(self, key: str, value: Any, ttl: int | None) -> None:
        expires = time.time() + float(ttl or self.default_ttl)
        payload = json.dumps(value, ensure_ascii=False)
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache(key, value, expires_at) VALUES(?,?,?)",
                (key, payload, expires),
            )
            conn.commit()

    def _sqlite_store_latency(self, key: str, latency_ms: float) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                "INSERT INTO cache_meta(key, cold_latency_ms) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET cold_latency_ms=excluded.cold_latency_ms",
                (key, float(latency_ms)),
            )
            conn.commit()

    # -- Core API -----------------------------------------------------------------------

    async def get(self, key: str) -> tuple[Any | None, dict[str, Any]]:
        namespaced = f"{self.namespace}:{key}"
        diag: dict[str, Any] = {"cache": {"hit": False, "layer": "miss"}}
        start = time.perf_counter()

        if self.redis:
            try:
                raw = await self.redis.get(namespaced)
            except Exception:
                raw = None
            if raw is not None:
                try:
                    value = json.loads(raw)
                except Exception:
                    value = None
                latency_ms = (time.perf_counter() - start) * 1000
                diag_payload = {"hit": True, "layer": "redis", "latency_ms": latency_ms}
                _, cold_latency = await asyncio.to_thread(self._sqlite_get, namespaced)
                if cold_latency is not None and cold_latency > 0:
                    reduction = max(0.0, 1.0 - (latency_ms / max(cold_latency, 1e-9))) * 100.0
                    diag_payload["savings_pct"] = round(reduction, 2)
                diag["cache"].update(diag_payload)
                return value, diag

        value_sql, cold_latency = await asyncio.to_thread(self._sqlite_get, namespaced)
        latency_ms = (time.perf_counter() - start) * 1000
        if value_sql is not None:
            diag["cache"].update(
                {
                    "hit": True,
                    "layer": "sqlite",
                    "latency_ms": latency_ms,
                }
            )
            if cold_latency is not None and cold_latency > 0:
                reduction = max(0.0, 1.0 - (latency_ms / max(cold_latency, 1e-9))) * 100.0
                diag["cache"]["savings_pct"] = round(reduction, 2)
            return value_sql, diag

        diag["cache"]["latency_ms"] = latency_ms
        return None, diag

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        namespaced = f"{self.namespace}:{key}"
        payload = json.dumps(value, ensure_ascii=False)
        ttl_use = int(ttl or self.default_ttl)
        if self.redis:
            try:
                await self.redis.setex(namespaced, ttl_use, payload)
            except Exception:
                pass
        await asyncio.to_thread(self._sqlite_set, namespaced, value, ttl_use)

    async def get_or_load(
        self,
        key: str,
        loader: Callable[[], Awaitable[Any] | Any],
        *,
        ttl: int | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        cached, diag = await self.get(key)
        if cached is not None:
            return cached, diag

        start = time.perf_counter()
        result = loader()
        if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
            result = await result  # type: ignore[assignment]
        latency_ms = (time.perf_counter() - start) * 1000
        diag["cache"].update({"hit": False, "latency_ms": latency_ms})
        await self.set(key, result, ttl=ttl)
        await asyncio.to_thread(self._sqlite_store_latency, f"{self.namespace}:{key}", latency_ms)
        return result, diag


__all__ = ["FeatureStore"]
