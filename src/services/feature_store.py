from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, MutableMapping, Optional

from src.core import DATA_ROOT, REDIS

RedisType = Any  # redis.asyncio.Redis but kept generic to avoid optional import issues


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover - defensive
            pass
    raise TypeError(f"Value of type {type(value)!r} is not JSON serializable")


class FeatureStoreCache:
    def __init__(self, redis: Optional[RedisType] = None, sqlite_path: Optional[Path] = None) -> None:
        self._redis = redis
        self._sqlite_path = sqlite_path or (Path(DATA_ROOT) / "feature_store_cache.sqlite")
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_sqlite()

    def _ensure_sqlite(self) -> None:
        con = sqlite3.connect(self._sqlite_path)
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    PRIMARY KEY (namespace, key)
                )
                """
            )
            con.commit()
        finally:
            con.close()

    async def _sqlite_get(self, namespace: str, key: str, now: float) -> Any | None:
        def _get() -> Any | None:
            with closing(sqlite3.connect(self._sqlite_path)) as con:
                row = con.execute(
                    "SELECT value, expires_at FROM cache WHERE namespace = ? AND key = ?",
                    [namespace, key],
                ).fetchone()
                if not row:
                    return None
                value, expires_at = row
                if expires_at < now:
                    con.execute("DELETE FROM cache WHERE namespace = ? AND key = ?", [namespace, key])
                    con.commit()
                    return None
                return json.loads(value)

        return await asyncio.to_thread(_get)

    async def _sqlite_set(self, namespace: str, key: str, value: Any, expires_at: float) -> None:
        payload = json.dumps(value, default=_json_default)

        def _set() -> None:
            with closing(sqlite3.connect(self._sqlite_path)) as con:
                con.execute(
                    "REPLACE INTO cache(namespace, key, value, expires_at) VALUES (?,?,?,?)",
                    [namespace, key, payload, expires_at],
                )
                con.commit()

        await asyncio.to_thread(_set)

    def _redis_key(self, namespace: str, key: str) -> str:
        return f"fs:{namespace}:{key}"

    async def fetch(
        self,
        namespace: str,
        key: str,
        loader: Callable[[], Awaitable[Any] | Any],
        *,
        ttl: int = 3600,
        diagnostics: MutableMapping[str, Any] | None = None,
        default: Any | None = None,
    ) -> Any:
        now = time.time()
        redis = self._redis
        cache_key = self._redis_key(namespace, key)
        diag = diagnostics.setdefault("feature_store", {}) if diagnostics is not None else None

        if redis is not None:
            try:
                cached = await redis.get(cache_key)
            except Exception:
                cached = None
            if cached:
                try:
                    data = json.loads(cached)
                except Exception:
                    data = None
                if data is not None:
                    self._record_diag(diag, namespace, hit=True, source="redis", latency_ms=0.0)
                    return data

        sqlite_hit = await self._sqlite_get(namespace, key, now)
        if sqlite_hit is not None:
            self._record_diag(diag, namespace, hit=True, source="sqlite", latency_ms=0.0)
            if redis is not None:
                try:
                    await redis.setex(cache_key, ttl, json.dumps(sqlite_hit, default=_json_default))
                except Exception:
                    pass
            return sqlite_hit

        start = time.perf_counter()
        try:
            value = loader()
            if asyncio.iscoroutine(value):
                value = await value
        except Exception:
            self._record_diag(diag, namespace, hit=False, source="loader", latency_ms=(time.perf_counter() - start) * 1000.0, error=True)
            if default is not None:
                return default
            raise

        latency_ms = (time.perf_counter() - start) * 1000.0
        self._record_diag(diag, namespace, hit=False, source="loader", latency_ms=latency_ms)

        if value is None:
            return default

        expires_at = now + ttl
        await self._sqlite_set(namespace, key, value, expires_at)
        if redis is not None:
            try:
                await redis.setex(cache_key, ttl, json.dumps(value, default=_json_default))
            except Exception:
                pass
        return value

    @staticmethod
    def _record_diag(
        diagnostics: MutableMapping[str, Any] | None,
        namespace: str,
        *,
        hit: bool,
        source: str,
        latency_ms: float,
        error: bool = False,
    ) -> None:
        if diagnostics is None:
            return
        diagnostics.setdefault("hits", 0)
        diagnostics.setdefault("misses", 0)
        diagnostics.setdefault("namespaces", {})
        ns_diag = diagnostics["namespaces"].setdefault(namespace, {})

        if hit:
            diagnostics["hits"] += 1
            ns_diag["last_hit_source"] = source
            ns_diag["last_hit_ms"] = round(latency_ms, 3)
            last_miss = ns_diag.get("last_miss_ms")
            if last_miss:
                improvement = max(0.0, 1.0 - (ns_diag["last_hit_ms"] / last_miss))
                ns_diag["latency_improvement"] = round(improvement, 3)
        else:
            diagnostics["misses"] += 1
            ns_diag["last_miss_source"] = source
            ns_diag["last_miss_ms"] = round(latency_ms, 3)
            if error:
                ns_diag["error"] = True


FEATURE_STORE_CACHE = FeatureStoreCache(redis=REDIS)


async def cached_feature(
    namespace: str,
    key: str,
    loader: Callable[[], Awaitable[Any] | Any],
    *,
    ttl: int = 3600,
    diagnostics: MutableMapping[str, Any] | None = None,
    default: Any | None = None,
) -> Any:
    return await FEATURE_STORE_CACHE.fetch(
        namespace,
        key,
        loader,
        ttl=ttl,
        diagnostics=diagnostics,
        default=default,
    )


__all__ = ["FEATURE_STORE_CACHE", "cached_feature", "FeatureStoreCache"]
