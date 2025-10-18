from __future__ import annotations

import inspect
import json
import time
from typing import Any, Awaitable, Callable, Dict, List

from redis.asyncio import Redis

from src.core import REDIS


Loader = Callable[[], Any | Awaitable[Any]]


class FeatureStoreCache:
    """Best-effort Redis + in-process cache with diagnostics for feature store queries."""

    def __init__(self, redis: Redis | None = None) -> None:
        self._redis = redis
        self._local: Dict[str, tuple[float, Any, float]] = {}
        self._diagnostics: List[Dict[str, Any]] = []

    async def fetch(
        self,
        *,
        key: str,
        loader: Loader,
        ttl: int = 600,
        namespace: str = "fs",
    ) -> Any:
        namespaced = f"fs:{namespace}:{key}"
        now = time.time()
        start = time.perf_counter()
        hit = False
        backend = "memory"
        stored_at = None
        value: Any | None = None

        entry = self._local.get(namespaced)
        if entry and entry[0] > now:
            value = entry[1]
            stored_at = entry[2]
            hit = True
            backend = "memory"

        if value is None and self._redis:
            try:
                raw = await self._redis.get(namespaced)
            except Exception:
                raw = None
            if raw:
                try:
                    payload = json.loads(raw)
                    expires = float(payload.get("expires", 0.0))
                    if expires > now:
                        value = payload.get("value")
                        stored_at = float(payload.get("stored_at", now))
                        hit = True
                        backend = "redis"
                except Exception:
                    value = None
            if value is not None:
                self._local[namespaced] = (now + ttl, value, stored_at or now)

        if value is None:
            result = loader()
            if inspect.isawaitable(result):
                result = await result
            stored_at = now
            value = result
            self._local[namespaced] = (now + ttl, value, stored_at)
            if self._redis:
                try:
                    await self._redis.setex(
                        namespaced,
                        int(ttl),
                        json.dumps({"value": value, "stored_at": stored_at, "expires": now + ttl}),
                    )
                except Exception:
                    pass
            backend = "loader"
        latency_ms = (time.perf_counter() - start) * 1000

        age_s = None if stored_at is None else max(0.0, now - stored_at)
        diag = {
            "key": key,
            "namespace": namespace,
            "hit": bool(hit),
            "backend": backend if hit or backend == "loader" else ("redis" if self._redis else "memory"),
            "latency_ms": round(latency_ms, 3),
            "age_s": round(age_s, 3) if age_s is not None else None,
            "ttl": int(ttl),
        }
        self._diagnostics.append(diag)
        return value

    def flush_diagnostics(self) -> List[Dict[str, Any]]:
        diag = self._diagnostics[:]
        self._diagnostics.clear()
        return diag


feature_store_cache = FeatureStoreCache(redis=REDIS)

__all__ = ["FeatureStoreCache", "feature_store_cache"]
