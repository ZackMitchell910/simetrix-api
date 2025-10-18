from __future__ import annotations

import asyncio
import inspect
import pickle
import time
from collections import defaultdict
from typing import Any, Awaitable, Callable, Mapping

from redis.asyncio import Redis

from src.core import REDIS
from src.observability import log_json

CacheLoader = Callable[[], Awaitable[Any] | Any]


class FeatureStore:
    """Thin async cache wrapper backed by Redis with an in-process fallback."""

    def __init__(self, redis: Redis | None = None) -> None:
        self._redis = redis or REDIS
        self._local: dict[str, tuple[float, Any]] = {}
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._lock = asyncio.Lock()

    async def get_or_load(
        self,
        namespace: str,
        key: str,
        loader: CacheLoader,
        *,
        ttl: int = 900,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> Any:
        composite = f"feature:{namespace}:{key}"
        start = time.perf_counter()
        cached = await self._read(composite)
        cold = cached is None

        if cold:
            lock = await self._lock_for(composite)
            async with lock:
                cached = await self._read(composite)
                if cached is None:
                    result = loader()
                    if inspect.isawaitable(result):
                        result = await result  # type: ignore[assignment]
                    await self._write(composite, result, ttl)
                    cached = result

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        log_json(
            "info",
            msg="feature_store_cache",
            namespace=namespace,
            key=key,
            hit=not cold,
            duration_ms=round(elapsed_ms, 3),
            backend="redis" if self._redis else "memory",
            diagnostics=diagnostics or {},
        )
        return cached

    async def _lock_for(self, key: str) -> asyncio.Lock:
        async with self._lock:
            return self._locks[key]

    async def _read(self, composite: str) -> Any:
        now = time.monotonic()

        if self._redis:
            try:
                raw = await self._redis.get(composite)
            except Exception:
                raw = None
            if raw:
                try:
                    return pickle.loads(raw)
                except Exception:
                    pass

        expiry, payload = self._local.get(composite, (0.0, None))
        if expiry and expiry > now:
            return payload
        if composite in self._local:
            self._local.pop(composite, None)
        return None

    async def _write(self, composite: str, value: Any, ttl: int) -> None:
        expires_at = time.monotonic() + float(max(1, ttl))
        self._local[composite] = (expires_at, value)

        if not self._redis:
            return
        try:
            payload = pickle.dumps(value)
            await self._redis.setex(composite, ttl, payload)
        except Exception:
            # Redis write failures should not break callers; they simply fall back
            # to the in-process cache.
            pass


FEATURE_STORE = FeatureStore()

__all__ = ["FeatureStore", "FEATURE_STORE"]
