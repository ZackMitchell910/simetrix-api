from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Optional

try:
    from prometheus_client import Gauge

    _QUEUE_DEPTH = Gauge(
        "simetrix_task_queue_depth",
        "Number of pending tasks in the async dispatcher",
        ["queue"],
    )
except Exception:  # pragma: no cover - metrics optional
    _QUEUE_DEPTH = None  # type: ignore

logger = logging.getLogger(__name__)

CoroutineFactory = Callable[[], Awaitable[None]]


class TaskDispatcher:
    """
    Minimal async task dispatcher that limits concurrency and keeps metrics.
    Submit coroutine factories; the dispatcher awaits them on a worker pool.
    """

    def __init__(self, name: str, concurrency: int = 1):
        self.name = name
        self._concurrency = max(1, int(concurrency))
        self._queue: asyncio.Queue[Optional[CoroutineFactory]] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        self._started = False
        self._stopping = False

    async def start(self) -> None:
        if self._started:
            return
        self._started = True
        for idx in range(self._concurrency):
            self._workers.append(asyncio.create_task(self._worker(idx)))

    async def _worker(self, idx: int) -> None:
        while True:
            item = await self._queue.get()
            self._set_depth()
            if item is None:
                self._queue.task_done()
                break
            try:
                await item()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - worker errors logged
                logger.exception("Task dispatcher %s worker %d failed: %s", self.name, idx, exc)
            finally:
                self._queue.task_done()

    def submit(self, factory: CoroutineFactory) -> None:
        if not self._started or self._stopping:
            raise RuntimeError("Dispatcher must be started before submitting tasks")
        self._queue.put_nowait(factory)
        self._set_depth()

    async def stop(self) -> None:
        if not self._started or self._stopping:
            return
        self._stopping = True
        for _ in self._workers:
            await self._queue.put(None)
        await self._queue.join()
        for worker in self._workers:
            worker.cancel()
        self._set_depth(value=0)

    def _set_depth(self, value: Optional[int] = None) -> None:
        if _QUEUE_DEPTH is None:
            return
        depth = value if value is not None else self._queue.qsize()
        try:
            _QUEUE_DEPTH.labels(self.name).set(depth)
        except Exception:  # pragma: no cover - metrics optional
            pass
