"""Helpers for Quant routes.

The public ``/quant/daily/today`` endpoint serves two distinct payloads: the
dashboard snapshot (equity/crypto picks, finalists, etc.) and the on-demand
minimal summary for a single ticker. Historically, the cached snapshot payload
could lack a handful of keys that the frontend now expects (``prob_up_end``,
``median_return_pct``, ``horizon_days`` and ``blurb``).  When stale data sneaks
in, the DailyQuant card falls back to placeholder copy and breaks the "Open in
Simulator" deep-link.

To make the route resilient, we sanitise every card-shaped document before it
is returned to the caller.  The sanitisation logic lives in
``src.services.quant_daily.normalize_pick_document`` so it can be re-used by the
prefill jobs and adapters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.services.quant_daily import normalize_pick_document

__all__ = ["ensure_daily_snapshot_payload"]


def _normalise_sequence(items: Any, fallback_horizon: int | None) -> list[dict[str, Any]]:
    if not isinstance(items, Sequence):
        return []
    normalised: list[dict[str, Any]] = []
    for entry in items:
        doc = normalize_pick_document(entry, fallback_horizon=fallback_horizon)
        if doc:
            normalised.append(doc)
    return normalised


def ensure_daily_snapshot_payload(
    payload: Mapping[str, Any] | None,
    *,
    fallback_horizon: int | None = None,
) -> dict[str, Any]:
    """Return a defensive copy of *payload* with pick documents normalised."""

    data: dict[str, Any] = dict(payload or {})
    data["equity"] = normalize_pick_document(data.get("equity"), fallback_horizon=fallback_horizon)
    data["crypto"] = normalize_pick_document(data.get("crypto"), fallback_horizon=fallback_horizon)

    for key in ("equity_top", "crypto_top", "equity_finalists", "crypto_finalists"):
        data[key] = _normalise_sequence(data.get(key), fallback_horizon)

    if fallback_horizon is not None and not data.get("horizon_days"):
        data["horizon_days"] = int(fallback_horizon)

    return data
