from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from src.db.duck import get_conn

__all__ = [
    "register_model_version",
    "promote_model_version",
    "get_active_model_version",
    "list_model_versions",
]

_VALID_STATUSES = {"pending", "active", "inactive", "archived"}


def _norm_group(model_group: str) -> str:
    value = (model_group or "").strip().lower()
    if not value:
        raise ValueError("model_group is required")
    return value


def _norm_symbol(symbol: str) -> str:
    value = (symbol or "").strip().upper()
    if not value:
        raise ValueError("symbol is required")
    return value


def _norm_version(version: str) -> str:
    value = (version or "").strip()
    if not value:
        raise ValueError("version is required")
    return value


def _norm_status(status: str) -> str:
    value = (status or "pending").strip().lower()
    if value not in _VALID_STATUSES:
        raise ValueError(f"status must be one of {_VALID_STATUSES!r}, got {status!r}")
    return value


def _dump_metadata(metadata: Optional[Mapping[str, Any]]) -> str:
    if metadata is None:
        return "{}"
    return json.dumps(metadata)


def _load_metadata(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        except Exception:
            return {"raw": value}
    if isinstance(value, dict):
        return value
    return {"value": value}


def promote_model_version(model_group: str, symbol: str, version: str) -> None:
    group = _norm_group(model_group)
    sym = _norm_symbol(symbol)
    ver = _norm_version(version)

    con = get_conn()
    con.execute("BEGIN TRANSACTION;")
    try:
        con.execute(
            """
            UPDATE model_registry
            SET status = 'inactive'
            WHERE model_group = ? AND symbol = ? AND status = 'active'
            """,
            [group, sym],
        )
        cur = con.execute(
            """
            UPDATE model_registry
            SET status = 'active', promoted_at = ?
            WHERE model_group = ? AND symbol = ? AND version = ?
            """,
            [datetime.now(timezone.utc), group, sym, ver],
        )
        if cur.rowcount == 0:
            raise ValueError(f"version {ver!r} not found for {group}:{sym}")
        con.execute("COMMIT;")
    except Exception:
        con.execute("ROLLBACK;")
        raise


def register_model_version(
    model_group: str,
    symbol: str,
    version: str,
    artifact_path: str,
    artifact_type: str,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    status: str = "pending",
) -> str:
    group = _norm_group(model_group)
    sym = _norm_symbol(symbol)
    ver = _norm_version(version)
    status_norm = _norm_status(status)

    artifact_path = (artifact_path or "").strip()
    if not artifact_path:
        raise ValueError("artifact_path is required")
    artifact_type = (artifact_type or "").strip().lower()
    if not artifact_type:
        raise ValueError("artifact_type is required")

    con = get_conn()
    created_at = datetime.now(timezone.utc)
    meta_json = _dump_metadata(metadata)

    con.execute(
        """
        INSERT INTO model_registry (model_group, symbol, version, artifact_path, artifact_type, metadata, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(model_group, symbol, version) DO UPDATE
        SET artifact_path = excluded.artifact_path,
            artifact_type = excluded.artifact_type,
            metadata = excluded.metadata,
            status = excluded.status
        """,
        [group, sym, ver, artifact_path, artifact_type, meta_json, status_norm, created_at],
    )

    if status_norm == "active":
        promote_model_version(group, sym, ver)

    return ver


def get_active_model_version(model_group: str, symbol: str) -> Optional[dict[str, Any]]:
    group = _norm_group(model_group)
    sym = _norm_symbol(symbol)
    con = get_conn()
    row = con.execute(
        """
        SELECT version,
               artifact_path,
               artifact_type,
               metadata,
               promoted_at,
               created_at,
               status
        FROM model_registry
        WHERE model_group = ? AND symbol = ? AND status = 'active'
        ORDER BY promoted_at DESC NULLS LAST, created_at DESC
        LIMIT 1
        """,
        [group, sym],
    ).fetchone()
    if not row:
        return None
    version, artifact_path, artifact_type, metadata, promoted_at, created_at, status = row
    return {
        "model_group": group,
        "symbol": sym,
        "version": version,
        "artifact_path": artifact_path,
        "artifact_type": artifact_type,
        "metadata": _load_metadata(metadata),
        "promoted_at": promoted_at,
        "created_at": created_at,
        "status": status,
    }


def list_model_versions(model_group: str, symbol: str, limit: int = 20) -> list[dict[str, Any]]:
    group = _norm_group(model_group)
    sym = _norm_symbol(symbol)
    limit = max(1, int(limit))
    con = get_conn()
    rows = con.execute(
        """
        SELECT version,
               artifact_path,
               artifact_type,
               metadata,
               status,
               created_at,
               promoted_at
        FROM model_registry
        WHERE model_group = ? AND symbol = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        [group, sym, limit],
    ).fetchall()
    return [
        {
            "model_group": group,
            "symbol": sym,
            "version": row[0],
            "artifact_path": row[1],
            "artifact_type": row[2],
            "metadata": _load_metadata(row[3]),
            "status": row[4],
            "created_at": row[5],
            "promoted_at": row[6],
        }
        for row in rows
    ]
