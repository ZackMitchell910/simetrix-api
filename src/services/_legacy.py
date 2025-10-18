from __future__ import annotations

from typing import Any


def legacy_module():
    try:
        from src import predictive_api as legacy  # type: ignore
    except Exception:
        legacy = None
    return legacy


def legacy_attr(name: str, default: Any = None) -> Any:
    module = legacy_module()
    if module is None:
        return default
    return getattr(module, name, default)
