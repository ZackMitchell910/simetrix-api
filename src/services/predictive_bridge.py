from __future__ import annotations

from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


def _load_module():
    import sys
    if "src.predictive_service" in sys.modules:
        return sys.modules["src.predictive_service"]
    from importlib import import_module
    return import_module("src.predictive_service")


def get_outcomes_label() -> Optional[Callable[..., Any]]:
    module = _load_module()
    try:
        return module.outcomes_label  # type: ignore[attr-defined]
    except AttributeError:
        return None


def get_learn_online() -> Optional[Callable[..., Any]]:
    module = _load_module()
    try:
        return module.learn_online  # type: ignore[attr-defined]
    except AttributeError:
        return None


def get_online_learn_request() -> Optional[type]:
    module = _load_module()
    try:
        return module.OnlineLearnRequest  # type: ignore[attr-defined]
    except AttributeError:
        return None
