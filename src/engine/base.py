from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from src.scenarios.models import EventShock
from .types import Artifact, StateVector


class PathEngine(ABC):
    """Interface for pluggable Monte Carlo engines."""

    @abstractmethod
    def simulate(
        self,
        state: StateVector,
        scenarios: Sequence[EventShock],
        horizon_days: int,
        n_paths: int,
        dt: str = "1d",
    ) -> Artifact:
        """Simulate Monte Carlo paths subject to optional scenarios."""
        raise NotImplementedError
