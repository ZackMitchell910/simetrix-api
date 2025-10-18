"""Correlated multi-asset simulation utilities."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from src.scenarios.models import EventShock

from .base import Artifact, PathEngine, StateVector
from .jd import JumpDiffusionEngine
from .iv_anchor import IVAnchor, TRADING_DAYS


class CorrelatedPathEngine(PathEngine):
    """Generate correlated price paths across several assets."""

    def __init__(self, *, base_engine: PathEngine | None = None, scheduler=None) -> None:
        super().__init__(scheduler=scheduler)
        self.base_engine = base_engine or JumpDiffusionEngine(scheduler=scheduler)

    def simulate(
        self,
        state: StateVector,
        scenarios: Sequence[EventShock],
        horizon_days: int,
        n_paths: int,
        dt: str = "1d",
        *,
        random_state: np.random.Generator | None = None,
    ) -> Artifact:
        components: Sequence[StateVector] = state.metadata.get("components") or [state]
        if len(components) == 1:
            return self.base_engine.simulate(
                state=components[0],
                scenarios=[shock for shock in scenarios if shock.symbol == components[0].symbol],
                horizon_days=horizon_days,
                n_paths=n_paths,
                dt=dt,
                random_state=random_state,
            )

        step_days = self._parse_dt(dt)
        time_grid = self._time_grid(state.asof, horizon_days, step_days)
        scheduler = self._ensure_scheduler()

        scenario_map: dict[str, list[EventShock]] = defaultdict(list)
        for shock in scenarios:
            scenario_map[shock.symbol].append(shock)

        compiled = []
        sigma_paths = []
        anchor_scales = []
        for component in components:
            comp_scenarios = scenario_map.get(component.symbol, [])
            adjustments = scheduler.compile(comp_scenarios, time_grid)
            compiled.append(adjustments)
            template_sigma = adjustments.effective_sigma(component.sigma)
            anchor = IVAnchor(component.iv_surface)
            scale = anchor.variance_scale(template_sigma, step_days, horizon_days)
            sigma_paths.append(template_sigma * scale)
            anchor_scales.append(scale)

        correlation = state.correlation
        n_assets = len(components)
        if correlation is None:
            corr = np.eye(n_assets)
        else:
            corr = np.asarray(correlation, dtype=float)
            if corr.shape != (n_assets, n_assets):
                raise ValueError("Correlation matrix shape does not match component count")
        chol = np.linalg.cholesky(corr)

        n_steps = len(time_grid) - 1
        rng = random_state or np.random.default_rng()
        joint_paths = np.empty((n_assets, n_paths, len(time_grid)), dtype=float)
        for idx, component in enumerate(components):
            joint_paths[idx, :, 0] = component.spot

        dt_years = step_days / TRADING_DAYS
        sqrt_dt = np.sqrt(dt_years)

        for step in range(n_steps):
            normals = rng.standard_normal((n_assets, n_paths))
            correlated_normals = chol @ normals
            for asset_idx, component in enumerate(components):
                adjustments = compiled[asset_idx]
                sigma = sigma_paths[asset_idx][step]
                diffusion = sigma * sqrt_dt * correlated_normals[asset_idx]
                drift = component.annualised_drift() + adjustments.drift[step]
                jump_lambda = max(adjustments.jump_intensity[step], 0.0)
                jump_mu = adjustments.jump_mean[step]
                jump_sigma = adjustments.jump_std[step]
                jump_component = np.zeros(n_paths, dtype=float)
                if jump_lambda > 0:
                    poisson = rng.poisson(jump_lambda * dt_years, size=n_paths)
                    mask = poisson > 0
                    if np.any(mask):
                        mean = jump_mu * poisson[mask]
                        std = jump_sigma * np.sqrt(poisson[mask])
                        if np.any(std > 0):
                            jump_component[mask] = rng.normal(mean, std)
                        else:
                            jump_component[mask] = mean
                drift_term = (drift - 0.5 * sigma ** 2) * dt_years
                joint_paths[asset_idx, :, step + 1] = joint_paths[asset_idx, :, step] * np.exp(
                    drift_term + diffusion + jump_component
                )

        primary_paths = joint_paths[0]
        metadata = {
            "components": {
                component.symbol: joint_paths[idx]
                for idx, component in enumerate(components)
            },
            "correlation": corr,
            "dt": dt,
            "anchor_scales": anchor_scales,
            "scheduler": [adj.metadata for adj in compiled],
        }
        return Artifact(paths=primary_paths, time_grid=time_grid, metadata=metadata)


__all__ = ["CorrelatedPathEngine"]
