"""Configuration objects for active temporal sampling experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ExperimentCondition:
    name: str
    volatility: float
    stochasticity: float

    def __post_init__(self) -> None:
        """Require valid stationary variances for the current Gaussian environment."""
        if (
            not np.isfinite([self.volatility, self.stochasticity]).all()
            or min(self.volatility, self.stochasticity) <= 0
        ):
            raise ValueError(
                "Condition variances must be finite and strictly positive."
            )


@dataclass(frozen=True)
class GridConfig:
    size: int = 25
    volatility_min: float = 0.08
    volatility_max: float = 8.0
    stochasticity_min: float = 0.25
    stochasticity_max: float = 18.0

    def __post_init__(self) -> None:
        """Reject grids that cannot define a finite positive-variance posterior."""
        if self.size < 2 or int(self.size) != self.size:
            raise ValueError("Grid size must be an integer of at least two.")
        for low, high in (
            (self.volatility_min, self.volatility_max),
            (self.stochasticity_min, self.stochasticity_max),
        ):
            if not np.isfinite([low, high]).all() or not 0 < low < high:
                raise ValueError(
                    "Grid bounds must be finite and satisfy 0 < minimum < maximum."
                )

    def make_grids(self) -> tuple[np.ndarray, np.ndarray]:
        volatility = np.geomspace(self.volatility_min, self.volatility_max, self.size)
        stochasticity = np.geomspace(
            self.stochasticity_min, self.stochasticity_max, self.size
        )
        return volatility, stochasticity


@dataclass(frozen=True)
class AgentConfig:
    action_space: tuple[int, ...] = (1, 2, 4, 8)
    wait_cost: float = 0.05
    observation_rate_cost: float = 0.08
    eig_samples: int = 128
    calibration_cycles: int = 0
    information_objective: str = "entropy"
    initial_variance: float = 12.0
    cost_mode: str = "per_observation"

    def __post_init__(self) -> None:
        """Validate interval, acquisition, and cost settings."""
        if not self.action_space or any(
            int(dt) != dt or dt <= 0 for dt in self.action_space
        ):
            raise ValueError("action_space must contain positive integer intervals.")
        if len(set(self.action_space)) != len(self.action_space):
            raise ValueError("action_space must not contain duplicates.")
        if self.eig_samples < 1 or self.calibration_cycles < 0:
            raise ValueError(
                "eig_samples must be positive and calibration_cycles nonnegative."
            )
        if self.information_objective not in {"entropy", "log_ratio"}:
            raise ValueError("Unknown information objective.")
        if self.cost_mode not in {"per_observation", "legacy_rate"}:
            raise ValueError("cost_mode must be per_observation or legacy_rate.")
        if not np.isfinite(
            [self.wait_cost, self.observation_rate_cost, self.initial_variance]
        ).all():
            raise ValueError("Costs and initial variance must be finite.")
        if min(self.wait_cost, self.observation_rate_cost, self.initial_variance) < 0:
            raise ValueError("Costs and initial variance must be nonnegative.")


@dataclass(frozen=True)
class SimulationConfig:
    run_name: str = "main"
    output_root: Path = Path("outputs")
    seed: int = 20260521
    episodes: int = 96
    steps: int = 80
    grid: GridConfig = GridConfig()
    agent: AgentConfig = AgentConfig()
    include_ablation: bool = True
    ablation_episodes: int = 24
    ablation_wait_costs: tuple[float, ...] = (0.0, 0.01, 0.025, 0.05, 0.1)
    include_model_comparison: bool = True
    model_comparison_episodes: int = 8
    include_competitive_comparison: bool = True
    competitive_episodes: int = 8
    time_budget: int | None = None

    def __post_init__(self) -> None:
        """Validate episode budgets before creating run artifacts."""
        if (
            min(
                self.episodes,
                self.steps,
                self.ablation_episodes,
                self.model_comparison_episodes,
                self.competitive_episodes,
            )
            < 1
        ):
            raise ValueError("Episode counts and steps must be positive.")
        if self.time_budget is not None and (
            self.time_budget < 1 or int(self.time_budget) != self.time_budget
        ):
            raise ValueError("time_budget must be a positive integer.")
        if self.time_budget is not None and min(self.agent.action_space) != 1:
            raise ValueError(
                "A time budget requires interval 1 for exact boundary handling."
            )

    def output_dir(self) -> Path:
        return self.output_root / self.run_name

    def with_run_name(self, run_name: str) -> "SimulationConfig":
        return replace(self, run_name=run_name)


def default_conditions() -> tuple[ExperimentCondition, ...]:
    return (
        ExperimentCondition("low_v_low_s", volatility=0.25, stochasticity=1.0),
        ExperimentCondition("low_v_high_s", volatility=0.25, stochasticity=9.0),
        ExperimentCondition("high_v_low_s", volatility=4.0, stochasticity=1.0),
        ExperimentCondition("high_v_high_s", volatility=4.0, stochasticity=9.0),
    )


def smoke_config(run_name: str = "smoke") -> SimulationConfig:
    return SimulationConfig(
        run_name=run_name,
        episodes=8,
        steps=40,
        include_ablation=True,
        ablation_episodes=4,
        model_comparison_episodes=2,
        competitive_episodes=2,
    )


def main_config(run_name: str = "main") -> SimulationConfig:
    return SimulationConfig(run_name=run_name)
