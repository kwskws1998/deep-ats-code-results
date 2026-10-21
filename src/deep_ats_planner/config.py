"""Shared ATS settings and configuration of the belief-conditioned DQN planner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ats.config import (
    AgentConfig,
    ExperimentCondition,
    GridConfig,
    SimulationConfig,
    default_conditions,
    main_config,
    smoke_config,
)


@dataclass(frozen=True)
class DeepPlannerConfig:
    run_name: str = "deep_ats_planner"
    output_root: Path = Path("outputs")
    seed: int = 20260521
    episodes: int = 64
    steps: int = 80
    grid: GridConfig = GridConfig()
    agent: AgentConfig = AgentConfig(wait_cost=0.02, observation_rate_cost=0.04)
    task_action_count: int = 2
    reward_scale: float = 1.4
    gamma: float = 0.96
    hidden_size: int = 64
    learning_rate: float = 3e-4
    batch_size: int = 64
    replay_capacity: int = 20_000
    warmup_steps: int = 128
    target_update_interval: int = 100
    epsilon_start: float = 0.35
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 3_000
    evaluation_episodes: int = 8
    discount_basis: str = "physical_time"
    use_belief_features: bool = True

    def __post_init__(self) -> None:
        """Validate the finite-horizon bandit and value-learning settings."""
        if self.task_action_count != 2:
            raise ValueError("The temporal bandit supports exactly two task actions.")
        if (
            min(
                self.episodes,
                self.steps,
                self.batch_size,
                self.hidden_size,
                self.target_update_interval,
                self.evaluation_episodes,
            )
            < 1
        ):
            raise ValueError("Training dimensions and episode counts must be positive.")
        if self.replay_capacity < self.batch_size or not 0 <= self.gamma <= 1:
            raise ValueError(
                "Replay capacity must cover a batch and gamma must lie in [0, 1]."
            )
        if self.discount_basis not in {"physical_time", "decision"}:
            raise ValueError("discount_basis must be physical_time or decision.")

    def output_dir(self) -> Path:
        return self.output_root / self.run_name


def deep_smoke_config(run_name: str = "deep_ats_smoke") -> DeepPlannerConfig:
    return DeepPlannerConfig(
        run_name=run_name,
        episodes=4,
        steps=24,
        grid=GridConfig(size=9),
        batch_size=16,
        replay_capacity=1_000,
        warmup_steps=16,
        target_update_interval=16,
        hidden_size=32,
        epsilon_decay_steps=200,
        evaluation_episodes=2,
    )


def deep_train_config(run_name: str = "deep_ats_planner") -> DeepPlannerConfig:
    return DeepPlannerConfig(run_name=run_name)
