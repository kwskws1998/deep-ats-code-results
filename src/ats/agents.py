"""Agent classes for fixed, random, passive, and active sampling policies."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ats.beliefs import GridBelief
from ats.config import AgentConfig
from ats.policies import choose_active_dt


@dataclass
class BaseAgent:
    name: str
    config: AgentConfig
    rng: np.random.Generator
    last_scores: dict[int, float] = field(default_factory=dict)

    def reset(self) -> None:
        self.last_scores = {}

    def choose_dt(self, belief: GridBelief, step_index: int) -> int:
        raise NotImplementedError


class FixedShortAgent(BaseAgent):
    def choose_dt(self, belief: GridBelief, step_index: int) -> int:
        self.last_scores = {}
        return min(self.config.action_space)


class FixedLongAgent(BaseAgent):
    def choose_dt(self, belief: GridBelief, step_index: int) -> int:
        self.last_scores = {}
        return max(self.config.action_space)


class RandomIntervalAgent(BaseAgent):
    def choose_dt(self, belief: GridBelief, step_index: int) -> int:
        self.last_scores = {}
        return int(self.rng.choice(self.config.action_space))


class PassiveGridAgent(BaseAgent):
    def choose_dt(self, belief: GridBelief, step_index: int) -> int:
        self.last_scores = {}
        return int(self.config.action_space[step_index % len(self.config.action_space)])


class ActiveTemporalSampler(BaseAgent):
    def choose_dt(self, belief: GridBelief, step_index: int) -> int:
        calibration_steps = (
            len(self.config.action_space) * self.config.calibration_cycles
        )
        if step_index < calibration_steps:
            self.last_scores = {}
            return int(
                self.config.action_space[step_index % len(self.config.action_space)]
            )

        dt, scores = choose_active_dt(
            belief=belief,
            action_space=self.config.action_space,
            wait_cost=self.config.wait_cost,
            observation_rate_cost=self.config.observation_rate_cost,
            eig_samples=self.config.eig_samples,
            information_objective=self.config.information_objective,
            rng=self.rng,
            cost_mode=self.config.cost_mode,
        )
        self.last_scores = scores
        return int(dt)


def make_agent(name: str, config: AgentConfig, seed: int) -> BaseAgent:
    rng = np.random.default_rng(seed)
    if name == "fixed_short":
        return FixedShortAgent(name=name, config=config, rng=rng)
    if name == "fixed_long":
        return FixedLongAgent(name=name, config=config, rng=rng)
    if name.startswith("fixed_dt_"):
        return FixedIntervalAgent(
            name=name,
            config=config,
            rng=rng,
            interval=int(name.removeprefix("fixed_dt_")),
        )
    if name == "random_interval":
        return RandomIntervalAgent(name=name, config=config, rng=rng)
    if name == "passive_grid":
        return PassiveGridAgent(name=name, config=config, rng=rng)
    if name == "active_temporal":
        return ActiveTemporalSampler(name=name, config=config, rng=rng)
    raise ValueError(f"Unknown agent name: {name}")


def default_agent_names() -> tuple[str, ...]:
    return (
        "fixed_short",
        "fixed_long",
        "fixed_dt_2",
        "fixed_dt_4",
        "random_interval",
        "passive_grid",
        "active_temporal",
    )


@dataclass
class FixedIntervalAgent(BaseAgent):
    """Prespecified intermediate-interval control with feasible boundary fallback."""

    interval: int = 2

    def choose_dt(self, belief: GridBelief, step_index: int) -> int:
        self.last_scores = {}
        feasible = [dt for dt in self.config.action_space if dt <= self.interval]
        return max(feasible) if feasible else min(self.config.action_space)
