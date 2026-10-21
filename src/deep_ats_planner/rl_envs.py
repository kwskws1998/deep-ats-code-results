"""RL task environments for Deep Active Temporal Sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deep_ats_planner.config import ExperimentCondition
from ats.envs import TemporalSamplingEnv


@dataclass(frozen=True)
class BanditStep:
    time: int
    dt: int
    action: int
    latent_state: float
    observation: float
    reward: float
    reward_probability: float
    process_increment: float
    observation_noise: float


class TemporalBanditEnv:
    """Two-action bandit whose latent reward preference follows ATS dynamics."""

    def __init__(
        self,
        condition: ExperimentCondition,
        rng: np.random.Generator,
        reward_scale: float = 1.4,
        max_time: int | None = None,
    ):
        self.condition = condition
        self.rng = rng
        self.reward_scale = float(reward_scale)
        self.time = 0
        self.state = 0.0
        self.max_time = max_time
        self.world = TemporalSamplingEnv(
            condition,
            np.random.default_rng(int(rng.integers(0, 2**63))),
            max_time=max_time,
        )
        self.reward_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
        self.reward_uniforms = None

    def reset(self, initial_state: float = 0.0) -> BanditStep:
        self.time = 0
        self.state = float(initial_state)
        initial = self.world.reset(initial_state)
        self.reward_uniforms = (
            None if self.max_time is None else self.reward_rng.random(self.max_time + 1)
        )
        observation_noise = initial.observation_noise
        return BanditStep(
            time=0,
            dt=0,
            action=-1,
            latent_state=self.state,
            observation=self.state + float(observation_noise),
            reward=0.0,
            reward_probability=0.0,
            process_increment=0.0,
            observation_noise=float(observation_noise),
        )

    def reward_probabilities(self, latent_state: float | None = None) -> np.ndarray:
        state = self.state if latent_state is None else float(latent_state)
        p_action_one = float(np.exp(-np.logaddexp(0.0, -self.reward_scale * state)))
        return np.array([1.0 - p_action_one, p_action_one], dtype=float)

    def step(self, action: int, dt: int) -> BanditStep:
        if action not in (0, 1):
            raise ValueError("TemporalBanditEnv currently supports actions 0 and 1.")
        if dt <= 0:
            raise ValueError("dt must be positive.")

        world_result = self.world.step(dt)
        self.state = world_result.latent_state
        self.time = world_result.time
        process_increment = world_result.process_increment
        observation_noise = world_result.observation_noise
        reward_probability = float(self.reward_probabilities()[int(action)])
        draw = (
            self.reward_rng.random()
            if self.reward_uniforms is None
            else self.reward_uniforms[self.time]
        )
        reward = float(draw < reward_probability)
        return BanditStep(
            time=self.time,
            dt=int(dt),
            action=int(action),
            latent_state=self.state,
            observation=self.state + float(observation_noise),
            reward=reward,
            reward_probability=reward_probability,
            process_increment=float(process_increment),
            observation_noise=float(observation_noise),
        )
