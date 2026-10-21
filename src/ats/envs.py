"""One-dimensional latent-state environment with process and observation noise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ats.config import ExperimentCondition


@dataclass(frozen=True)
class StepResult:
    time: int
    dt: int
    latent_state: float
    observation: float
    process_increment: float
    observation_noise: float
    latent_path: tuple[float, ...] = ()


class TemporalSamplingEnv:
    """Simulates observations from a random-walk latent state."""

    def __init__(
        self,
        condition: ExperimentCondition,
        rng: np.random.Generator,
        max_time: int | None = None,
    ):
        self.condition = condition
        self.rng = rng
        self.time = 0
        self.state = 0.0
        self.max_time = max_time
        if max_time is not None and (int(max_time) != max_time or max_time < 1):
            raise ValueError("max_time must be a positive integer.")
        self._increments = None
        self._noises = None
        self._latent_values = None

    def reset(self, initial_state: float = 0.0) -> StepResult:
        self.time = 0
        self.state = float(initial_state)
        if self.max_time is None:
            noise = self.rng.normal(0.0, np.sqrt(self.condition.stochasticity))
        else:
            process_rng = np.random.default_rng(int(self.rng.integers(0, 2**63)))
            observation_rng = np.random.default_rng(int(self.rng.integers(0, 2**63)))
            self._increments = process_rng.normal(
                0.0, np.sqrt(self.condition.volatility), size=self.max_time
            )
            self._latent_values = np.r_[
                self.state, self.state + np.cumsum(self._increments)
            ]
            self._noises = observation_rng.normal(
                0.0, np.sqrt(self.condition.stochasticity), size=self.max_time + 1
            )
            noise = self._noises[0]
        return StepResult(
            time=self.time,
            dt=0,
            latent_state=self.state,
            observation=self.state + noise,
            process_increment=0.0,
            observation_noise=float(noise),
        )

    def step(self, dt: int) -> StepResult:
        if dt <= 0 or int(dt) != dt:
            raise ValueError("dt must be a positive integer.")
        if self.max_time is not None and self.time + dt > self.max_time:
            raise ValueError("Step exceeds the prespecified physical trajectory.")
        if self._increments is None:
            increments = self.rng.normal(
                0.0, np.sqrt(self.condition.volatility), size=int(dt)
            )
        else:
            increments = self._increments[self.time : self.time + int(dt)]
        if self._latent_values is None:
            latent_path = tuple((self.state + np.cumsum(increments)).tolist())
        else:
            latent_path = tuple(
                self._latent_values[self.time + 1 : self.time + int(dt) + 1].tolist()
            )
        process_increment = float(latent_path[-1] - self.state)
        self.state = float(latent_path[-1])
        observation_noise = (
            self.rng.normal(0.0, np.sqrt(self.condition.stochasticity))
            if self._noises is None
            else self._noises[self.time + int(dt)]
        )
        self.time += int(dt)
        return StepResult(
            time=self.time,
            dt=int(dt),
            latent_state=self.state,
            observation=self.state + observation_noise,
            process_increment=float(process_increment),
            observation_noise=float(observation_noise),
            latent_path=latent_path,
        )

    def sample_process_increments(self, dt: int, size: int) -> np.ndarray:
        if dt <= 0:
            raise ValueError("dt must be positive.")
        return self.rng.normal(0.0, np.sqrt(self.condition.volatility * dt), size=size)
