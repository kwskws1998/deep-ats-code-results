"""Gymnasium-compatible wrappers for RWRL and generic continuous-control environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def flatten_observation(observation: Any) -> np.ndarray:
    if isinstance(observation, dict):
        parts = [
            flatten_observation(observation[key]) for key in sorted(observation.keys())
        ]
        return np.concatenate(parts).astype(np.float32)
    if isinstance(observation, (list, tuple)):
        parts = [flatten_observation(item) for item in observation]
        return np.concatenate(parts).astype(np.float32)
    return np.asarray(observation, dtype=np.float32).reshape(-1)


def _space_from_spec(spec: Any) -> spaces.Box:
    shape = tuple(int(dim) for dim in spec.shape)
    dtype = np.float32
    if hasattr(spec, "minimum") and hasattr(spec, "maximum"):
        low = np.asarray(spec.minimum, dtype=dtype)
        high = np.asarray(spec.maximum, dtype=dtype)
        if low.shape == ():
            low = np.full(shape, float(low), dtype=dtype)
        if high.shape == ():
            high = np.full(shape, float(high), dtype=dtype)
        return spaces.Box(low=low, high=high, dtype=dtype)
    return spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=dtype)


class RWRLGymWrapper(gym.Env):
    """Wraps a realworldrl-suite dm_env environment as a Gymnasium environment."""

    metadata = {"render_modes": []}

    def __init__(self, env: Any):
        self.env = env
        self.action_space = _space_from_spec(env.action_spec())
        initial = env.reset()
        flat = flatten_observation(initial.observation)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=flat.shape, dtype=np.float32
        )
        self._needs_reset = False

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if (
            seed is not None
            and hasattr(self.env, "task")
            and hasattr(self.env.task, "random")
        ):
            self.env.task.random.seed(seed)
        timestep = self.env.reset()
        self._needs_reset = False
        return flatten_observation(timestep.observation), {}

    def step(self, action: np.ndarray):
        if self._needs_reset:
            observation, info = self.reset()
            return observation, 0.0, False, False, info
        timestep = self.env.step(np.asarray(action, dtype=np.float32))
        observation = flatten_observation(timestep.observation)
        reward = 0.0 if timestep.reward is None else float(timestep.reward)
        ended = bool(timestep.last())
        terminated = ended and (
            timestep.discount is None or float(timestep.discount) == 0.0
        )
        truncated = ended and not terminated
        self._needs_reset = ended
        return (
            observation,
            reward,
            terminated,
            truncated,
            {"dm_env_discount": timestep.discount},
        )

    def close(self) -> None:
        """Release the underlying simulator and its logger."""
        if hasattr(self.env, "close"):
            self.env.close()


@dataclass(frozen=True)
class RWRLSpec:
    domain_name: str = "walker"
    task_name: str = "realworld_walk"
    combined_challenge: str = "easy"
    log_output: str | None = None


def make_rwrl_env(spec: RWRLSpec) -> gym.Env:
    try:
        import realworldrl_suite.environments as rwrl
    except ImportError as exc:
        raise ImportError(
            "realworldrl_suite is not installed. Install it from "
            "https://github.com/google-research/realworldrl_suite to run RWRL experiments."
        ) from exc
    env = rwrl.load(
        domain_name=spec.domain_name,
        task_name=spec.task_name,
        combined_challenge=spec.combined_challenge,
        log_output=spec.log_output,
        environment_kwargs={"log_safety_vars": True, "flat_observation": True},
    )
    return RWRLGymWrapper(env)


def make_continuous_env(
    env_id: str, seed: int, rwrl_spec: RWRLSpec | None = None
) -> gym.Env:
    if env_id == "rwrl":
        env = make_rwrl_env(rwrl_spec or RWRLSpec())
    else:
        env = gym.make(env_id)
    env.action_space.seed(seed)
    return env
