"""ATS belief features for high-dimensional continuous-control observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deep_ats_planner.beliefs import GridBelief
from deep_ats_planner.config import AgentConfig, GridConfig


@dataclass(frozen=True)
class ATSFeatureSpec:
    projection_count: int = 8
    projection_seed: int = 20260521
    update_dt: int = 1
    observation_scale: tuple[float, ...] | None = None


ATS_FEATURE_NAMES = (
    "ats_proxy_process_variance_mean",
    "ats_proxy_observation_variance_mean",
    "ats_proxy_process_variance_between_projection_sd",
    "ats_proxy_observation_variance_between_projection_sd",
    "ats_proxy_log_ratio_mean",
    "ats_proxy_log_ratio_between_projection_sd",
    "ats_proxy_grid_entropy_mean",
    "ats_proxy_effective_grid_size_mean",
    "ats_proxy_learning_rate_mean",
    "ats_proxy_prediction_mean",
    "ats_scaled_projection_variance",
)


class RandomProjectionATSEncoder:
    """Runs several scalar ATS filters over random projections of vector observations."""

    def __init__(
        self,
        observation_dim: int,
        grid: GridConfig,
        agent: AgentConfig,
        spec: ATSFeatureSpec,
    ):
        self.observation_dim = int(observation_dim)
        self.spec = spec
        if self.observation_dim < 1 or spec.projection_count < 1 or spec.update_dt < 1:
            raise ValueError(
                "Observation dimension, projection count, and update interval must be positive."
            )
        self.observation_scale = (
            np.ones(self.observation_dim)
            if spec.observation_scale is None
            else np.asarray(spec.observation_scale, dtype=float)
        )
        if (
            self.observation_scale.shape != (self.observation_dim,)
            or not np.isfinite(self.observation_scale).all()
            or np.any(self.observation_scale <= 0)
        ):
            raise ValueError(
                "observation_scale must contain one finite positive scale per coordinate."
            )
        self.action_space = agent.action_space
        volatility_grid, stochasticity_grid = grid.make_grids()
        self.beliefs = [
            GridBelief(
                volatility_grid,
                stochasticity_grid,
                initial_variance=agent.initial_variance,
            )
            for _ in range(int(spec.projection_count))
        ]
        rng = np.random.default_rng(spec.projection_seed)
        projections = rng.normal(
            size=(int(spec.projection_count), self.observation_dim)
        )
        norms = np.linalg.norm(projections, axis=1, keepdims=True)
        self.projections = projections / np.maximum(norms, 1e-9)
        self.initialized = False
        self.last_projected = np.zeros(int(spec.projection_count), dtype=float)

    @property
    def feature_dim(self) -> int:
        return len(ATS_FEATURE_NAMES) + len(self.action_space)

    @property
    def feature_names(self) -> tuple[str, ...]:
        predictive = tuple(
            f"ats_proxy_log_pred_var_dt_{int(dt)}" for dt in self.action_space
        )
        return ATS_FEATURE_NAMES + predictive

    def reset(self, observation: np.ndarray) -> np.ndarray:
        projected = self.project(observation)
        for belief, value in zip(self.beliefs, projected, strict=True):
            belief.reset(float(value))
        self.last_projected = projected
        self.initialized = True
        return self.features()

    def update(self, observation: np.ndarray, dt: int | None = None) -> np.ndarray:
        if not self.initialized:
            return self.reset(observation)
        projected = self.project(observation)
        step_dt = self.spec.update_dt if dt is None else int(dt)
        for belief, value in zip(self.beliefs, projected, strict=True):
            belief.update(float(value), step_dt)
        self.last_projected = projected
        return self.features()

    def project(self, observation: np.ndarray) -> np.ndarray:
        flat = np.asarray(observation, dtype=float).reshape(-1)
        if flat.size != self.observation_dim:
            raise ValueError(
                f"Expected observation_dim={self.observation_dim}, got {flat.size}."
            )
        if not np.isfinite(flat).all():
            raise ValueError("Observations must be finite.")
        return self.projections @ (flat / self.observation_scale)

    def features(self) -> np.ndarray:
        summaries = [belief.summary() for belief in self.beliefs]
        volatility = np.array([item.volatility_mean for item in summaries], dtype=float)
        stochasticity = np.array(
            [item.stochasticity_mean for item in summaries], dtype=float
        )
        log_ratio = np.array(
            [item.expected_log_ratio for item in summaries], dtype=float
        )
        entropy = np.array([item.entropy for item in summaries], dtype=float)
        effective_n = np.array([item.effective_n for item in summaries], dtype=float)
        learning_rate = np.array(
            [item.learning_rate for item in summaries], dtype=float
        )
        predictions = np.array(
            [belief.predictive_mean() for belief in self.beliefs], dtype=float
        )
        base = np.array(
            [
                np.mean(volatility),
                np.mean(stochasticity),
                np.std(volatility),
                np.std(stochasticity),
                np.mean(log_ratio),
                np.std(log_ratio),
                np.mean(entropy),
                np.mean(effective_n),
                np.mean(learning_rate),
                np.mean(predictions),
                np.var(self.last_projected),
            ],
            dtype=np.float32,
        )
        predictive = np.array(
            [
                np.mean(
                    [
                        np.log1p(max(belief.mixture_predictive_variance(int(dt)), 0.0))
                        for belief in self.beliefs
                    ]
                )
                for dt in self.action_space
            ],
            dtype=np.float32,
        )
        features = np.concatenate([base, predictive])
        if not np.isfinite(features).all():
            raise FloatingPointError("ATS proxy features became nonfinite.")
        return features


class NullATSEncoder:
    """Drop-in encoder used for PPO baselines without ATS features."""

    feature_dim = 0
    feature_names: tuple[str, ...] = ()

    def reset(self, observation: np.ndarray) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)

    def update(self, observation: np.ndarray, dt: int | None = None) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)


class HistoryEncoder:
    """Expose raw observation history as a temporal-memory control."""

    def __init__(self, observation_dim: int, length: int):
        self.length = int(length)
        self.feature_dim = int(observation_dim) * self.length
        self.feature_names = tuple(
            f"history_{lag}_{dim}"
            for lag in range(self.length)
            for dim in range(observation_dim)
        )
        self.history = None

    def reset(self, observation: np.ndarray) -> np.ndarray:
        self.history = np.tile(
            np.asarray(observation, dtype=np.float32).reshape(1, -1), (self.length, 1)
        )
        return self.history.reshape(-1).copy()

    def update(self, observation: np.ndarray, dt: int | None = None) -> np.ndarray:
        if self.history is None:
            return self.reset(observation)
        self.history = np.concatenate(
            [
                np.asarray(observation, dtype=np.float32).reshape(1, -1),
                self.history[:-1],
            ]
        )
        return self.history.reshape(-1).copy()


class EMAEncoder:
    """Track inexpensive per-coordinate means, residual variances, and increments."""

    def __init__(self, observation_dim: int, rate: float = 0.1):
        self.rate = rate
        self.feature_dim = 3 * observation_dim
        self.feature_names = tuple(
            f"ema_{stat}_{dim}"
            for stat in ("mean", "variance", "delta")
            for dim in range(observation_dim)
        )
        self.mean = None

    def reset(self, observation: np.ndarray) -> np.ndarray:
        self.mean = np.asarray(observation, dtype=np.float32).copy()
        self.variance = np.zeros_like(self.mean)
        self.previous = self.mean.copy()
        return np.concatenate([self.mean, self.variance, np.zeros_like(self.mean)])

    def update(self, observation: np.ndarray, dt: int | None = None) -> np.ndarray:
        if self.mean is None:
            return self.reset(observation)
        observation = np.asarray(observation, dtype=np.float32)
        error = observation - self.mean
        delta = observation - self.previous
        self.mean += self.rate * error
        self.variance = (1 - self.rate) * self.variance + self.rate * error**2
        self.previous = observation.copy()
        return np.concatenate([self.mean, self.variance, delta])


class PaddedEncoder:
    """Keep actor-critic input dimensions identical across feature ablations."""

    def __init__(self, encoder, width: int):
        self.encoder = encoder
        self.feature_dim = int(width)
        if width < encoder.feature_dim:
            raise ValueError("Feature padding width is too small.")
        self.feature_names = tuple(encoder.feature_names) + tuple(
            f"padding_{index}" for index in range(width - encoder.feature_dim)
        )

    def _pad(self, features: np.ndarray) -> np.ndarray:
        return np.pad(features, (0, self.feature_dim - features.size)).astype(
            np.float32
        )

    def reset(self, observation: np.ndarray) -> np.ndarray:
        return self._pad(self.encoder.reset(observation))

    def update(self, observation: np.ndarray, dt: int | None = None) -> np.ndarray:
        return self._pad(self.encoder.update(observation, dt))
