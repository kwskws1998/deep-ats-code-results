"""Belief-state feature extraction for Deep Active Temporal Sampling."""

from __future__ import annotations

import numpy as np

from deep_ats_planner.beliefs import GridBelief


BASE_FEATURE_NAMES = (
    "predictive_mean",
    "log_expected_volatility",
    "log_expected_stochasticity",
    "log_volatility_sd",
    "log_stochasticity_sd",
    "expected_log_ratio",
    "entropy_fraction",
    "effective_hypothesis_fraction",
    "expected_learning_rate",
    "last_observation",
    "last_reward",
    "last_dt_fraction",
    "remaining_decisions_fraction",
)


def feature_names(
    task_action_count: int,
    action_space: tuple[int, ...] = (1, 2, 4, 8),
) -> tuple[str, ...]:
    predictive_names = tuple(
        f"log_predictive_variance_dt_{int(dt)}" for dt in action_space
    )
    action_names = tuple(f"last_action_{index}" for index in range(task_action_count))
    return BASE_FEATURE_NAMES + predictive_names + action_names


def belief_features(
    belief: GridBelief,
    action_space: tuple[int, ...],
    task_action_count: int,
    last_observation: float,
    last_reward: float,
    last_action: int,
    last_dt: int,
    remaining_decisions_fraction: float = 0.0,
    use_belief_features: bool = True,
) -> np.ndarray:
    summary = belief.summary()
    max_dt = max(action_space)
    hypothesis_count = float(np.prod(belief.shape))
    max_entropy = float(np.log(hypothesis_count))
    last_action_features = np.zeros(int(task_action_count), dtype=float)
    if 0 <= int(last_action) < int(task_action_count):
        last_action_features[int(last_action)] = 1.0

    base = np.array(
        [
            belief.predictive_mean(),
            np.log(max(summary.volatility_mean, 1e-9)),
            np.log(max(summary.stochasticity_mean, 1e-9)),
            np.log1p(max(summary.volatility_sd, 0.0)),
            np.log1p(max(summary.stochasticity_sd, 0.0)),
            summary.expected_log_ratio,
            summary.entropy / max(max_entropy, 1e-9),
            summary.effective_n / max(hypothesis_count, 1e-9),
            summary.learning_rate,
            float(last_observation),
            float(last_reward),
            float(last_dt) / float(max_dt),
            float(remaining_decisions_fraction),
        ],
        dtype=np.float32,
    )
    predictive_variances = np.array(
        [
            np.log1p(max(belief.mixture_predictive_variance(int(dt)), 0.0))
            for dt in action_space
        ],
        dtype=np.float32,
    )
    features = np.concatenate(
        [base, predictive_variances, last_action_features.astype(np.float32)]
    )
    if not use_belief_features:
        features[:9] = 0.0
        features[0] = float(last_observation)
        features[
            len(BASE_FEATURE_NAMES) : len(BASE_FEATURE_NAMES) + len(action_space)
        ] = 0.0
    if not np.isfinite(features).all():
        raise FloatingPointError("Planner features became nonfinite.")
    return features.astype(np.float32)
