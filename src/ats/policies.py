"""Decision rules for active temporal sampling."""

from __future__ import annotations

import numpy as np

from ats.beliefs import GridBelief


def sampling_cost(
    dt: int,
    wait_cost: float,
    observation_cost: float,
    cost_mode: str = "per_observation",
) -> float:
    """Charge once per observation; retain the old rate penalty only for explicit historical reproductions."""
    if dt <= 0 or int(dt) != dt:
        raise ValueError("dt must be a positive integer.")
    if cost_mode == "per_observation":
        return float(wait_cost * dt + observation_cost)
    if cost_mode == "legacy_rate":
        return float(wait_cost * dt + observation_cost / dt)
    raise ValueError(f"Unknown cost mode: {cost_mode}")


def weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    mean = float(np.sum(weights * values))
    return float(np.sum(weights * (values - mean) ** 2))


def active_temporal_sampling_scores(
    belief: GridBelief,
    action_space: tuple[int, ...],
    wait_cost: float,
    observation_rate_cost: float,
    eig_samples: int,
    information_objective: str,
    rng: np.random.Generator,
    cost_mode: str = "per_observation",
) -> dict[int, float]:
    scores: dict[int, float] = {}
    common_seed = int(rng.integers(0, 2**32))
    for dt in action_space:
        candidate_rng = np.random.default_rng(common_seed)
        if information_objective == "entropy":
            information_gain = belief.expected_information_gain(
                dt, eig_samples, candidate_rng
            )
        elif information_objective == "log_ratio":
            information_gain = belief.expected_log_ratio_information_gain(
                dt, eig_samples, candidate_rng
            )
        else:
            raise ValueError(f"Unknown information objective: {information_objective}")
        score = information_gain - sampling_cost(
            dt, wait_cost, observation_rate_cost, cost_mode
        )
        scores[int(dt)] = float(score)
    return scores


def choose_active_dt(
    belief: GridBelief,
    action_space: tuple[int, ...],
    wait_cost: float,
    observation_rate_cost: float,
    eig_samples: int,
    information_objective: str,
    rng: np.random.Generator,
    cost_mode: str = "per_observation",
) -> tuple[int, dict[int, float]]:
    scores = active_temporal_sampling_scores(
        belief=belief,
        action_space=action_space,
        wait_cost=wait_cost,
        observation_rate_cost=observation_rate_cost,
        eig_samples=eig_samples,
        information_objective=information_objective,
        rng=rng,
        cost_mode=cost_mode,
    )
    best_dt = min(action_space)
    best_score = -np.inf
    for dt in sorted(action_space):
        score = scores[int(dt)]
        if score > best_score:
            best_dt = int(dt)
            best_score = score
    return best_dt, scores
