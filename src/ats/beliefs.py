"""Grid posterior over volatility and stochasticity with Kalman updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def logsumexp(values: np.ndarray) -> float:
    max_value = float(np.max(values))
    if not np.isfinite(max_value):
        raise FloatingPointError("No finite likelihood mass remains.")
    return max_value + float(np.log(np.sum(np.exp(values - max_value))))


def logsumexp_axis(values: np.ndarray, axis: int) -> np.ndarray:
    max_value = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(
        max_value
        + np.log(np.sum(np.exp(values - max_value), axis=axis, keepdims=True)),
        axis=axis,
    )


@dataclass(frozen=True)
class BeliefSummary:
    volatility_mean: float
    stochasticity_mean: float
    volatility_sd: float
    stochasticity_sd: float
    expected_log_ratio: float
    entropy: float
    effective_n: float
    learning_rate: float


class GridBelief:
    """Tracks each noise hypothesis with its own scalar Kalman filter."""

    def __init__(
        self,
        volatility_grid: np.ndarray,
        stochasticity_grid: np.ndarray,
        initial_variance: float = 12.0,
    ):
        if volatility_grid.ndim != 1 or stochasticity_grid.ndim != 1:
            raise ValueError(
                "volatility_grid and stochasticity_grid must be one-dimensional."
            )
        for grid in (volatility_grid, stochasticity_grid):
            if not grid.size or not np.all(np.isfinite(grid)) or np.any(grid <= 0):
                raise ValueError(
                    "Noise grids must contain finite, strictly positive variances."
                )
        if not np.isfinite(initial_variance) or initial_variance < 0:
            raise ValueError("initial_variance must be finite and nonnegative.")
        self.volatility_values = volatility_grid.astype(float)
        self.stochasticity_values = stochasticity_grid.astype(float)
        self.volatility, self.stochasticity = np.meshgrid(
            self.volatility_values,
            self.stochasticity_values,
            indexing="ij",
        )
        self.initial_variance = float(initial_variance)
        self.shape = self.volatility.shape
        self.mean = np.zeros(self.shape, dtype=float)
        self.variance = np.full(self.shape, self.initial_variance, dtype=float)
        self.log_weights = np.full(
            self.shape, -np.log(self.volatility.size), dtype=float
        )
        self.last_learning_rates = np.zeros(self.shape, dtype=float)

    def reset(self, initial_observation: float) -> None:
        self.mean = np.full(self.shape, float(initial_observation), dtype=float)
        self.variance = np.full(self.shape, self.initial_variance, dtype=float)
        self.log_weights = np.full(
            self.shape, -np.log(self.volatility.size), dtype=float
        )
        self.last_learning_rates = np.zeros(self.shape, dtype=float)

    @property
    def posterior(self) -> np.ndarray:
        weights = np.exp(self.log_weights)
        return weights / np.sum(weights)

    def predictive_mean(self) -> float:
        return float(np.sum(self.posterior * self.mean))

    def predictive_variance_by_hypothesis(self, dt: int) -> np.ndarray:
        if int(dt) != dt or dt <= 0:
            raise ValueError("dt must be a positive integer.")
        return self.variance + self.volatility * int(dt) + self.stochasticity

    def predictive_log_density(self, observation: float, dt: int) -> float:
        """Evaluate the exact posterior predictive Gaussian mixture before updating."""
        values = self.predictive_log_likelihood(np.array([float(observation)]), dt)[0]
        return logsumexp((self.log_weights + values).ravel())

    def mixture_predictive_variance(self, dt: int) -> float:
        weights = self.posterior
        within = self.predictive_variance_by_hypothesis(dt)
        mix_mean = float(np.sum(weights * self.mean))
        between = (self.mean - mix_mean) ** 2
        return float(np.sum(weights * (within + between)))

    def predictive_log_likelihood(
        self, observations: np.ndarray, dt: int
    ) -> np.ndarray:
        pred_var = self.predictive_variance_by_hypothesis(dt)
        delta = observations[:, None, None] - self.mean[None, :, :]
        return -0.5 * (
            np.log(2.0 * np.pi * pred_var[None, :, :])
            + (delta**2) / pred_var[None, :, :]
        )

    def sampled_observations(
        self, dt: int, sample_count: int, rng: np.random.Generator
    ) -> np.ndarray:
        weights = self.posterior.ravel()
        means = self.mean.ravel()
        variances = self.predictive_variance_by_hypothesis(dt).ravel()
        sample_indices = rng.choice(weights.size, size=int(sample_count), p=weights)
        return rng.normal(means[sample_indices], np.sqrt(variances[sample_indices]))

    def hypothetical_posteriors(
        self, observations: np.ndarray, dt: int
    ) -> tuple[np.ndarray, np.ndarray]:
        log_likelihood = self.predictive_log_likelihood(observations, dt)
        candidate_log_weights = self.log_weights[None, :, :] + log_likelihood
        normalizers = logsumexp_axis(
            candidate_log_weights.reshape(observations.size, -1), axis=1
        )
        normalized = candidate_log_weights - normalizers[:, None, None]
        return np.exp(normalized), normalized

    def expected_entropy_after_observation(
        self,
        dt: int,
        sample_count: int,
        rng: np.random.Generator,
    ) -> float:
        observations = self.sampled_observations(dt, sample_count, rng)
        candidate_weights, normalized = self.hypothetical_posteriors(observations, dt)
        entropies = -np.sum(candidate_weights * normalized, axis=(1, 2))
        return float(np.mean(entropies))

    def expected_information_gain(
        self,
        dt: int,
        sample_count: int,
        rng: np.random.Generator,
    ) -> float:
        """Estimate mutual information using posterior-to-prior KL for each simulated outcome."""
        if sample_count <= 0:
            raise ValueError("sample_count must be positive.")
        observations = self.sampled_observations(dt, sample_count, rng)
        weights, log_weights = self.hypothetical_posteriors(observations, dt)
        terms = np.zeros_like(weights)
        np.multiply(
            weights,
            log_weights - self.log_weights[None, :, :],
            out=terms,
            where=weights > 0,
        )
        return float(max(np.mean(np.sum(terms, axis=(1, 2))), 0.0))

    def marginal_interval(
        self, parameter: str, mass: float = 0.95
    ) -> tuple[float, float]:
        """Return equal-tailed marginal credible limits on the finite grid."""
        if not 0 < mass < 1 or parameter not in {"volatility", "stochasticity"}:
            raise ValueError(
                "Specify volatility or stochasticity and a mass between zero and one."
            )
        axis = 1 if parameter == "volatility" else 0
        values = (
            self.volatility_values
            if parameter == "volatility"
            else self.stochasticity_values
        )
        cumulative = np.cumsum(self.posterior.sum(axis=axis))
        indices = np.searchsorted(cumulative, [(1 - mass) / 2, (1 + mass) / 2])
        return tuple(float(values[min(index, len(values) - 1)]) for index in indices)

    def log_ratio_variance(self, weights: np.ndarray | None = None) -> float:
        if weights is None:
            weights = self.posterior
        log_ratio = np.log(self.volatility / self.stochasticity)
        mean = float(np.sum(weights * log_ratio))
        return float(np.sum(weights * (log_ratio - mean) ** 2))

    def expected_log_ratio_variance_after_observation(
        self,
        dt: int,
        sample_count: int,
        rng: np.random.Generator,
    ) -> float:
        observations = self.sampled_observations(dt, sample_count, rng)
        candidate_weights, _ = self.hypothetical_posteriors(observations, dt)
        log_ratio = np.log(self.volatility / self.stochasticity)[None, :, :]
        means = np.sum(candidate_weights * log_ratio, axis=(1, 2))
        variances = np.sum(
            candidate_weights * (log_ratio - means[:, None, None]) ** 2, axis=(1, 2)
        )
        return float(np.mean(variances))

    def expected_log_ratio_information_gain(
        self,
        dt: int,
        sample_count: int,
        rng: np.random.Generator,
    ) -> float:
        return (
            self.log_ratio_variance()
            - self.expected_log_ratio_variance_after_observation(
                dt,
                sample_count,
                rng,
            )
        )

    def update(self, observation: float, dt: int) -> BeliefSummary:
        if not np.isfinite(observation):
            raise ValueError("observation must be finite.")
        pred_var = self.predictive_variance_by_hypothesis(dt)
        delta = float(observation) - self.mean
        log_likelihood = -0.5 * (np.log(2.0 * np.pi * pred_var) + (delta**2) / pred_var)
        self.log_weights = self.log_weights + log_likelihood
        self.log_weights = self.log_weights - logsumexp(self.log_weights)

        prior_var = self.variance + self.volatility * int(dt)
        kalman_gain = prior_var / (prior_var + self.stochasticity)
        self.mean = self.mean + kalman_gain * delta
        self.variance = np.maximum((1.0 - kalman_gain) * prior_var, 1e-12)
        self.last_learning_rates = kalman_gain
        return self.summary()

    def summary(self) -> BeliefSummary:
        weights = self.posterior
        v_mean = float(np.sum(weights * self.volatility))
        s_mean = float(np.sum(weights * self.stochasticity))
        v_sd = float(np.sqrt(np.sum(weights * (self.volatility - v_mean) ** 2)))
        s_sd = float(np.sqrt(np.sum(weights * (self.stochasticity - s_mean) ** 2)))
        log_ratio = float(
            np.sum(weights * np.log(self.volatility / self.stochasticity))
        )
        entropy = float(-np.sum(weights * self.log_weights))
        learning_rate = float(np.sum(weights * self.last_learning_rates))
        return BeliefSummary(
            volatility_mean=v_mean,
            stochasticity_mean=s_mean,
            volatility_sd=v_sd,
            stochasticity_sd=s_sd,
            expected_log_ratio=log_ratio,
            entropy=entropy,
            effective_n=float(np.exp(entropy)),
            learning_rate=learning_rate,
        )
