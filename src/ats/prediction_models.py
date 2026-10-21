"""Causal perceptual-model fits to quantized human bucket predictions.

The response layer is shared across models: response inertia, additive bias, and
Gaussian response noise. The likelihood integrates over the two-unit response
bins and censors the endpoint bins. A prediction never uses its current outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr

from ats.beliefs import GridBelief
from ats.config import GridConfig
from ats.fit_support import select_optimization


PREDICTION_MODELS = (
    "joint_grid",
    "volatility_only_grid",
    "stochasticity_only_grid",
    "delta_rule",
    "fitted_kalman",
    "blockwise_delta",
    "blockwise_kalman",
    "last_outcome",
    "response_history",
)
FITTED_PREDICTION_MODELS = (
    "delta_rule",
    "fitted_kalman",
    "blockwise_delta",
    "blockwise_kalman",
)


@dataclass(frozen=True)
class PredictionFit:
    """Per-person training estimates; the first three parameters form the common response layer."""

    model: str
    parameters: np.ndarray
    training_nll: float
    success: bool
    boundary_parameters: tuple[int, ...]


def perceptual_predictions(
    outcomes: np.ndarray,
    model: str,
    parameter: float | None = None,
    grid_size: int = 25,
) -> np.ndarray:
    """Predict trial t from outcomes strictly before t, resetting the prior at each condition block."""
    outcomes = np.asarray(outcomes, dtype=float)
    if outcomes.ndim != 2 or not np.isfinite(outcomes).all():
        raise ValueError("Expected a finite trial-by-block outcome matrix.")
    count, blocks = outcomes.shape
    predictions = np.full_like(outcomes, 60.0)
    if model == "response_history":
        return predictions
    if model == "last_outcome":
        predictions[1:] = outcomes[:-1]
        return predictions
    if model in FITTED_PREDICTION_MODELS:
        if parameter is None:
            raise ValueError("A learning parameter is required.")
        mean, variance = np.full(blocks, 60.0), np.full(blocks, 12.0)
        for trial in range(count):
            predictions[trial] = mean
            if model in ("delta_rule", "blockwise_delta"):
                gain = np.asarray(parameter)
            else:
                process = np.exp(parameter)
                gain = (variance + process) / (variance + process + 1)
                variance = (1 - gain) * (variance + process)
            mean = mean + gain * (outcomes[trial] - mean)
        return predictions
    volatility, stochasticity = GridConfig(size=grid_size).make_grids()
    if model == "volatility_only_grid":
        stochasticity = np.array([3.0])
    elif model == "stochasticity_only_grid":
        volatility = np.array([1.0])
    elif model != "joint_grid":
        raise ValueError(f"Unknown perceptual model: {model}")
    for block in range(blocks):
        belief = GridBelief(volatility, stochasticity, initial_variance=12.0)
        belief.reset(0.0)
        for trial in range(count):
            predictions[trial, block] = 60 + 4 * belief.predictive_mean()
            belief.update((outcomes[trial, block] - 60) / 4, 1)
    return predictions


def bucket_log_probabilities(
    responses: np.ndarray, means: np.ndarray, sigma: float
) -> np.ndarray:
    """Compute stable Gaussian interval probabilities for quantization with censored 10/90 endpoints."""
    if (
        responses.shape != means.shape
        or not np.isfinite(responses).all()
        or not np.isfinite(means).all()
        or not np.isfinite(sigma)
        or sigma <= 0
    ):
        raise ValueError("Invalid response likelihood inputs.")
    if np.any((responses < 10) | (responses > 90)) or np.any(responses % 2):
        raise ValueError("Responses must lie on the even-valued bucket grid.")
    lower = (responses - 1 - means) / sigma
    upper = (responses + 1 - means) / sigma
    lower = np.where(responses == 10, -np.inf, lower)
    upper = np.where(responses == 90, np.inf, upper)
    use_survival = lower > 0
    log_big = log_ndtr(np.where(use_survival, -lower, upper))
    log_small = log_ndtr(np.where(use_survival, -upper, lower))
    return log_big + np.log(-np.expm1(log_small - log_big))


def response_means(
    template: np.ndarray, responses: np.ndarray, parameters: np.ndarray
) -> np.ndarray:
    """Condition one-step responses on the preceding human response; never read the current target."""
    previous = np.vstack((np.full((1, responses.shape[1]), 60.0), responses[:-1]))
    inertia, bias = parameters[:2]
    return template + inertia * (previous - template) + bias


def observed_response_expectation(means: np.ndarray, sigma: float) -> np.ndarray:
    """Account for clipping and rounding when reporting expected observed bucket positions."""
    thresholds = np.arange(11, 90, 2)
    return 10 + 2 * np.sum(ndtr((means[..., None] - thresholds) / sigma), axis=-1)


def prediction_parameter_bounds(model: str) -> tuple:
    """Use the same response-parameter ranges for all perceptual alternatives."""
    bounds = ((0, 0.98), (-10, 10), (np.log(0.25), np.log(30)))
    if model == "delta_rule":
        bounds += ((0.001, 0.999),)
    elif model == "fitted_kalman":
        bounds += ((np.log(0.001), np.log(30)),)
    elif model == "blockwise_delta":
        bounds += ((0.001, 0.999),) * 4
    elif model == "blockwise_kalman":
        bounds += ((np.log(0.001), np.log(30)),) * 4
    return bounds


def learning_parameters(model: str, parameters: np.ndarray):
    """Distinguish a shared learning rule from a stronger four-condition fitted control."""
    if model.startswith("blockwise_"):
        return parameters[3:]
    return parameters[3] if len(parameters) > 3 else None


def prediction_parameter_names(model: str) -> tuple[str, ...]:
    """Name response parameters and any fitted shared or condition-specific learning parameter."""
    names = ("response_inertia", "response_bias", "log_response_sigma")
    if model in ("delta_rule", "blockwise_delta"):
        names += tuple(
            "learning_rate"
            + (f"_condition_{i}" if model.startswith("blockwise") else "")
            for i in range(4 if model.startswith("blockwise") else 1)
        )
    elif model in ("fitted_kalman", "blockwise_kalman"):
        names += tuple(
            "log_process_to_observation_ratio"
            + (f"_condition_{i}" if model.startswith("blockwise") else "")
            for i in range(4 if model.startswith("blockwise") else 1)
        )
    return names


def fit_prediction_model(
    model: str,
    outcomes: np.ndarray,
    responses: np.ndarray,
    training_mask: np.ndarray,
    fixed_template: np.ndarray | None = None,
) -> PredictionFit:
    """Fit only selected past response targets; held-out outcomes remain causal sensory inputs."""
    if (
        outcomes.shape != responses.shape
        or training_mask.shape != responses.shape
        or not training_mask.any()
    ):
        raise ValueError("Invalid prediction fitting arrays.")
    bounds = prediction_parameter_bounds(model)
    if model not in FITTED_PREDICTION_MODELS and fixed_template is None:
        fixed_template = perceptual_predictions(outcomes, model)

    def objective(parameters):
        """Score quantized responses without using target bucket positions to update sensory beliefs."""
        template = (
            fixed_template
            if fixed_template is not None
            else perceptual_predictions(
                outcomes, model, learning_parameters(model, parameters)
            )
        )
        means = response_means(template, responses, parameters)
        log_probability = bucket_log_probabilities(
            responses[training_mask], means[training_mask], np.exp(parameters[2])
        )
        return float(-log_probability.sum())

    starts = []
    for inertia in (0.1, 0.6):
        start = [inertia, 0, np.log(6)]
        if model == "delta_rule":
            start.append(0.3 if inertia < 0.5 else 0.7)
        elif model == "fitted_kalman":
            start.append(np.log(0.5) if inertia < 0.5 else np.log(3))
        elif model == "blockwise_delta":
            start.extend([0.3 if inertia < 0.5 else 0.7] * 4)
        elif model == "blockwise_kalman":
            start.extend([np.log(0.5) if inertia < 0.5 else np.log(3)] * 4)
        starts.append(start)
    results = [
        minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 400, "ftol": 1e-10, "gtol": 1e-6},
        )
        for start in starts
    ]
    best = select_optimization(results, objective, bounds)
    boundary = tuple(
        i
        for i, (value, (low, high)) in enumerate(zip(best.x, bounds))
        if min(value - low, high - value) < 1e-5 * max(1, high - low)
    )
    return PredictionFit(model, best.x, float(best.fun), bool(best.success), boundary)


def predict_bucket_responses(
    fit: PredictionFit,
    outcomes: np.ndarray,
    responses: np.ndarray,
    fixed_template: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return frozen-fit expected responses and proper probabilities for observed response bins."""
    template = (
        fixed_template
        if fixed_template is not None
        else perceptual_predictions(
            outcomes, fit.model, learning_parameters(fit.model, fit.parameters)
        )
    )
    means = response_means(template, responses, fit.parameters)
    sigma = float(np.exp(fit.parameters[2]))
    return observed_response_expectation(means, sigma), bucket_log_probabilities(
        responses, means, sigma
    )


def simulate_bucket_responses(
    outcomes: np.ndarray, model: str, parameters: np.ndarray, seed: int
) -> np.ndarray:
    """Generate quantized closed-loop responses with each synthetic participant's own response history."""
    template = perceptual_predictions(
        outcomes, model, learning_parameters(model, parameters)
    )
    rng = np.random.default_rng(seed)
    responses = np.empty_like(outcomes)
    previous = np.full(outcomes.shape[1], 60.0)
    for trial in range(len(outcomes)):
        mean = (
            template[trial]
            + parameters[0] * (previous - template[trial])
            + parameters[1]
        )
        continuous = mean + rng.normal(0, np.exp(parameters[2]), size=outcomes.shape[1])
        responses[trial] = np.clip(2 * np.round(continuous / 2), 10, 90)
        previous = responses[trial]
    return responses
