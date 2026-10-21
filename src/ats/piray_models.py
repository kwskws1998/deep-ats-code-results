"""Causal Python implementations of the core Piray--Daw 2024 comparison.

References are the archived kf_fit.m, other_rw.m, hpl_model.m, and paper
equations 1--9. Predictions precede the current bag outcome. Response inertia
and bias from the earlier exploratory analysis are deliberately absent here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import LinearConstraint, minimize, minimize_scalar
from scipy.special import digamma, expit, gammaln, logsumexp
from scipy.special import ndtr

from ats.prediction_models import bucket_log_probabilities, perceptual_predictions


DETERMINISTIC_MODELS = ("rw4", "kalman4", "ats_grid_fixed", "last_outcome")
PARAMETER_COUNTS = dict(
    rw4=4, kalman4=4, ats_grid_fixed=0, last_outcome=0, hierarchical_pf=3
)
EFFECT_MATRIX = np.array(
    [[0.25, 0.25, 0.25, 0.25], [1, 1, -1, -1], [-1, 1, -1, 1], [1, -1, -1, 1]]
)


@dataclass(frozen=True)
class PerceptualFit:
    """Training-only perceptual estimates and an independent response-noise readout."""

    model: str
    parameters: np.ndarray
    training_sse: float
    success: bool
    boundary: bool


def deterministic_predictions(outcomes, model, parameters=()):
    """Return causal prior means and derivatives with the paper's initial mean and Kalman variance."""
    y = np.asarray(outcomes, dtype=float)
    if y.ndim != 2 or y.shape[1] != 4 or not np.isfinite(y).all():
        raise ValueError("Expected finite outcomes in four condition-aligned blocks.")
    if model in ("ats_grid_fixed", "last_outcome"):
        template = perceptual_predictions(
            y, "joint_grid" if model == "ats_grid_fixed" else model
        )
        return template, np.zeros_like(template)
    p = np.asarray(parameters, dtype=float)
    if model not in ("rw4", "kalman4") or p.shape != (4,) or not np.isfinite(p).all():
        raise ValueError("Unknown model or invalid parameters.")
    if np.any(p < 0) or (model == "rw4" and np.any(p > 1)):
        raise ValueError("Invalid learning parameters.")
    mean, variance = np.full(4, 60.0), np.ones(4)
    derivative, variance_derivative = np.zeros(4), np.zeros(4)
    values, gradients = np.empty_like(y), np.empty_like(y)
    for trial, outcome in enumerate(y):
        values[trial], gradients[trial] = mean, derivative
        if model == "rw4":
            gain, gain_derivative = p, np.ones(4)
        else:
            predicted_variance = variance + p
            gain = predicted_variance / (predicted_variance + 1)
            gain_derivative = (variance_derivative + 1) / (predicted_variance + 1) ** 2
            variance, variance_derivative = gain, gain_derivative
        error = outcome - mean
        derivative = (1 - gain) * derivative + gain_derivative * error
        mean = mean + gain * error
    return values, gradients


def fit_deterministic(model, outcomes, responses, mask, starts=10):
    """Minimize response SSE with common starts and the released Kalman effect constraints."""
    y, response, mask = (
        np.asarray(outcomes),
        np.asarray(responses),
        np.asarray(mask, dtype=bool),
    )
    if y.shape != response.shape or mask.shape != y.shape or not mask.any():
        raise ValueError("Invalid fitting mask.")
    if model in ("ats_grid_fixed", "last_outcome"):
        mean, _ = deterministic_predictions(y, model)
        return PerceptualFit(
            model,
            np.array([]),
            float(np.square(mean[mask] - response[mask]).sum()),
            True,
            False,
        )

    def objective(parameters):
        """Differentiate each condition's causal predictions before summing masked response errors."""
        mean, derivatives = deterministic_predictions(y, model, parameters)
        error = np.where(mask, mean - response, 0.0)
        return (
            float(np.square(error).sum() / mask.sum()),
            2 * (error * derivatives).sum(axis=0) / mask.sum(),
        )

    rng = np.random.default_rng(20241021)
    if model == "rw4":
        bounds, constraints, method = [(0, 1)] * 4, (), "L-BFGS-B"
        initials = [np.full(4, 0.25), np.full(4, 0.75)]
        initials += list(rng.uniform(0, 1, size=(max(0, starts - 2), 4)))
    elif model == "kalman4":
        bounds, method = [(0, 400)] * 4, "SLSQP"
        constraints = (
            LinearConstraint(
                EFFECT_MATRIX, [0, -100, -100, -100], [100, 100, 100, 100]
            ),
        )
        initials = [np.ones(4), np.full(4, 20)]
        initials += list(rng.uniform(0, 50, size=(max(0, starts - 2), 4)))
    else:
        raise ValueError("Unknown deterministic model.")
    results = [
        minimize(
            objective,
            x,
            jac=True,
            method=method,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        for x in initials[:starts]
    ]
    valid = []
    for result in results:
        effects = EFFECT_MATRIX @ result.x
        feasible = model != "kalman4" or (
            np.all(effects >= np.array([0, -100, -100, -100]) - 1e-6)
            and np.all(effects <= 100 + 1e-6)
        )
        if np.isfinite(result.fun) and feasible:
            valid.append(result)
    if not valid:
        raise FloatingPointError("No finite feasible perceptual fit.")
    best = min(valid, key=lambda r: r.fun)
    equivalent = [
        r
        for r in valid
        if r.success and r.fun <= best.fun + 1e-7 * max(1, abs(best.fun))
    ]
    if equivalent:
        best = min(equivalent, key=lambda r: r.fun)
    boundary = any(
        min(value - low, high - value) <= 1e-5 * max(1, high - low)
        for value, (low, high) in zip(best.x, bounds)
    )
    if model == "kalman4":
        effect = EFFECT_MATRIX @ best.x
        boundary |= bool(
            np.any(np.abs(effect[1:]) >= 100 - 1e-5) or effect[0] >= 100 - 1e-5
        )
    return PerceptualFit(
        model, best.x, float(best.fun * mask.sum()), bool(best.success), boundary
    )


def fit_response_sigma(responses, means, mask):
    """Fit only training response noise using the proper quantized/censored response likelihood."""
    result = minimize_scalar(
        lambda x: -bucket_log_probabilities(
            responses[mask], means[mask], np.exp(x)
        ).sum(),
        bounds=(np.log(0.25), np.log(30)),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not result.success or not np.isfinite(result.fun):
        raise FloatingPointError("Response-noise fitting failed.")
    return float(np.exp(result.x))


def unit_gaussian_bic(sse, observations, parameters):
    """Use exactly one unit-Gaussian normalization constant per response for every model."""
    if not np.isfinite(sse) or sse < 0 or observations <= 0 or parameters < 0:
        raise ValueError("Invalid BIC inputs.")
    return float(
        sse + observations * np.log(2 * np.pi) + parameters * np.log(observations)
    )


def particle_predictions(
    outcomes, parameters, particles=10000, seed=0, released_floor=True
):
    """Port the released Rao--Blackwellized particle dynamics with systematic ESS resampling.

    Parameters are change probabilities mu_s, mu_v, and log-variance random-walk
    standard deviation sigma. The optional epsilon likelihood floor is present
    in the released code; disabling it uses a stable version of paper Eq. 9.
    Initial s=v=1, state variance=10, and state mean=60 match the release.
    """
    y = np.asarray(outcomes, dtype=float)
    p = np.asarray(parameters, dtype=float)
    if y.ndim != 2 or y.shape[1] != 4 or not np.isfinite(y).all() or p.shape != (3,):
        raise ValueError("Invalid particle-filter input.")
    mu_s, mu_v, sigma = p
    if not 0 <= mu_s <= 1 or not 0 <= mu_v <= 1 or not 0 < sigma <= 1 or particles < 2:
        raise ValueError("Invalid particle-filter parameters.")
    result = np.empty_like(y)
    resampling_count = 0
    for block in range(4):
        rng = np.random.default_rng(np.random.SeedSequence([seed, block]))
        stochasticity, volatility = np.ones(particles), np.ones(particles)
        mean, variance = np.full(particles, 60.0), np.full(particles, 10.0)
        weights = np.full(particles, 1 / particles)
        for trial in range(len(y)):
            result[trial, block] = weights @ mean
            log_s = np.log(stochasticity) + (
                rng.random(particles) < mu_s
            ) * sigma * rng.standard_normal(particles)
            log_v = np.log(volatility) + (
                rng.random(particles) < mu_v
            ) * sigma * rng.standard_normal(particles)
            if max(np.max(log_s), np.max(log_v)) > 700:
                raise FloatingPointError("Particle variance overflow.")
            stochasticity, volatility = np.exp(log_s), np.exp(log_v)
            total_variance = variance + volatility + stochasticity
            log_likelihood = -0.5 * (
                np.log(2 * np.pi * total_variance)
                + np.square(y[trial, block] - mean) / total_variance
            )
            if released_floor:
                log_likelihood = np.logaddexp(
                    log_likelihood, np.log(np.finfo(float).eps)
                )
            log_weight = (
                np.log(np.maximum(weights, np.finfo(float).tiny)) + log_likelihood
            )
            weights = np.exp(log_weight - logsumexp(log_weight))
            if 1 / (weights @ weights) < particles / 2:
                positions = (np.arange(particles) + rng.random()) / particles
                cumulative = np.cumsum(weights)
                cumulative[-1] = 1.0
                indices = np.searchsorted(cumulative, positions, side="right")
                stochasticity, volatility, mean, variance = [
                    a[indices] for a in (stochasticity, volatility, mean, variance)
                ]
                weights.fill(1 / particles)
                resampling_count += 1
            gain = (variance + volatility) / (variance + volatility + stochasticity)
            mean = mean + gain * (y[trial, block] - mean)
            variance = (1 - gain) * (variance + volatility)
    return result, resampling_count


def particle_objective(
    parameters, outcomes, responses, mask, particles, repeats, seed, released_floor=True
):
    """Average SSE plus its Monte Carlo SEM across repeated particle runs, using training targets only."""
    if repeats < 1 or mask.shape != responses.shape or not mask.any():
        raise ValueError("Invalid Monte Carlo objective or mask.")
    last_trial = np.where(mask.any(axis=1))[0][-1] + 1
    y, target, selected = (
        outcomes[:last_trial],
        responses[:last_trial],
        mask[:last_trial],
    )
    errors = np.array(
        [
            np.square(
                particle_predictions(
                    y, parameters, particles, seed + i, released_floor
                )[0][selected]
                - target[selected]
            ).sum()
            for i in range(repeats)
        ]
    )
    standard_error = errors.std(ddof=1) / np.sqrt(repeats) if repeats > 1 else 0.0
    return float(errors.mean() + standard_error)


def fit_particle(
    outcomes,
    responses,
    mask,
    particles=10000,
    repeats=10,
    evaluations=200,
    seed=0,
    sigma_min=0.05,
    released_floor=True,
):
    """Run a reproducible fixed-budget GP expected-improvement search for the released PF family.

    Common Monte Carlo seeds replace MATLAB's nondeterministic objective calls.
    GP hyperparameters are fixed, and every evaluated candidate is returned.
    Completion of this search is not a guarantee of global convergence.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel

    if evaluations < 5 or repeats < 1 or not 0 < sigma_min < 1:
        raise ValueError("Invalid particle fitting budget.")
    lower, upper = np.array([0.01, 0.01, sigma_min]), np.ones(3)
    rng = np.random.default_rng(seed)
    initial = [
        np.zeros(3),
        np.array([1, 1, 0]),
        np.ones(3),
        rng.random(3),
        rng.random(3),
    ]
    points, values = [], []
    for evaluation in range(evaluations):
        if evaluation < len(initial):
            proposal = initial[evaluation]
        else:
            kernel = Matern(
                length_scale=np.full(3, 0.25), length_scale_bounds="fixed", nu=2.5
            ) + WhiteKernel(noise_level=1e-5, noise_level_bounds="fixed")
            process = GaussianProcessRegressor(
                kernel=kernel, optimizer=None, normalize_y=True, alpha=1e-8
            )
            process.fit(np.asarray(points), np.log1p(np.asarray(values) / mask.sum()))
            candidates = rng.random((2048, 3))
            mean, deviation = process.predict(candidates, return_std=True)
            improvement = (
                np.min(np.log1p(np.asarray(values) / mask.sum())) - mean - 0.001
            )
            z = improvement / np.maximum(deviation, 1e-12)
            expected = improvement * ndtr(z) + deviation * np.exp(
                -0.5 * z**2
            ) / np.sqrt(2 * np.pi)
            proposal = candidates[np.argmax(expected)]
        parameters = lower + proposal * (upper - lower)
        value = particle_objective(
            parameters,
            outcomes,
            responses,
            mask,
            particles,
            repeats,
            seed,
            released_floor,
        )
        points.append(proposal)
        values.append(value)
    best = int(np.argmin(values))
    parameters = lower + points[best] * (upper - lower)
    history = np.column_stack((lower + np.asarray(points) * (upper - lower), values))
    predictions = np.array(
        [
            particle_predictions(
                outcomes, parameters, particles, seed + i, released_floor
            )[0]
            for i in range(repeats)
        ]
    )
    mean_sse = float(
        np.square(predictions[:, mask] - responses[mask]).sum(axis=1).mean()
    )
    fit = PerceptualFit(
        "hierarchical_pf",
        parameters,
        mean_sse,
        True,
        bool(
            np.any(np.isclose(parameters, lower))
            or np.any(np.isclose(parameters, upper))
        ),
    )
    return fit, predictions.mean(axis=0), history


def random_effects_bms(log_evidence, seed=0, samples=200000):
    """Estimate model frequencies and protected exceedance from a common evidence convention.

    Variational Dirichlet updates and the equal-frequency null follow the
    equations implemented in the released cbm_spm_BMS.m. The inputs here are
    BIC approximations, not exact integrated model evidences.
    """
    evidence = np.asarray(log_evidence, dtype=float)
    if evidence.ndim != 2 or evidence.shape[1] < 2 or not np.isfinite(evidence).all():
        raise ValueError("Expected finite subject-by-model log evidence.")
    count = evidence.shape[1]
    centered = evidence - evidence.max(axis=1, keepdims=True)
    prior, alpha = np.ones(count), np.ones(count)
    for _ in range(100000):
        elog = digamma(alpha) - digamma(alpha.sum())
        utility = centered + elog
        responsibility = np.exp(utility - logsumexp(utility, axis=1, keepdims=True))
        new_alpha = prior + responsibility.sum(axis=0)
        if np.linalg.norm(new_alpha - alpha) < 1e-8:
            alpha = new_alpha
            break
        alpha = new_alpha
    else:
        raise RuntimeError("Group BMS failed to converge.")
    elog = digamma(alpha) - digamma(alpha.sum())
    dirichlet_entropy = (
        gammaln(alpha).sum() - gammaln(alpha.sum()) - ((alpha - 1) * elog).sum()
    )
    assignment_entropy = -np.sum(
        responsibility * np.log(np.maximum(responsibility, np.finfo(float).tiny))
    )
    log_prior = gammaln(prior.sum()) - gammaln(prior).sum() + ((prior - 1) * elog).sum()
    free_energy = (
        log_prior
        + np.sum(responsibility * (centered + elog))
        + dirichlet_entropy
        + assignment_entropy
    )
    null = np.sum(logsumexp(centered, axis=1) - np.log(count))
    omnibus_risk = float(expit(null - free_energy))
    rng = np.random.default_rng(seed)
    exceedance = np.zeros(count)
    for start in range(0, samples, 10000):
        draws = rng.dirichlet(alpha, size=min(10000, samples - start))
        exceedance += np.bincount(np.argmax(draws, axis=1), minlength=count)
    exceedance /= samples
    return {
        "frequency": alpha / alpha.sum(),
        "exceedance": exceedance,
        "protected_exceedance": (1 - omnibus_risk) * exceedance + omnibus_risk / count,
        "omnibus_risk": omnibus_risk,
    }
