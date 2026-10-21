"""Independent causal, derivative, likelihood-convention, and group-comparison checks."""

import numpy as np
import pytest

from ats.piray_models import (
    deterministic_predictions,
    fit_deterministic,
    fit_particle,
    fit_response_sigma,
    particle_predictions,
    particle_objective,
    random_effects_bms,
    unit_gaussian_bic,
)


def test_paper_kalman_initial_variance_matches_hand_calculation():
    y = np.full((3, 4), 72.0)
    mean, _ = deterministic_predictions(y, "kalman4", np.ones(4))
    np.testing.assert_allclose(mean[:, 0], [60, 68, 70.5])


@pytest.mark.parametrize("model,parameter", [("rw4", 0.3), ("kalman4", 2.5)])
def test_prediction_gradients_match_independent_finite_difference(model, parameter):
    y = np.random.default_rng(16).normal(60, 8, size=(16, 4))
    theta = np.full(4, parameter)
    _, derivative = deterministic_predictions(y, model, theta)
    for block in range(4):
        step = np.zeros(4)
        step[block] = 1e-5
        plus, _ = deterministic_predictions(y, model, theta + step)
        minus, _ = deterministic_predictions(y, model, theta - step)
        np.testing.assert_allclose(
            (plus[:, block] - minus[:, block]) / 2e-5,
            derivative[:, block],
            atol=1e-8,
            rtol=1e-7,
        )


@pytest.mark.parametrize("model", ["rw4", "kalman4"])
def test_perceptual_fit_does_not_read_heldout_response_labels(model):
    y = np.random.default_rng(17).normal(60, 8, (20, 4))
    r = np.clip(2 * np.round(y / 2), 10, 90)
    other = r.copy()
    other[12:] = 10
    mask = np.zeros_like(r, dtype=bool)
    mask[:12] = True
    a = fit_deterministic(model, y, r, mask, starts=3)
    b = fit_deterministic(model, y, other, mask, starts=3)
    np.testing.assert_array_equal(a.parameters, b.parameters)


def test_paper_unit_gaussian_bic_has_one_constant_per_observation():
    assert unit_gaussian_bic(123, 200, 3) == pytest.approx(
        123 + 200 * np.log(2 * np.pi) + 3 * np.log(200)
    )
    assert unit_gaussian_bic(123, 200, 4) - unit_gaussian_bic(
        123, 200, 3
    ) == pytest.approx(np.log(200))


def test_particle_model_with_no_parameter_transitions_matches_kalman_recursion():
    y = np.random.default_rng(28).normal(60, 8, (12, 4))
    actual, _ = particle_predictions(y, [0, 0, 0.5], 64, 3)
    mean, variance = np.full(4, 60.0), np.full(4, 10.0)
    expected = np.empty_like(y)
    for t in range(len(y)):
        expected[t] = mean
        gain = (variance + 1) / (variance + 2)
        mean = mean + gain * (y[t] - mean)
        variance = (1 - gain) * (variance + 1)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def test_particle_model_causality_and_independent_block_randomness():
    y = np.random.default_rng(11).normal(60, 8, (15, 4))
    changed = y.copy()
    changed[8:, 0] += 40
    first, _ = particle_predictions(y, [0.4, 0.6, 0.5], 100, 5)
    second, _ = particle_predictions(changed, [0.4, 0.6, 0.5], 100, 5)
    np.testing.assert_array_equal(first[:9, 0], second[:9, 0])
    np.testing.assert_array_equal(first[:, 1:], second[:, 1:])
    np.testing.assert_allclose(first[0], np.full(4, 60), atol=1e-12, rtol=0)


def test_particle_training_objective_cannot_read_future_outcomes_or_responses():
    y = np.random.default_rng(40).normal(60, 8, (15, 4))
    r = 2 * np.round(y / 2)
    yy, rr = y.copy(), r.copy()
    yy[8:], rr[8:] = 20, 10
    mask = np.zeros_like(y, dtype=bool)
    mask[:8] = True
    a = particle_objective([0.4, 0.6, 0.5], y, r, mask, 60, 2, 31)
    b = particle_objective([0.4, 0.6, 0.5], yy, rr, mask, 60, 2, 31)
    assert a == b


def test_response_sigma_fitting_uses_training_targets_only():
    mean = np.full((20, 4), 60.0)
    r = np.random.default_rng(9).choice(np.arange(48, 73, 2), size=mean.shape)
    other = r.copy()
    other[10:] = 10
    mask = np.zeros_like(mean, dtype=bool)
    mask[:10] = True
    assert fit_response_sigma(r, mean, mask) == fit_response_sigma(other, mean, mask)


def test_gp_particle_fit_is_reproducible_and_cannot_use_test_labels():
    y = np.random.default_rng(37).normal(60, 8, (12, 4))
    r = np.clip(2 * np.round(y / 2), 10, 90)
    other = r.copy()
    other[8:] = 10
    mask = np.zeros_like(y, dtype=bool)
    mask[:8] = True
    first, _, trace_first = fit_particle(
        y, r, mask, particles=40, repeats=2, evaluations=7
    )
    second, _, trace_second = fit_particle(
        y, other, mask, particles=40, repeats=2, evaluations=7
    )
    np.testing.assert_array_equal(first.parameters, second.parameters)
    np.testing.assert_array_equal(trace_first, trace_second)
    assert len(trace_first) == 7


def test_group_bms_is_invariant_to_subject_specific_likelihood_constants():
    evidence = np.random.default_rng(7).normal(size=(30, 3))
    shifted = evidence + np.arange(30)[:, None] * 20
    a = random_effects_bms(evidence, seed=40, samples=10000)
    b = random_effects_bms(shifted, seed=40, samples=10000)
    for name in a:
        np.testing.assert_allclose(a[name], b[name], atol=1e-10, rtol=0)


def test_group_bms_protects_symmetric_null_and_detects_strong_winner():
    null = random_effects_bms(np.zeros((80, 3)), samples=10000)
    np.testing.assert_allclose(null["frequency"], np.full(3, 1 / 3), atol=1e-12)
    assert null["omnibus_risk"] > 0.9
    evidence = np.zeros((80, 3))
    evidence[:, 1] = 30
    alternative = random_effects_bms(evidence, samples=10000)
    assert alternative["frequency"][1] > 0.97
    assert alternative["protected_exceedance"][1] > 0.99
