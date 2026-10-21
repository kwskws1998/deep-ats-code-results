"""Causal likelihood, source-schema, and recovery-interface tests for human behavioral validation."""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat
from scipy.special import ndtr

from ats.choice_models import (
    BinaryNoiseBelief,
    categorical_probabilities,
    choice_design,
    choice_objective,
    fit_choice_model,
    fit_design,
    interval_design,
    block_features,
    threshold_probabilities,
    simulate_observe_participant,
)
from ats.human_data import ObserveBlock, load_observe_blocks, load_bucket_participants
from ats.prediction_models import (
    bucket_log_probabilities,
    perceptual_predictions,
    response_means,
    observed_response_expectation,
    fit_prediction_model,
    predict_bucket_responses,
    simulate_bucket_responses,
)


@pytest.mark.parametrize(
    "mean,sigma", [(-200, 0.25), (10, 0.25), (50, 3), (90, 0.25), (200, 0.25), (60, 30)]
)
def test_bucket_quantized_likelihood_normalizes_in_extreme_tails(mean, sigma):
    responses = np.arange(10, 91, 2, dtype=float)
    logp = bucket_log_probabilities(responses, np.full_like(responses, mean), sigma)
    assert np.isfinite(logp).all()
    assert np.exp(logp).sum() == pytest.approx(1, abs=1e-12)


def test_bucket_interior_bin_matches_manual_cdf():
    value = bucket_log_probabilities(np.array([60.0]), np.array([62.0]), 4)[0]
    assert np.exp(value) == pytest.approx(ndtr((61 - 62) / 4) - ndtr((59 - 62) / 4))


@pytest.mark.parametrize(
    "model,parameter",
    [
        ("joint_grid", None),
        ("volatility_only_grid", None),
        ("stochasticity_only_grid", None),
        ("delta_rule", 0.3),
        ("fitted_kalman", 0),
        ("last_outcome", None),
    ],
)
def test_perceptual_prediction_never_uses_current_or_future_outcomes(model, parameter):
    outcomes = np.tile(np.linspace(40, 80, 20)[:, None], (1, 4))
    changed = outcomes.copy()
    changed[8:] += 100
    first = perceptual_predictions(outcomes, model, parameter)
    second = perceptual_predictions(changed, model, parameter)
    np.testing.assert_array_equal(first[:9], second[:9])
    np.testing.assert_array_equal(first[0], np.full(4, 60))


def test_response_inertia_is_strictly_one_step_causal():
    template = np.full((5, 1), 60.0)
    responses = np.array([[50.0], [52.0], [54.0], [56.0], [58.0]])
    changed = responses.copy()
    changed[2:] += 10
    before = response_means(template, responses, np.array([0.5, 1, 0]))
    after = response_means(template, changed, np.array([0.5, 1, 0]))
    np.testing.assert_array_equal(before[:3], after[:3])
    assert before[1, 0] == 56


def test_delta_filter_matches_hand_calculation():
    predictions = perceptual_predictions(
        np.array([[70.0], [40.0], [90.0]]), "delta_rule", 0.5
    )
    np.testing.assert_allclose(predictions.ravel(), [60, 65, 52.5])


def test_condition_specific_learning_controls_have_independent_rates():
    outcomes = np.full((3, 4), 80.0)
    predictions = perceptual_predictions(
        outcomes, "blockwise_delta", np.array([0.0, 0.25, 0.5, 1.0])
    )
    np.testing.assert_allclose(predictions[1], [60, 65, 70, 80])
    np.testing.assert_allclose(predictions[2], [60, 68.75, 75, 80])


def test_observed_response_expectation_respects_rounding_and_clipping():
    expected = observed_response_expectation(np.array([-100.0, 60.0, 200.0]), 0.25)
    np.testing.assert_allclose(expected, [10, 60, 90], atol=1e-8)


def test_training_fit_cannot_use_later_response_targets():
    outcomes = np.tile(np.linspace(45, 75, 20)[:, None], (1, 4))
    truth = np.array([0.3, 1, np.log(4)])
    responses = simulate_bucket_responses(outcomes, "joint_grid", truth, 12)
    changed = responses.copy()
    changed[12:] = 10
    mask = np.zeros_like(responses, dtype=bool)
    mask[:12] = True
    template = perceptual_predictions(outcomes, "joint_grid")
    first = fit_prediction_model("joint_grid", outcomes, responses, mask, template)
    second = fit_prediction_model("joint_grid", outcomes, changed, mask, template)
    np.testing.assert_array_equal(first.parameters, second.parameters)
    expected, logp = predict_bucket_responses(first, outcomes, responses, template)
    assert np.isfinite(logp).all()
    assert np.all((expected >= 10) & (expected <= 90))


def test_softmax_analytic_gradient_matches_numerical_derivative():
    rng = np.random.default_rng(14)
    design = rng.normal(size=(50, 4, 3))
    choices = rng.integers(0, 4, size=50)
    parameters = np.array([0.7, -0.3, 1.2])
    value, gradient = choice_objective(parameters, design, choices)
    numerical = np.array(
        [
            (
                choice_objective(parameters + np.eye(3)[i] * 1e-5, design, choices)[0]
                - choice_objective(parameters - np.eye(3)[i] * 1e-5, design, choices)[0]
            )
            / 2e-5
            for i in range(3)
        ]
    )
    np.testing.assert_allclose(gradient, numerical, rtol=1e-6, atol=1e-6)
    assert value > 0


def test_choice_probabilities_normalize_and_constant_cost_cancels():
    gains = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.2, 0.1, 0.01]])
    design = interval_design(gains, np.array([1, 2, 4, 8]))
    parameters = np.array([4.0, 0.2])
    first = categorical_probabilities(design, parameters)
    augmented = np.concatenate((design, np.ones((2, 4, 1))), axis=2)
    second = categorical_probabilities(augmented, np.r_[parameters, -10.0])
    np.testing.assert_allclose(first, second, atol=1e-14)
    np.testing.assert_allclose(first.sum(axis=1), 1)


def test_interval_coefficients_can_be_recovered_with_varying_information_contexts():
    rng = np.random.default_rng(17)
    gains = rng.uniform(0, 0.5, size=(6000, 4))
    design = interval_design(gains, np.array([1, 2, 4, 8]))
    truth = np.array([6.0, 0.25])
    probabilities = categorical_probabilities(design, truth)
    choices = np.array([rng.choice(4, p=p) for p in probabilities])
    fit = fit_design("interval", design, choices, ((0, 100), (0, 8)))
    np.testing.assert_allclose(fit.parameters, truth, rtol=0.12, atol=0.03)


def test_binary_belief_update_matches_enumerated_two_state_bayes():
    belief = BinaryNoiseBelief(hazards=(0.1,), reliabilities=(0.8,))
    assert belief.features()[0] == pytest.approx(0)
    belief.advance(1)
    np.testing.assert_allclose(belief.joint, [[0.26, 0.74]])
    assert belief.features()[0] == pytest.approx(2 * (0.74 * 0.8 + 0.26 * 0.2) - 1)
    belief.advance(0)
    np.testing.assert_allclose(belief.joint, [[0.308, 0.692]])


def test_binary_information_nonnegative_and_parameter_information_zero_initially():
    belief = BinaryNoiseBelief()
    assert belief.features()[1] == pytest.approx(0, abs=1e-14)
    assert belief.features()[2] > 0
    for value in (1, -1, 1, 0, 0, -1):
        belief.advance(value)
        assert np.all(belief.features()[1:3] >= 0)
        assert belief.joint.sum() == pytest.approx(1)


def test_choice_features_do_not_use_current_feedback_and_misses_keep_clock():
    choices = np.array([0, 0, -2, 1, 0])
    feedback = np.array([1, -1, 0, 0, 1])
    block = ObserveBlock("s", "train", "b", choices, feedback)
    first = block_features(block)
    changed = feedback.copy()
    changed[1] = 1
    second = block_features(ObserveBlock("s", "train", "b", choices, changed))
    np.testing.assert_array_equal(first[:2], second[:2])
    assert first[3, 0] != first[2, 0]


def test_deadline_resets_in_each_block_and_value_ig_have_matched_parameter_counts():
    feature = np.zeros((50, 5))
    feature[:, 1] = 0.4
    time, _ = choice_design(feature, "bayes_time")
    info, _ = choice_design(feature, "ats_binary_parameter_ig")
    assert time.shape == info.shape == (50, 3, 4)
    np.testing.assert_allclose(time[[0, -1], 1, 3], [1, 0])
    assert info[0, 1, 3] == 0.4


def test_threshold_probabilities_are_normalized_before_feedback():
    choices = np.zeros(50, dtype=int)
    feedback = np.ones(50, dtype=int)
    block = ObserveBlock("s", "train", "b", choices, feedback)
    parameters = np.array([0.1, 3, 0.2, 0.3, 0.5])
    probability = threshold_probabilities(parameters, [block])
    np.testing.assert_allclose(probability.sum(axis=1), 1)
    assert np.all(probability > 0)
    assert probability[0, 0] == pytest.approx(probability[0, 2])
    assert probability[10, 2] > probability[10, 0]


def test_synthetic_choice_feedback_follows_own_actions_and_fits_are_finite():
    blocks = simulate_observe_participant(
        "ats_binary_parameter_ig",
        np.array([6, 0, 0, 12]),
        14,
        train_blocks=2,
        test_blocks=1,
    )
    for block in blocks:
        assert np.all(block.feedback[block.choices != 0] == 0)
        assert np.all(np.abs(block.feedback[block.choices == 0]) == 1)
    fit = fit_choice_model("ats_binary_parameter_ig", blocks[:2])
    assert np.isfinite(fit.training_nll)
    assert len(fit.parameters) == 4


def test_observe_loader_separates_sessions_and_rejects_hidden_outcomes(tmp_path: Path):
    root = tmp_path / "unpacked" / "ObserveBet-pinned" / "TaskData" / "subj1"
    root.mkdir(parents=True)
    data = np.column_stack((np.zeros(50, dtype=int), np.ones(50, dtype=int)))
    np.savetxt(root / "behav.behav001.csv", data, delimiter=",")
    np.savetxt(root / "behav.001.csv", data, delimiter=",")
    blocks, metadata = load_observe_blocks(tmp_path)
    assert {block.session for block in blocks} == {"train", "test"}
    assert metadata["participants"] == 1
    data[0] = [1, 1]
    np.savetxt(root / "behav.001.csv", data, delimiter=",")
    with pytest.raises(ValueError, match="Feedback"):
        load_observe_blocks(tmp_path)


def test_bucket_loader_preserves_condition_alignment_and_omits_demographics(
    tmp_path: Path,
):
    for experiment in (1, 2):
        directory = tmp_path / "unpacked" / f"experiment{experiment}"
        directory.mkdir(parents=True)
        outcomes = np.tile(np.arange(50, dtype=float)[:, None], (1, 4))
        metadata = {
            "bag": outcomes,
            "bird": outcomes,
            "true_vol": np.array([4, 49, 4, 49]),
            "true_sto": np.array([16, 16, 64, 64]),
        }
        participant = {
            "bucket": np.full((50, 4), 60),
            "randomization_order": np.array([3, 1, 2, 0]),
            "age": 99,
            "gender": "unused",
        }
        savemat(
            directory / "data.mat",
            {
                "meta_data": metadata,
                "data": np.array([participant, participant], dtype=object),
            },
        )
    participants, metadata = load_bucket_participants(tmp_path)
    assert len(participants) == 4
    np.testing.assert_array_equal(participants[0].outcomes, outcomes)
    assert not hasattr(participants[0], "age")
    assert metadata["shared_stimuli_across_experiments"]
