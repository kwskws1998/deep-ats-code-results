"""Analytical filtering, acquisition causality, gradients, and full pilot integrity."""

from dataclasses import replace
import csv
import json

import numpy as np
import pytest
import torch

from ats.artifacts import verify_run
from ats_vision.data import load_data, moving_blobs, validate_splits
from ats_vision.experiment import (
    VisionConfig,
    load_checkpoint,
    rollout,
    run_experiment,
    score_rollout,
    sequence_loss,
    training_mask,
)
from ats_vision.model import Belief, VideoATSModel, gaussian_nll
from ats_vision.sampling import (
    SamplingPolicy,
    covariance_step,
    covariance_update,
    uniform_times,
)


@pytest.fixture(autouse=True)
def bounded_threads():
    """Keep small matrix tests from spawning a large BLAS thread pool."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_joseph_update_matches_analytical_gaussian_and_downweights_noisy_reading():
    model = VideoATSModel(image_size=12, hidden_size=8)
    belief = Belief(torch.zeros(1, 4), torch.eye(4)[None], torch.zeros(1, 8))
    measurement = torch.ones(1, 2)
    precise = model.update(belief, measurement, torch.full((1, 2), 0.25), torch.ones(1))
    noisy = model.update(belief, measurement, torch.full((1, 2), 4.0), torch.ones(1))
    torch.testing.assert_close(precise.mean[0, :2], torch.full((2,), 0.8))
    torch.testing.assert_close(precise.covariance[0, :2, :2], 0.2 * torch.eye(2))
    torch.testing.assert_close(noisy.mean[0, :2], torch.full((2,), 0.2))
    assert torch.linalg.eigvalsh(precise.covariance).min() > 0


def test_physical_propagation_and_planning_covariance_agree():
    model = VideoATSModel(image_size=12, hidden_size=8)
    belief = model.initial_belief(1)
    q = model.process_variance(belief).detach().numpy()[0]
    predicted, _ = model.predict(belief)
    expected = covariance_step(
        belief.covariance.numpy()[0],
        model.transition.numpy(),
        model.acceleration_map.numpy(),
        q,
    )
    np.testing.assert_allclose(
        predicted.covariance.detach().numpy()[0], expected, atol=1e-7
    )
    variance = np.array([0.02, 0.04], np.float32)
    updated = model.update(
        predicted, torch.zeros(1, 2), torch.from_numpy(variance)[None], torch.ones(1)
    )
    np.testing.assert_allclose(
        updated.covariance.detach().numpy()[0],
        covariance_update(expected, variance),
        atol=1e-7,
    )


def test_nll_is_joint_two_dimensional_density():
    value = gaussian_nll(torch.zeros(1, 2), torch.zeros(1, 2), torch.eye(2)[None])
    assert float(value) == pytest.approx(np.log(2 * np.pi), abs=1e-6)


def test_gradients_reach_encoder_decoder_context_and_both_noise_heads():
    torch.manual_seed(5)
    model = VideoATSModel(image_size=12, hidden_size=8)
    data = moving_blobs(3, length=6, image_size=12)
    mask = torch.tensor([[True, False, True, False, False, True]] * 3)
    losses = sequence_loss(
        model,
        torch.from_numpy(data.frames),
        torch.from_numpy(data.clean_frames),
        torch.from_numpy(data.states),
        mask,
    )
    losses["total"].backward()
    for module in (
        model.encoder,
        model.decoder,
        model.context_update,
        model.process_noise_head,
        model.measurement_noise_head,
    ):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(
            value is not None and torch.isfinite(value).all() for value in gradients
        )
        assert sum(float(value.abs().sum()) for value in gradients) > 0


@pytest.mark.parametrize(
    "policy",
    ["uniform", "random", "total_uncertainty", "ats", "ats_fixed_q", "ats_fixed_r"],
)
@pytest.mark.parametrize("budget", [1, 3, 7])
def test_every_scheduler_honors_exact_budget_and_encoder_only_sees_acquired_frames(
    policy, budget
):
    model = VideoATSModel(image_size=12, hidden_size=8)
    data = moving_blobs(1, length=7, image_size=12)
    counts = []
    handle = model.encoder.register_forward_pre_hook(
        lambda module, args: counts.append(args[0].shape[0])
    )
    prediction = rollout(
        model,
        data.frames[0],
        policy,
        budget,
        9,
        3,
        fixed_q=np.full(2, 1e-5),
        fixed_r=np.full(2, 0.01),
    )
    handle.remove()
    assert sum(counts) == budget
    assert prediction["acquired"].sum() == budget
    assert prediction["acquired"][0]
    assert np.linalg.eigvalsh(prediction["covariance"]).min() > 0
    assert np.isfinite(prediction["mean"]).all()


@pytest.mark.parametrize("policy", ["uniform", "total_uncertainty", "ats"])
def test_unread_images_can_be_destroyed_without_changing_predictions_or_schedule(
    policy,
):
    model = VideoATSModel(image_size=12, hidden_size=8)
    frames = moving_blobs(1, length=10, image_size=12).frames[0]
    first = rollout(model, frames, policy, 3, 2, 5)
    destroyed = frames.copy()
    destroyed[~first["acquired"]] = np.nan
    second = rollout(model, destroyed, policy, 3, 2, 5)
    np.testing.assert_array_equal(first["mean"], second["mean"])
    np.testing.assert_array_equal(first["acquired"], second["acquired"])
    assert first["decisions"] == second["decisions"]


def test_future_reading_cannot_change_past_predictions():
    model = VideoATSModel(image_size=12, hidden_size=8)
    frames = moving_blobs(1, length=8, image_size=12).frames[0]
    first = rollout(model, frames, "uniform", 2, 2, 5)
    changed = frames.copy()
    changed[4:] = 1 - changed[4:]
    second = rollout(model, changed, "uniform", 2, 2, 5)
    np.testing.assert_array_equal(first["mean"][:4], second["mean"][:4])


def test_single_remaining_observation_scores_match_independent_gaussian_conditioning():
    model = VideoATSModel(image_size=12, hidden_size=8)
    transition, g = model.transition.numpy().astype(
        float
    ), model.acceleration_map.numpy().astype(float)
    covariance = np.diag([0.03, 0.02, 0.01, 0.02])
    q, r = np.array([1e-3, 2e-3]), np.array([0.02, 0.04])
    policy = SamplingPolicy("ats", 8, 2, transition, g, max_gap=7)
    selected, scores = policy.next_time(0, 1, covariance, q, r)
    expected = {}
    for read in range(1, 8):
        p = covariance.copy()
        risk = 0.0
        for clock in range(1, 8):
            p = transition @ p @ transition.T + g @ np.diag(q) @ g.T
            if clock == read:
                p = p - p[:, :2] @ np.linalg.solve(p[:2, :2] + np.diag(r), p[:2, :])
            risk += np.trace(p[:2, :2])
        expected[read] = risk
    np.testing.assert_allclose(
        list(scores.values()), list(expected.values()), rtol=1e-10
    )
    assert selected == min(expected, key=expected.get)


def test_dataset_split_identity_and_portable_roundtrip(tmp_path):
    train = moving_blobs(4, length=5, image_size=12, seed=1)
    test = moving_blobs(4, length=5, image_size=12, seed=2, prefix="test")
    validate_splits(dict(train=train, test=test))
    train.save(tmp_path / "train.npz")
    restored = load_data(tmp_path / "train.npz")
    np.testing.assert_array_equal(restored.frames, train.frames)
    with pytest.raises(ValueError, match="Repeated"):
        validate_splits(dict(train=train, test=restored))
    duplicate = moving_blobs(
        4, length=5, image_size=12, seed=1, prefix="different_name"
    )
    with pytest.raises(ValueError, match="Repeated"):
        validate_splits(dict(train=train, test=duplicate))


def test_equal_total_predictive_uncertainty_can_produce_different_qr_aware_schedules():
    model = VideoATSModel(image_size=12, hidden_size=8)
    policy = SamplingPolicy(
        "ats",
        24,
        4,
        model.transition.numpy().astype(float),
        model.acceleration_map.numpy().astype(float),
        max_gap=12,
    )
    covariance = np.diag([0.001, 0.001, 0.0001, 0.0001])
    fast_q, precise_r = np.full(2, 0.04), np.full(2, 0.0001)
    slow_q, noisy_r = np.full(2, 0.0004), np.full(2, 0.01)
    first = covariance_step(
        covariance, policy.transition, policy.acceleration_map, fast_q
    )
    second = covariance_step(
        covariance, policy.transition, policy.acceleration_map, slow_q
    )
    assert np.trace(first[:2, :2]) + precise_r.sum() == pytest.approx(
        np.trace(second[:2, :2]) + noisy_r.sum()
    )
    fast_time, _ = policy.next_time(0, 1, covariance, fast_q, precise_r)
    slow_time, _ = policy.next_time(0, 1, covariance, slow_q, noisy_r)
    assert fast_time != slow_time


def test_invalid_inputs_fail_explicitly():
    with pytest.raises(ValueError):
        uniform_times(5, 6)
    with pytest.raises(ValueError):
        VisionConfig(observation_budget=100).validate()
    with pytest.raises(ValueError):
        VisionConfig(learning_rate=float("nan")).validate()
    data = moving_blobs(1, length=5, image_size=12)
    data.frames = data.frames.astype(np.float64)
    with pytest.raises(ValueError, match="float32"):
        data.validate()
    model = VideoATSModel(image_size=12, hidden_size=8)
    with pytest.raises(ValueError, match="Fixed noise"):
        SamplingPolicy(
            "ats_fixed_q",
            5,
            2,
            model.transition.numpy(),
            model.acceleration_map.numpy(),
            fixed_q=np.array([-1.0, 1.0]),
        )


def test_random_training_masks_are_prospective_and_include_first_frame():
    first = training_mask(8, 12, 4, np.random.default_rng(2))
    second = training_mask(8, 12, 4, np.random.default_rng(2))
    np.testing.assert_array_equal(first, second)
    assert first[:, 0].all()
    assert ((first.sum(1) >= 2) & (first.sum(1) <= 8)).all()


def test_all_frame_reference_and_full_budget_uniform_are_identical():
    model = VideoATSModel(image_size=12, hidden_size=8)
    frames = moving_blobs(1, length=5, image_size=12).frames[0]
    reference = rollout(model, frames, "all_frames", 5, 1, 3)
    comparison = rollout(model, frames, "uniform", 5, 1, 3)
    np.testing.assert_array_equal(reference["mean"], comparison["mean"])


def test_full_pilot_checkpoint_metrics_and_manifest_are_reproducible(tmp_path):
    config = VisionConfig(
        output_root=tmp_path / "run",
        train_clips=4,
        validation_clips=4,
        test_clips=4,
        sequence_length=6,
        image_size=12,
        observation_budget=2,
        hidden_size=8,
        warmup_epochs=1,
        epochs=1,
        batch_size=4,
        threads=1,
        max_gap=4,
    )
    output = run_experiment(config)
    assert verify_run(output, check_current_source=True) == []
    model = load_checkpoint(output / "model.pt")
    test = load_data(output / "data/test.npz")
    protocol = json.loads((output / "protocol.json").read_text())
    with (output / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 7 * 4
    for row in rows:
        assert int(row["observations"]) == (6 if row["policy"] == "all_frames" else 2)
    first = next(
        row
        for row in rows
        if row["policy"] == "ats" and row["clip_id"] == test.clip_ids[0]
    )
    repeated = rollout(
        model,
        test.frames[0],
        "ats",
        2,
        config.seed + 20000,
        config.max_gap,
        protocol["total_uncertainty_threshold"],
        np.array(protocol["fixed_q"]),
        np.array(protocol["fixed_r"]),
    )
    scored = score_rollout(repeated, test.states[0], test.clean_frames[0])
    assert float(first["position_mse"]) == pytest.approx(
        scored["position_mse"], abs=1e-10
    )
    with np.load(output / "predictions.npz") as predictions:
        np.testing.assert_array_equal(predictions["ats__mean"][0], repeated["mean"])
        assert predictions["ats__acquired"].sum(axis=1).tolist() == [2] * 4
    with pytest.raises(FileExistsError):
        run_experiment(config)
    assert verify_run(output) == []
    second_output = run_experiment(replace(config, output_root=tmp_path / "replicate"))
    second_model = load_checkpoint(second_output / "model.pt")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(
            value, second_model.state_dict()[key], atol=0, rtol=0
        )
