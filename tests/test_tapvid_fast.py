"""Shared-feature equivalence, acquisition causality, SPD algebra, and execution forks."""

from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from ats.artifacts import verify_run
from ats_vision.model import gaussian_nll
from ats_vision.tap_batching import point_batch_training_loss
from ats_vision.tap_data import sha256
from ats_vision.tap_fast_math import gaussian_nll2, solve_spd2
from ats_vision.tap_fast_model import SharedTAPATSModel
from ats_vision.tap_fast_rollout import shared_point_rollouts
from ats_vision.tap_model import TAPATSModel
from ats_vision.tap_rollout import point_rollout, point_training_loss
from ats_vision.tap_training import (
    TAPTrainConfig,
    train_tap,
    load_tap_checkpoint,
    configure_runtime,
)
from test_tapvid import create_manifest, fixture_clip
from test_tapvid_diagnostics import batch_fixture


@pytest.fixture(autouse=True)
def bounded_threads():
    """Keep tiny fixtures independent of host CPU core count."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def paired_models(device="cpu"):
    """Use identical parameters without relying on matching constructor RNG draws."""
    torch.manual_seed(19)
    original = TAPATSModel(8, 8).to(device)
    shared = SharedTAPATSModel(8, 8).to(device)
    shared.load_state_dict(original.state_dict())
    return original, shared


def test_spd_algebra_values_and_gradients_match_reference():
    torch.manual_seed(4)
    raw = torch.randn(6, 2, 2, dtype=torch.float64)
    covariance = (raw @ raw.transpose(-1, -2) + torch.eye(2) * 0.05).requires_grad_()
    rhs = torch.randn(6, 2, 4, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(
        solve_spd2(covariance, rhs), torch.linalg.solve(covariance, rhs)
    )
    target = torch.randn(6, 2, dtype=torch.float64)
    mean = torch.randn(6, 2, dtype=torch.float64, requires_grad=True)
    first = gaussian_nll(target, mean, covariance)
    second = gaussian_nll2(target, mean, covariance)
    torch.testing.assert_close(first, second)
    for reference, accelerated in zip(
        torch.autograd.grad(first.sum(), (mean, covariance), retain_graph=True),
        torch.autograd.grad(second.sum(), (mean, covariance)),
    ):
        torch.testing.assert_close(reference, accelerated, atol=1e-10, rtol=1e-10)
    assert torch.autograd.gradcheck(
        lambda matrix, right: solve_spd2(
            matrix @ matrix.transpose(-1, -2) + torch.eye(2) * 0.1, right
        ),
        (raw.requires_grad_(), rhs),
    )


def test_invalid_spd_is_not_clipped_into_a_valid_objective():
    covariance = torch.tensor([[[-1.0, 0.0], [0.0, 1.0]]])
    assert not torch.isfinite(
        gaussian_nll2(torch.zeros(1, 2), torch.zeros(1, 2), covariance)
    ).all()


@pytest.mark.parametrize("chunk", [1, 3, 8])
def test_shared_loss_and_gradients_preserve_reference_and_encode_union_once(chunk):
    original, shared = paired_models()
    clip, ids, queries, masks = batch_fixture()
    rows = [
        point_training_loss(
            original, clip.frames, query, clip.points[index], clip.occluded[index], mask
        )
        for index, query, mask in zip(ids, queries, masks)
    ]
    counts = []
    handle = shared.encoder.register_forward_pre_hook(
        lambda module, args: counts.append(len(args[0]))
    )
    terms = point_batch_training_loss(
        shared,
        clip.frames,
        queries,
        clip.points[ids],
        clip.occluded[ids],
        masks,
        shared_features=True,
        encoder_chunk_size=chunk,
    )
    handle.remove()
    assert sum(counts) == masks.any(0).sum()
    assert max(counts) <= chunk
    assert sum(counts) < masks.sum()
    for key in terms:
        torch.testing.assert_close(
            terms[key], torch.stack([row[key] for row in rows]), atol=2e-5, rtol=3e-5
        )
    torch.stack([row["total"] for row in rows]).mean().backward()
    terms["total"].mean().backward()
    for (name, parameter), (_, other) in zip(
        original.named_parameters(), shared.named_parameters()
    ):
        torch.testing.assert_close(
            parameter.grad, other.grad, atol=5e-5, rtol=4e-4, msg=name
        )


def test_shared_training_cannot_read_another_querys_private_frame():
    _, shared = paired_models()
    clip, ids, queries, masks = batch_fixture()
    terms = point_batch_training_loss(
        shared,
        clip.frames,
        queries,
        clip.points[ids],
        clip.occluded[ids],
        masks,
        shared_features=True,
    )
    changed = clip.frames.copy()
    changed[1] = 255 - changed[1]
    other = point_batch_training_loss(
        shared,
        changed,
        queries,
        clip.points[ids],
        clip.occluded[ids],
        masks,
        shared_features=True,
    )
    for key in terms:
        torch.testing.assert_close(terms[key][0], other[key][0], atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize(
    "policy",
    ["uniform", "random", "ats", "ats_fixed_q", "ats_fixed_r", "total_uncertainty"],
)
@pytest.mark.parametrize("fraction", [0.25, 0.5, 1.0])
def test_batched_inference_preserves_masks_predictions_noise_and_decisions(
    policy, fraction
):
    original, shared = paired_models()
    clip = fixture_clip(length=12)
    queries = clip.queries()[1]
    options = dict(fixed_q=np.array([0.0001, 0.0002]), fixed_r=np.array([0.002, 0.003]))
    before = [
        point_rollout(
            original, clip.frames, query, fraction, policy, seed=17 + i, **options
        )
        for i, query in enumerate(queries)
    ]
    after = shared_point_rollouts(
        shared,
        clip.frames,
        queries,
        fraction,
        policy,
        seeds=[17, 18, 19],
        query_batch_size=2,
        encoder_chunk_size=3,
        **options,
    )
    for first, second in zip(before, after):
        np.testing.assert_array_equal(first["acquired"], second["acquired"])
        assert first["observations"] == second["observations"]
        for key in (
            "points",
            "visibility_logits",
            "covariance",
            "q_reads",
            "r_reads",
            "q",
            "last_r",
        ):
            np.testing.assert_allclose(
                first[key], second[key], atol=2e-6, rtol=5e-5, err_msg=key
            )
        assert [(row["time"], row["next_time"]) for row in first["decisions"]] == [
            (row["time"], row["next_time"]) for row in second["decisions"]
        ]
        for a, b in zip(first["decisions"], second["decisions"]):
            assert a["scores"].keys() == b["scores"].keys()
            np.testing.assert_allclose(
                list(a["scores"].values()),
                list(b["scores"].values()),
                rtol=1e-5,
                atol=1e-7,
            )


def test_full_read_validation_encodes_each_frame_once_for_multiple_query_groups():
    _, shared = paired_models()
    clip = fixture_clip(length=12)
    counts = []
    handle = shared.encoder.register_forward_pre_hook(
        lambda module, args: counts.append(len(args[0]))
    )
    shared_point_rollouts(
        shared,
        clip.frames,
        clip.queries()[1],
        1,
        query_batch_size=1,
        encoder_chunk_size=5,
    )
    handle.remove()
    assert counts == [5, 5, 2]


def test_static_preencoding_does_not_leak_future_images_into_earlier_predictions():
    _, shared = paired_models()
    clip = fixture_clip(length=12)
    queries = clip.queries()[1]
    first = shared_point_rollouts(shared, clip.frames, queries, 1)
    changed = clip.frames.copy()
    changed[8:] = 255 - changed[8:]
    second = shared_point_rollouts(shared, changed, queries, 1)
    for before, after in zip(first, second):
        np.testing.assert_array_equal(before["points"][:8], after["points"][:8])


def test_shared_inference_rejects_nonfinite_covariance_at_boundary():
    _, shared = paired_models()
    with torch.no_grad():
        shared.process_noise_head.weight.fill_(float("nan"))
    clip = fixture_clip()
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        shared_point_rollouts(shared, clip.frames, clip.queries()[1], 1)


def test_shared_paths_do_not_call_synchronous_linalg(monkeypatch):
    _, shared = paired_models()
    clip, ids, queries, masks = batch_fixture()

    def forbidden(*args, **kwargs):
        """Fail if a frame reintroduces the synchronous reference matrix operations."""
        pytest.fail("Shared execution called synchronous linalg")

    monkeypatch.setattr(torch.linalg, "cholesky", forbidden)
    monkeypatch.setattr(torch.linalg, "solve", forbidden)
    terms = point_batch_training_loss(
        shared,
        clip.frames,
        queries,
        clip.points[ids],
        clip.occluded[ids],
        masks,
        shared_features=True,
    )
    terms["total"].mean().backward()
    shared_point_rollouts(shared, clip.frames, queries, 1)


def test_execution_fork_keeps_step_optimizer_rng_and_original_files(
    tmp_path, monkeypatch
):
    import ats_vision.tap_training as training

    data = create_manifest(tmp_path / "data")
    parent = TAPTrainConfig(
        data,
        tmp_path / "parent",
        steps=3,
        queries_per_step=2,
        validation_every=1,
        validation_queries=2,
        validation_clips=1,
        feature_dim=8,
        hidden_size=8,
        threads=1,
        validation_observation_fraction=1,
    )
    real = training.point_training_loss
    calls = 0

    def interrupt(*args, **kwargs):
        """Stop after exactly one completed optimizer update."""
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("fixture stop")
        return real(*args, **kwargs)

    monkeypatch.setattr(training, "point_training_loss", interrupt)
    with pytest.raises(RuntimeError, match="fixture stop"):
        train_tap(parent)
    monkeypatch.setattr(training, "point_training_loss", real)
    legacy = torch.load(parent.output_root / "last.pt", weights_only=True)
    for key in ("kind", "model_config", "best_snapshot"):
        legacy.pop(key)
    for row in legacy["logs"]:
        for key in list(row):
            if key.startswith("loss") or key in {
                "execution_backend",
                "training_queries",
            }:
                row.pop(key)
    torch.save(legacy, parent.output_root / "last.pt")
    old = {
        path.name: sha256(path)
        for path in parent.output_root.iterdir()
        if path.is_file()
    }
    child = replace(
        parent,
        output_root=tmp_path / "child",
        execution_backend="shared",
        query_batch_size=2,
        encoder_chunk_size=3,
    )
    train_tap(child, fork_from=parent.output_root / "last.pt")
    record = json.loads((child.output_root / "training_fork.json").read_text())
    assert record["inherited_step"] == 1
    assert record["inherited_optimizer_and_rng"]
    assert not record["exact_replay"]
    for name, digest in old.items():
        assert sha256(parent.output_root / name) == digest
    saved = torch.load(child.output_root / "last.pt", weights_only=True)
    assert saved["step"] == 3
    assert all(
        int(value["step"]) == 3 for value in saved["optimizer"]["state"].values()
    )
    assert [row["step"] for row in saved["logs"]] == [1, 2, 3]
    model, _ = load_tap_checkpoint(child.output_root / "model.pt")
    assert isinstance(model, SharedTAPATSModel)
    assert verify_run(child.output_root, check_current_source=True) == []
    with pytest.raises(ValueError, match="scientific"):
        train_tap(
            replace(child, output_root=tmp_path / "bad", learning_rate=0.03),
            fork_from=parent.output_root / "last.pt",
        )


def test_shared_exact_resume_is_bitwise_equal_to_uninterrupted_run(
    tmp_path, monkeypatch
):
    import ats_vision.tap_training as training

    data = create_manifest(tmp_path / "data")
    config = TAPTrainConfig(
        data,
        tmp_path / "reference",
        steps=3,
        queries_per_step=2,
        query_batch_size=2,
        execution_backend="shared",
        validation_every=1,
        validation_queries=2,
        validation_clips=1,
        feature_dim=8,
        hidden_size=8,
        threads=1,
    )
    train_tap(config)
    interrupted = replace(config, output_root=tmp_path / "interrupted")
    actual = training.validation_scores
    calls = 0

    def fail_second_validation(*args, **kwargs):
        """Interrupt after an unsaved update, forcing replay from the durable state."""
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("fixture interruption")
        return actual(*args, **kwargs)

    monkeypatch.setattr(training, "validation_scores", fail_second_validation)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        train_tap(interrupted)
    monkeypatch.setattr(training, "validation_scores", actual)
    train_tap(interrupted, resume=True)
    first = torch.load(config.output_root / "last.pt", weights_only=True)
    second = torch.load(interrupted.output_root / "last.pt", weights_only=True)
    for key in first["state_dict"]:
        torch.testing.assert_close(
            first["state_dict"][key], second["state_dict"][key], atol=0, rtol=0
        )
    assert first["logs"] == second["logs"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware required")
def test_shared_cuda_backward_matches_reference_and_is_finite():
    configure_runtime("cuda", 1, 19)
    original, shared = paired_models("cuda")
    clip, ids, queries, masks = batch_fixture()
    torch.use_deterministic_algorithms(True)
    rows = [
        point_training_loss(
            original, clip.frames, query, clip.points[index], clip.occluded[index], mask
        )
        for index, query, mask in zip(ids, queries, masks)
    ]
    terms = point_batch_training_loss(
        shared,
        clip.frames,
        queries,
        clip.points[ids],
        clip.occluded[ids],
        masks,
        shared_features=True,
    )
    reference = torch.stack([row["total"] for row in rows]).mean()
    torch.testing.assert_close(terms["total"].mean(), reference, atol=1e-4, rtol=1e-4)
    reference.backward()
    terms["total"].mean().backward()
    for (name, parameter), (_, other) in zip(
        original.named_parameters(), shared.named_parameters()
    ):
        assert torch.isfinite(other.grad).all(), name
        torch.testing.assert_close(
            parameter.grad, other.grad, atol=1e-3, rtol=3e-3, msg=name
        )
