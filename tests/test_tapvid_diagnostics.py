"""Independent-query batching, component diagnosis, and interruption fault injection."""

from copy import deepcopy
from dataclasses import replace
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ats.artifacts import verify_run
from ats_vision.sampling import SamplingPolicy
from ats_vision.tap_batching import point_batch_training_loss, query_microbatches
from ats_vision.tap_calibration import position_calibration, visibility_calibration
from ats_vision.tap_checkpoint import load_diagnostic_checkpoint
from ats_vision.tap_data import load_tap_clip
from ats_vision.tap_diagnostics import (
    analyze_traces,
    diagnose_checkpoint,
    full_observation_trace,
)
from ats_vision.tap_metrics import tapvid_metrics
from ats_vision.tap_model import TAPATSModel
from ats_vision.tap_probes import benchmark_queries, overfit_probe
from ats_vision.tap_readout import heatmap_readouts
from ats_vision.tap_rollout import point_rollout, point_training_loss
from ats_vision.tap_training import TAPTrainConfig, train_tap
from test_tapvid import create_manifest, fixture_clip


@pytest.fixture(autouse=True)
def single_thread():
    """Keep generated CPU fixtures fast and deterministic."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def batch_fixture():
    """Use two same-clock queries with different occlusions and observation schedules."""
    clip = fixture_clip(length=8)
    ids, queries = clip.queries()
    selected = np.array([0, 2])
    masks = np.array(
        [
            [True, False, True, False, True, False, True, True],
            [True, True, False, True, False, True, False, True],
        ]
    )
    return clip, ids[selected], queries[selected], masks


def test_batched_losses_gradients_and_encoder_read_counts_match_sequential():
    torch.manual_seed(19)
    sequential = TAPATSModel(8, 8)
    batched = deepcopy(sequential)
    clip, ids, queries, masks = batch_fixture()
    individual = [
        point_training_loss(
            sequential,
            clip.frames,
            query,
            clip.points[index],
            clip.occluded[index],
            mask,
        )
        for index, query, mask in zip(ids, queries, masks)
    ]
    reference = {
        key: torch.stack([row[key] for row in individual]) for key in individual[0]
    }
    calls = []
    handle = batched.encoder.register_forward_pre_hook(
        lambda module, args: calls.append(len(args[0]))
    )
    combined = point_batch_training_loss(
        batched, clip.frames, queries, clip.points[ids], clip.occluded[ids], masks
    )
    handle.remove()
    assert sum(calls) == masks.sum()
    for key in reference:
        torch.testing.assert_close(combined[key], reference[key], atol=2e-5, rtol=2e-5)
    reference["total"].mean().backward()
    combined["total"].mean().backward()
    for (name, first), (_, second) in zip(
        sequential.named_parameters(), batched.named_parameters()
    ):
        assert first.grad is not None, name
        torch.testing.assert_close(
            first.grad, second.grad, atol=5e-5, rtol=3e-4, msg=name
        )


def test_query_batches_do_not_mix_observation_information():
    clip, ids, queries, masks = batch_fixture()
    torch.manual_seed(2)
    model = TAPATSModel(8, 8)
    first = point_batch_training_loss(
        model, clip.frames, queries, clip.points[ids], clip.occluded[ids], masks
    )
    changed = clip.frames.copy()
    changed[1] = 255 - changed[1]
    second = point_batch_training_loss(
        model, changed, queries, clip.points[ids], clip.occluded[ids], masks
    )
    for key in first:
        torch.testing.assert_close(first[key][0], second[key][0], atol=0, rtol=0)


def test_microbatches_preserve_sequential_order_and_group_only_equal_clocks():
    queries = np.array([[2, 0.2, 0.3], [0, 0.2, 0.3], [2, 0.5, 0.6], [0, 0.1, 0.2]])
    assert [int(group[0]) for group in query_microbatches(queries, 1)] == [0, 1, 2, 3]
    groups = query_microbatches(queries, 2)
    assert sorted(np.concatenate(groups).tolist()) == [0, 1, 2, 3]
    assert all(len(set(queries[group, 0])) == 1 for group in groups)


def test_diagnostic_trace_matches_original_full_observation_predictions():
    torch.manual_seed(7)
    model = TAPATSModel(8, 8).eval()
    clip = fixture_clip()
    query = clip.queries()[1][1]
    original = point_rollout(model, clip.frames, query, 1, "uniform")
    diagnostic = full_observation_trace(model, clip.frames, query)
    np.testing.assert_array_equal(original["points"], diagnostic["filtered"])
    start = int(query[0])
    probability = torch.as_tensor(original["visibility_logits"]).sigmoid().numpy()
    np.testing.assert_allclose(
        probability[start:], diagnostic["visibility_probability"][start:], atol=1e-7
    )


def test_readout_diagnosis_identifies_global_mean_between_separate_peaks():
    logits = torch.full((1, 4, 8), -30.0)
    logits[0, 1, 1] = logits[0, 1, 6] = 10
    output = heatmap_readouts(logits, radius=1)
    assert output["global_mean"][0, 0] == pytest.approx(0.5)
    assert output["peak"][0, 0] == pytest.approx(1.5 / 8)
    torch.testing.assert_close(output["local_mean"], output["peak"], atol=1e-6, rtol=0)


def test_diagnostic_oracles_and_horizon_masks_are_explicit():
    clip = fixture_clip(length=8)
    ids, queries = clip.queries()
    ids, queries = ids[:1], queries[:1]
    trace = {
        key: clip.points[ids].copy()
        for key in (
            "filtered",
            "measurement_global",
            "measurement_peak",
            "measurement_local",
            "query_hold",
        )
    }
    trace.update(
        visibility_probability=np.zeros((1, 8)),
        covariance=np.broadcast_to(np.eye(2) * 0.01, (1, 8, 2, 2)).copy(),
    )
    rows, calibration = analyze_traces(clip, ids, queries, trace)
    full = [
        row
        for row in rows
        if row["horizon"] == "all_future" and row["readout"] == "filtered"
    ]
    assert (
        next(row for row in full if row["visibility_rule"] == "predicted_0.5")[
            "average_jaccard"
        ]
        == 0
    )
    assert (
        next(row for row in full if row["visibility_rule"].startswith("oracle"))[
            "average_jaccard"
        ]
        == 1
    )
    assert all(not row["metric_defined"] for row in rows if row["horizon"] == "61_plus")
    assert calibration[0]["position_samples"] == 7


def test_calibration_uses_only_declared_scored_frames():
    mask = np.array([[False, True, True]])
    visible = np.array([[True, True, False]])
    values = visibility_calibration(np.array([[0.0, 0.9, 0.1]]), visible, mask)
    assert values["visibility_brier"] == pytest.approx(0.01)
    assert values["visibility_tp"] == values["visibility_tn"] == 1
    target = np.zeros((1, 3, 2))
    mean = target.copy()
    mean[:, 0] = 1000
    covariance = np.broadcast_to(np.eye(2), (1, 3, 2, 2))
    calibrated = position_calibration(target, mean, covariance, mask)
    assert calibrated["coverage95"] == 1
    assert calibrated["mean_mahalanobis_squared"] == 0


def test_invalid_covariance_is_not_silently_filtered_out():
    with pytest.raises(FloatingPointError, match="semidefinite"):
        position_calibration(
            np.zeros((1, 2)),
            np.zeros((1, 2)),
            np.array([[[-1, 0], [0, 1]]]),
            np.ones(1, bool),
        )


def test_training_probe_does_not_open_final_test_files(tmp_path, monkeypatch):
    import ats_vision.tap_data as data
    from ats_vision.tap_probes import train_probe_sample

    manifest = create_manifest(tmp_path / "data")
    actual = data.sha256

    def no_final_test(path, *args, **kwargs):
        """Reject even checksum reads of a final-test archive during train probes."""
        assert Path(path).name != "test.npz"
        return actual(path, *args, **kwargs)

    monkeypatch.setattr(data, "sha256", no_final_test)
    entry, _, _, _ = train_probe_sample(manifest, 4, 1)
    assert entry["split"] == "train"


def test_optional_metric_mask_preserves_query_exclusion():
    clip = fixture_clip()
    ids, queries = clip.queries()
    mask = np.ones(clip.occluded.shape, bool)
    default = tapvid_metrics(
        queries, clip.points, clip.occluded, clip.points, clip.occluded
    )
    masked = tapvid_metrics(
        queries, clip.points, clip.occluded, clip.points, clip.occluded, mask
    )
    assert default == masked
    mask[:] = False
    mask[:, -1] = True
    masked = tapvid_metrics(
        queries, clip.points, clip.occluded, clip.points, clip.occluded, mask
    )
    assert masked["evaluated_frames"] == len(ids)
    assert masked["average_jaccard"] == 1


def test_forced_acquisitions_do_not_run_variance_search(monkeypatch):
    import ats_vision.sampling as sampling

    model = TAPATSModel(8, 8)
    policy = SamplingPolicy(
        "ats", 8, 8, model.transition.numpy(), model.acceleration_map.numpy()
    )
    monkeypatch.setattr(
        sampling,
        "covariance_step",
        lambda *args: pytest.fail("Forced schedule must not perform candidate search"),
    )
    assert policy.next_time(0, 1, np.eye(4), np.ones(2), np.ones(2)) == (1, {})


def test_best_and_last_checkpoint_remain_consistent_across_write_interruption(
    tmp_path, monkeypatch
):
    import ats_vision.tap_training as training

    data = create_manifest(tmp_path / "data")
    counter = 0

    def scores(*args, **kwargs):
        """Force a better validation key at each chronological selection call."""
        nonlocal counter
        counter += 1
        return dict(
            average_jaccard=counter / 100,
            visible_position_mse=1,
            average_pts_within_thresh=0.2,
            occlusion_accuracy=0.5,
        )

    monkeypatch.setattr(training, "validation_scores", scores)
    base = TAPTrainConfig(
        data,
        tmp_path / "reference",
        steps=3,
        queries_per_step=2,
        query_batch_size=2,
        validation_every=1,
        validation_queries=1,
        validation_clips=1,
        feature_dim=8,
        hidden_size=8,
        threads=1,
    )
    train_tap(base)
    counter = 0
    interrupted = replace(base, output_root=tmp_path / "interrupted")
    actual = training.atomic_checkpoint

    def interrupted_write(path, payload):
        """Fail after model.pt advances but before last.pt commits the same step."""
        if path.name == "last.pt" and payload["step"] == 2:
            raise RuntimeError("fixture checkpoint write interruption")
        return actual(path, payload)

    monkeypatch.setattr(training, "atomic_checkpoint", interrupted_write)
    with pytest.raises(RuntimeError, match="checkpoint write interruption"):
        train_tap(interrupted)
    stale_best = torch.load(interrupted.output_root / "model.pt", weights_only=True)
    durable = torch.load(interrupted.output_root / "last.pt", weights_only=True)
    assert stale_best["selected_step"] == 2
    assert durable["step"] == durable["best_snapshot"]["selected_step"] == 1
    monkeypatch.setattr(training, "atomic_checkpoint", actual)
    counter = 1
    train_tap(interrupted, resume=True)
    first = torch.load(base.output_root / "model.pt", weights_only=True)
    second = torch.load(interrupted.output_root / "model.pt", weights_only=True)
    for key in first["state_dict"]:
        torch.testing.assert_close(
            first["state_dict"][key], second["state_dict"][key], atol=0, rtol=0
        )


def test_checkpoint_diagnostics_and_train_probes_are_separate_verified_artifacts(
    tmp_path,
):
    data = create_manifest(tmp_path / "data")
    config = TAPTrainConfig(
        data,
        tmp_path / "training",
        steps=1,
        queries_per_step=2,
        query_batch_size=2,
        validation_every=1,
        validation_queries=1,
        validation_clips=1,
        feature_dim=8,
        hidden_size=8,
        threads=1,
    )
    train_tap(config)
    diagnostic = diagnose_checkpoint(
        data,
        config.output_root / "last.pt",
        tmp_path / "diagnostic",
        max_clips=1,
        max_queries=1,
        threads=1,
    )
    assert verify_run(diagnostic, check_current_source=True) == []
    protocol = json.loads((diagnostic / "protocol.json").read_text())
    assert not protocol["final_test_accessed"] and not protocol["weights_changed"]
    assert (diagnostic / "validation_overlay.png").exists()
    with (config.output_root / "training.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {
        "loss_coordinate",
        "loss_heatmap",
        "loss_visibility",
        "average_pts_within_thresh",
        "occlusion_accuracy",
        "learning_rate",
    } <= rows[0].keys()
    with pytest.raises(ValueError, match="final test"):
        diagnose_checkpoint(
            data, config.output_root / "model.pt", tmp_path / "forbidden", split="test"
        )
    probe = overfit_probe(
        data,
        tmp_path / "overfit",
        steps=2,
        max_frames=4,
        feature_dim=8,
        hidden_size=8,
        threads=1,
    )
    assert verify_run(probe, check_current_source=True) == []
    benchmark = benchmark_queries(
        data,
        tmp_path / "benchmark",
        batch_size=2,
        repeats=1,
        max_frames=4,
        feature_dim=8,
        hidden_size=8,
        device="cpu",
        threads=1,
    )
    assert verify_run(benchmark, check_current_source=True) == []


def test_diagnostic_loader_refuses_changed_model_operator(tmp_path):
    from ats.artifacts import source_fingerprints
    from ats_vision.tap_data import sha256

    data = create_manifest(tmp_path / "data")
    model = TAPATSModel(8, 8)
    identity = dict(
        data_manifest_sha256=sha256(data),
        sources=source_fingerprints(Path(__file__).resolve().parents[1]),
    )
    identity["sources"]["src/ats_vision/tap_model.py"] = "changed"
    path = tmp_path / "bad.pt"
    torch.save(
        dict(
            identity=identity,
            state_dict=model.state_dict(),
            schema_version=1,
            kind="tap_ats",
            model_config=dict(feature_dim=8, hidden_size=8),
        ),
        path,
    )
    with pytest.raises(ValueError, match="operator source"):
        load_diagnostic_checkpoint(path, data)


def test_atomic_metadata_failure_preserves_last_complete_record(tmp_path, monkeypatch):
    from ats.artifacts import atomic_write_json

    path = tmp_path / "manifest.json"
    atomic_write_json(path, dict(status="running", step=1))

    def fail_replace(*args, **kwargs):
        """Emulate interruption before publication of the next metadata record."""
        raise OSError("fixture rename failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="rename failure"):
        atomic_write_json(path, dict(status="running", step=2))
    assert json.loads(path.read_text())["step"] == 1
    assert sorted(item.name for item in tmp_path.iterdir()) == ["manifest.json"]


def test_completed_download_partial_is_recovered_without_http_request(
    tmp_path, monkeypatch
):
    import hashlib
    import runpy

    project = Path(__file__).resolve().parents[1]
    module = runpy.run_path(str(project / "scripts/download_tapvid_data.py"))
    content = b"complete public archive fixture"
    destination = tmp_path / "scene.tar.gz"
    destination.with_suffix(".gz.part").write_bytes(content)
    monkeypatch.setattr(
        module["urllib"].request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail(
            "A fully verified partial must not send an EOF range request"
        ),
    )
    result = module["download"](
        "https://example.invalid/fixture",
        destination,
        hashlib.sha256(content).hexdigest(),
        "sha256",
    )
    assert destination.read_bytes() == content
    assert result["bytes"] == len(content)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA hardware is unavailable locally"
)
def test_true_query_batch_backward_with_deterministic_cuda():
    from ats_vision.tap_training import configure_runtime

    configure_runtime("cuda", 1, 17)
    clip, ids, queries, masks = batch_fixture()
    model = TAPATSModel(8, 8).cuda()
    terms = point_batch_training_loss(
        model, clip.frames, queries, clip.points[ids], clip.occluded[ids], masks
    )
    terms["total"].mean().backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
