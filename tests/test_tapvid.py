"""TAP metric parity, public-query causality, source isolation, and train/test replay."""

from dataclasses import replace
import ast
import csv
import io
import json
from pathlib import Path
import pickle
import tarfile
import types
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch

from ats.artifacts import verify_run
from ats_vision.tap_data import (
    ArrayUnpickler,
    TapClip,
    load_tap_clip,
    load_tap_manifest,
    prepare_tap_data,
    read_kubric_archive,
    sha256,
)
from ats_vision.tap_metrics import tapvid_metrics
from ats_vision.tap_model import TAPATSModel
from ats_vision.tap_rollout import point_rollout, point_training_loss
from ats_vision.tap_training import (
    TAPTrainConfig,
    train_tap,
    evaluate_tap,
    load_tap_checkpoint,
)
from ats_vision.tap_baselines import baseline_rollout, evaluate_baseline


@pytest.fixture(autouse=True)
def one_thread():
    """Keep tiny matrix fixtures bounded on shared CPU machines."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def fixture_clip(source="scene0", dataset="cotracker3_kubric", length=8):
    """Generate explicit test-only point/occlusion arrays, never a research dataset."""
    import hashlib

    seed = int(hashlib.sha256(source.encode()).hexdigest()[:8], 16)
    frames = np.random.default_rng(seed).integers(
        0, 256, (length, 16, 16, 3), dtype=np.uint8
    )
    points = np.full((3, length, 2), 0.4, np.float32)
    points[0, :, 0] = np.linspace(0.2, 0.6, length)
    points[1, :, 1] = np.linspace(0.6, 0.3, length)
    occluded = np.zeros((3, length), bool)
    occluded[1, :2] = True
    occluded[2, 3:5] = True
    return TapClip(frames, points, occluded, source, dataset)


def create_manifest(root):
    """Keep synthetic fixtures in distinct source roles for end-to-end testing."""
    root.mkdir()
    entries = []
    for offset, split in enumerate(("train", "validation", "test")):
        clip = fixture_clip(
            f"scene{offset}", "tapvid_davis" if split == "test" else "cotracker3_kubric"
        )
        clip.frames = np.roll(clip.frames, offset, axis=1)
        path = root / f"{split}.npz"
        clip.save(path)
        import hashlib

        entries.append(
            dict(
                clip_id=split,
                path=path.name,
                split=split,
                source_id=clip.source_id,
                dataset=clip.dataset,
                sha256=sha256(path),
                frame_sha256=hashlib.sha256(clip.frames.tobytes()).hexdigest(),
                frames=8,
                tracks=3,
            )
        )
    path = root / "manifest.json"
    path.write_text(json.dumps(dict(schema_version=1, image_size=16, clips=entries)))
    return path


def test_first_query_preserves_delayed_visibility_and_coordinate_order():
    clip = fixture_clip()
    ids, queries = clip.queries()
    np.testing.assert_array_equal(ids, [0, 1, 2])
    np.testing.assert_array_equal(queries[:, 0], [0, 2, 0])
    np.testing.assert_allclose(queries[1, 1:], clip.points[1, 2, ::-1])


def test_perfect_metrics_and_past_frames_are_excluded():
    clip = fixture_clip()
    _, queries = clip.queries()
    predictions = clip.points.copy()
    occlusion = clip.occluded.copy()
    predictions[1, :3] = 100
    occlusion[1, :3] = ~occlusion[1, :3]
    metrics = tapvid_metrics(
        queries, clip.points, clip.occluded, predictions, occlusion
    )
    assert metrics["average_jaccard"] == 1
    assert metrics["average_pts_within_thresh"] == 1
    assert metrics["occlusion_accuracy"] == 1


def test_metrics_use_strict_pixel_thresholds_and_visibility_false_positives():
    queries = np.array([[0, 0.5, 0.5]], np.float32)
    target = np.zeros((1, 3, 2), np.float32)
    prediction = target.copy()
    prediction[0, 1, 0] = 1 / 256
    occluded = np.array([[False, False, True]])
    result = tapvid_metrics(
        queries, target, occluded, prediction, np.zeros((1, 3), bool)
    )
    assert result["pts_within_1"] == 0
    assert result["jaccard_1"] == 0
    assert result["pts_within_2"] == 1
    assert result["jaccard_2"] == 0.5


def test_parity_with_pinned_official_evaluator_when_available():
    path = (
        Path(__file__).resolve().parents[1]
        / "data/external/tapvid/official_evaluation_datasets.py"
    )
    if not path.exists():
        pytest.skip(
            "Download the pinned official evaluator for the reference-parity test."
        )
    document = ast.parse(path.read_text())
    definition = next(
        item
        for item in document.body
        if isinstance(item, ast.FunctionDef) and item.name == "compute_tapvid_metrics"
    )
    definition.returns = None
    for argument in definition.args.args:
        argument.annotation = None
    namespace = {"np": np}
    exec(
        compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    rng = np.random.default_rng(33)
    for _ in range(10):
        truth = rng.uniform(0, 1, (4, 8, 2)).astype(np.float32)
        estimate = truth + rng.normal(0, 0.03, truth.shape)
        occluded = rng.random((4, 8)) < 0.3
        occluded[:, 0] = False
        pred_occluded = rng.random((4, 8)) < 0.4
        query = np.column_stack((np.zeros(4), truth[:, 0, 1], truth[:, 0, 0])).astype(
            np.float32
        )
        ours = tapvid_metrics(query, truth, occluded, estimate, pred_occluded)
        official = namespace["compute_tapvid_metrics"](
            query[None],
            occluded[None],
            truth[None] * 256,
            pred_occluded[None],
            estimate[None] * 256,
            "first",
        )
        for key in official:
            assert ours[key] == pytest.approx(float(official[key][0]), abs=1e-12)


@pytest.mark.parametrize(
    "policy",
    ["uniform", "random", "total_uncertainty", "ats", "ats_fixed_q", "ats_fixed_r"],
)
@pytest.mark.parametrize("fraction", [0.25, 1.0])
def test_budget_no_unread_encoder_access_and_delayed_query(policy, fraction):
    model = TAPATSModel(feature_dim=8, hidden_size=8).eval()
    clip = fixture_clip()
    _, queries = clip.queries()
    calls = []
    handle = model.encoder.register_forward_pre_hook(
        lambda module, args: calls.append(args[0].shape[0])
    )
    result = point_rollout(
        model,
        clip.frames,
        queries[1],
        fraction,
        policy,
        fixed_q=np.full(2, 1e-5),
        fixed_r=np.full(2, 0.01),
    )
    handle.remove()
    assert result["observations"] == int(np.ceil(6 * fraction))
    assert sum(calls) == result["observations"]
    assert not result["acquired"][:2].any()
    assert result["acquired"][2]
    assert np.isfinite(result["points"]).all()


def test_destroying_unread_or_prequery_images_does_not_change_rollout():
    model = TAPATSModel(feature_dim=8, hidden_size=8).eval()
    clip = fixture_clip()
    _, queries = clip.queries()
    first = point_rollout(model, clip.frames, queries[1], 0.5, "ats")
    changed = clip.frames.copy()
    changed[~first["acquired"]] = 255 - changed[~first["acquired"]]
    second = point_rollout(model, changed, queries[1], 0.5, "ats")
    for key in ("points", "visibility_logits", "acquired"):
        np.testing.assert_array_equal(first[key], second[key])
    assert first["decisions"] == second["decisions"]


def test_training_without_velocity_or_clean_targets_reaches_all_components():
    torch.manual_seed(2)
    model = TAPATSModel(feature_dim=8, hidden_size=8)
    clip = fixture_clip()
    _, query = clip.queries()
    mask = np.array([True, False, True, False, True, False, True, True])
    loss = point_training_loss(
        model, clip.frames, query[0], clip.points[0], clip.occluded[0], mask
    )
    loss["total"].backward()
    for module in (
        model.encoder,
        model.decoder,
        model.context_update,
        model.acceleration_head,
        model.process_noise_head,
        model.measurement_noise_head,
        model.observed_visibility_head,
        model.forecast_visibility_head,
    ):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(
            value is not None and torch.isfinite(value).all() for value in gradients
        )
        assert sum(float(value.abs().sum()) for value in gradients) > 0


def test_source_isolation_and_davis_fitting_rejected(tmp_path):
    path = create_manifest(tmp_path / "data")
    manifest = load_tap_manifest(path)
    manifest.pop("root")
    manifest["clips"][1]["source_id"] = manifest["clips"][0]["source_id"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="leaks"):
        load_tap_manifest(path)
    manifest["clips"][0]["dataset"] = "tapvid_davis"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Only Kubric"):
        load_tap_manifest(path)


def test_unpickler_accepts_numpy_but_rejects_executable_globals():
    arrays = dict(x=np.arange(4, dtype=np.float32), flag=np.array([True]))
    recovered = ArrayUnpickler(io.BytesIO(pickle.dumps(arrays, protocol=4))).load()
    np.testing.assert_array_equal(recovered["x"], arrays["x"])
    import os

    with pytest.raises(pickle.UnpicklingError):
        ArrayUnpickler(io.BytesIO(pickle.dumps(os.system))).load()


def make_kubric_archive(path, source):
    """Build the exact released coords/visibility structure, including extra arrays."""
    clip = fixture_clip(source)
    stream = io.BytesIO()
    np.save(stream, dict(coords=clip.points * 16, visibility=clip.occluded))
    with tarfile.open(path, "w:gz") as archive:
        content = stream.getvalue()
        item = tarfile.TarInfo(f"{source}/{source}.npy")
        item.size = len(content)
        archive.addfile(item, io.BytesIO(content))
        item = tarfile.TarInfo(f"{source}/depths/000.npy")
        item.size = 3
        archive.addfile(item, io.BytesIO(b"bad"))
        for clock, frame in enumerate(clip.frames):
            stream = io.BytesIO()
            Image.fromarray(frame).save(stream, format="PNG")
            content = stream.getvalue()
            item = tarfile.TarInfo(f"{source}/frames/{clock:03d}.png")
            item.size = len(content)
            archive.addfile(item, io.BytesIO(content))
    return clip


def test_archive_formats_and_scene_split_before_windows(tmp_path):
    paths = [tmp_path / f"{index:04d}.tar.gz" for index in range(2)]
    for path in paths:
        original = make_kubric_archive(path, path.name[:4])
        actual = read_kubric_archive(path, 16)
        np.testing.assert_array_equal(actual.occluded, original.occluded)
        np.testing.assert_array_equal(actual.points, original.points)
    davis = tmp_path / "davis.zip"
    clip = fixture_clip(dataset="tapvid_davis")
    with zipfile.ZipFile(davis, "w") as archive:
        archive.writestr(
            "tapvid_davis.pkl",
            pickle.dumps(
                dict(
                    video0=dict(
                        video=clip.frames, points=clip.points, occluded=clip.occluded
                    )
                )
            ),
        )
    manifest_path = prepare_tap_data(paths, davis, tmp_path / "prepared", 16, 4)
    manifest = load_tap_manifest(manifest_path)
    assert sorted(
        entry["frames"] for entry in manifest["clips"] if entry["split"] == "train"
    ) == [4, 4]
    assert [
        entry["frames"] for entry in manifest["clips"] if entry["split"] == "validation"
    ] == [8]
    assert [
        entry["frames"] for entry in manifest["clips"] if entry["split"] == "test"
    ] == [8]


def test_train_validation_frozen_test_and_checkpoint_replay(tmp_path):
    data = create_manifest(tmp_path / "data")
    config = TAPTrainConfig(
        data,
        tmp_path / "fit",
        steps=2,
        queries_per_step=1,
        validation_every=1,
        validation_queries=2,
        validation_clips=1,
        feature_dim=8,
        hidden_size=8,
        threads=1,
    )
    train_tap(config)
    assert verify_run(config.output_root, check_current_source=True) == []
    model, checkpoint = load_tap_checkpoint(config.output_root / "model.pt")
    assert checkpoint["selection_complete"]
    evaluation = evaluate_tap(
        data,
        config.output_root / "model.pt",
        tmp_path / "test",
        fractions=(0.5, 1.0),
        policies=("uniform", "ats"),
        max_queries=2,
        threads=1,
    )
    assert verify_run(evaluation, check_current_source=True) == []
    with (evaluation / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    full = [row for row in rows if float(row["fraction"]) == 1]
    assert full[0]["average_jaccard"] == full[1]["average_jaccard"]
    clip = load_tap_clip(tmp_path / "data/test.npz")
    arrays = np.load(evaluation / "predictions/test__uniform__0.500000.npz")
    query = arrays["queries"][0]
    selection = checkpoint["selection"]
    result = point_rollout(
        model,
        clip.frames,
        query,
        0.5,
        "uniform",
        selection["uncertainty_threshold"],
        fixed_q=np.array(selection["fixed_q"]),
        fixed_r=np.array(selection["fixed_r"]),
    )
    np.testing.assert_array_equal(arrays["points"][0], result["points"])
    with pytest.raises(FileExistsError):
        train_tap(config)
    assert verify_run(config.output_root) == []


def test_interrupted_training_resumes_exactly_and_rejects_changed_input(
    tmp_path, monkeypatch
):
    import ats_vision.tap_training as training

    data = create_manifest(tmp_path / "data")
    base = TAPTrainConfig(
        data,
        tmp_path / "reference",
        steps=2,
        queries_per_step=1,
        validation_every=1,
        validation_queries=1,
        validation_clips=1,
        feature_dim=8,
        hidden_size=8,
        threads=1,
    )
    train_tap(base)
    config = replace(base, output_root=tmp_path / "interrupted")
    real = training.point_training_loss
    count = 0

    def interrupted(*args, **kwargs):
        """Interrupt after the first durable checkpoint, before the next update."""
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("simulated interruption")
        return real(*args, **kwargs)

    monkeypatch.setattr(training, "point_training_loss", interrupted)
    with pytest.raises(RuntimeError, match="simulated"):
        train_tap(config)
    monkeypatch.setattr(training, "point_training_loss", real)
    with pytest.raises(ValueError, match="Resume"):
        train_tap(replace(config, learning_rate=0.01), resume=True)
    train_tap(config, resume=True)
    first, _ = load_tap_checkpoint(base.output_root / "model.pt")
    second, _ = load_tap_checkpoint(config.output_root / "model.pt")
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], atol=0, rtol=0)


def test_external_baseline_only_reads_selected_frames_and_uses_past_extrapolation():
    clip = fixture_clip()
    query = clip.queries()[1][1]

    class FakeTracker:
        """Deterministic observed-image interface, independent of any target labels."""

        def initialize(self, frame, xy):
            self.point = np.asarray(xy).copy()
            self.reads = 1
            return self.point, True

        def observe(self, frame):
            self.reads += 1
            self.point = self.point + np.array([0.1, 0.05])
            return self.point, True

    tracker = FakeTracker()
    first = baseline_rollout(tracker, clip.frames, query, 0.5)
    assert tracker.reads == first["acquired"].sum() == 3
    assert not first["acquired"][:2].any()
    modified = clip.frames.copy()
    modified[~first["acquired"]] = 0
    second = baseline_rollout(FakeTracker(), modified, query, 0.5)
    np.testing.assert_array_equal(first["points"], second["points"])


def test_lk_evaluation_writes_recomputable_metrics(tmp_path):
    pytest.importorskip("cv2")
    manifest = create_manifest(tmp_path / "data")
    output = evaluate_baseline(
        manifest,
        tmp_path / "lk",
        fractions=(0.5, 1.0),
        schedules=("uniform",),
        max_queries=2,
    )
    assert verify_run(output, check_current_source=True) == []
    with (output / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert int(rows[1]["observations"]) >= int(rows[0]["observations"])


def test_compute_gate_preserves_checkpoint_and_blocks_unselected_test_model(tmp_path):
    data = create_manifest(tmp_path / "data")
    config = TAPTrainConfig(
        data,
        tmp_path / "deferred",
        steps=1000,
        queries_per_step=1,
        validation_every=1,
        validation_queries=1,
        validation_clips=1,
        feature_dim=8,
        hidden_size=8,
        threads=1,
        max_hours=1e-9,
    )
    train_tap(config)
    manifest = json.loads((config.output_root / "manifest.json").read_text())
    assert manifest["status"] == "deferred"
    assert (config.output_root / "last.pt").is_file()
    with pytest.raises(ValueError, match="selection"):
        evaluate_tap(
            data, config.output_root / "model.pt", tmp_path / "not_tested", threads=1
        )


def test_deterministic_descriptor_sampling_matches_grid_values_and_gradients():
    features = torch.randn(5, 8, 4, 6, requires_grad=True)
    query = torch.tensor([[0, 0], [1, 1], [0.3, 0.7], [0.5, 0.5], [0.02, 0.98]])
    actual = TAPATSModel.sample_descriptor(features, query)
    reference = torch.nn.functional.grid_sample(
        features,
        (2 * query - 1).reshape(5, 1, 1, 2),
        padding_mode="border",
        align_corners=False,
    )[:, :, 0, 0]
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)
    first = torch.autograd.grad(actual.square().sum(), features, retain_graph=True)[0]
    second = torch.autograd.grad(reference.square().sum(), features)[0]
    torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA hardware is unavailable locally"
)
def test_cuda_query_sampler_has_deterministic_backward():
    torch.use_deterministic_algorithms(True)
    features = torch.randn(2, 8, 4, 4, device="cuda", requires_grad=True)
    query = torch.tensor([[0.3, 0.7], [0.5, 0.5]], device="cuda")
    TAPATSModel.sample_descriptor(features, query).square().sum().backward()
    assert torch.isfinite(features.grad).all()
