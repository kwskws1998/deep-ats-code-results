"""Server-plan validation, exact scheduled replay, and published-comparison safeguards."""

from dataclasses import replace
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ats.artifacts import complete_run, prepare_run, verify_run
from ats_vision.experiment import write_csv, write_json
from ats_vision.server_experiment import (
    PROJECT,
    METRICS,
    audit_evaluation,
    build_report,
    freeze_suite,
    load_profile,
    official_metric_function,
    published_comparison_rows,
)
from ats_vision.tap_data import TapClip, load_tap_clip, sha256
from ats_vision.tap_metrics import tapvid_metrics
from ats_vision.tap_training import (
    TAPTrainConfig,
    load_tap_checkpoint,
    step_learning_rate,
    train_tap,
)
from test_tapvid import create_manifest


@pytest.fixture(autouse=True)
def single_thread():
    """Bound generated fixture execution independently of the server thread profile."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_profile_selects_one_published_model_and_full_input_validation():
    profile = load_profile(PROJECT / "configs/tapvid_4090.json")
    assert profile["image_size"] == 256
    assert profile["training"]["validation_observation_fraction"] == 1
    assert profile["scene_indices"] == [0, 128]
    assert profile["seeds"] == [17]
    assert profile["training"]["warmup_steps"] < profile["training"]["steps"]


@pytest.mark.parametrize(
    "change",
    [
        {"image_size": 512},
        {"seeds": [17, 17]},
        {"train_length": 48},
        {"evaluation": {"policies": ["ats"], "fractions": [0.5]}},
        {"training": {"device": "cuda", "validation_observation_fraction": 0.5}},
    ],
)
def test_profile_refuses_incompatible_comparison_conditions(tmp_path, change):
    profile = json.loads((PROJECT / "configs/tapvid_4090.json").read_text())
    profile.update(change)
    path = tmp_path / "profile.json"
    write_json(path, profile)
    with pytest.raises(ValueError):
        load_profile(path)


def test_published_reference_never_becomes_a_sparse_input_baseline():
    reference = json.loads(
        (PROJECT / "references/track_on_r_davis_first.json").read_text()
    )
    summaries = [
        dict(
            seed=17,
            policy="uniform",
            fraction=fraction,
            **{"mean_" + key: 0.5 for key in METRICS},
        )
        for fraction in (0.5, 1)
    ]
    rows = published_comparison_rows(summaries, reference)
    assert rows[0]["average_jaccard_percent"] == 50
    assert rows[0]["difference_vs_published_average_jaccard_pp"] is None
    assert rows[1]["difference_vs_published_average_jaccard_pp"] == pytest.approx(-18.1)
    assert rows[-1]["average_jaccard_percent"] == 68.1
    assert rows[-1]["evidence"] == "published_table_not_locally_reproduced"


def test_warmup_cosine_endpoints_and_exact_resume(tmp_path, monkeypatch):
    import ats_vision.tap_training as training

    data = create_manifest(tmp_path / "data")
    base = TAPTrainConfig(
        data,
        tmp_path / "reference",
        steps=4,
        warmup_steps=2,
        learning_rate_schedule="cosine",
        minimum_learning_rate=0.0001,
        learning_rate=0.001,
        queries_per_step=1,
        validation_every=1,
        validation_queries=1,
        validation_clips=1,
        feature_dim=8,
        hidden_size=8,
        threads=1,
        validation_observation_fraction=1,
        data_cache_clips=2,
    )
    assert step_learning_rate(base, 1) == 0.0005
    assert step_learning_rate(base, 2) == 0.001
    assert step_learning_rate(base, 4) == 0.0001
    train_tap(base)
    interrupted = replace(base, output_root=tmp_path / "interrupted")
    actual = training.point_training_loss
    count = 0

    def stop_after_checkpoint(*args, **kwargs):
        """Interrupt before update two while preserving the first durable state."""
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("fixture interruption")
        return actual(*args, **kwargs)

    monkeypatch.setattr(training, "point_training_loss", stop_after_checkpoint)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        train_tap(interrupted)
    monkeypatch.setattr(training, "point_training_loss", actual)
    train_tap(interrupted, resume=True)
    first, _ = load_tap_checkpoint(base.output_root / "model.pt")
    second, _ = load_tap_checkpoint(interrupted.output_root / "model.pt")
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], atol=0, rtol=0)
    protocol = json.loads((base.output_root / "protocol.json").read_text())
    assert "100%" in protocol["selection"]


def test_validation_fraction_is_forwarded_without_using_test_labels(
    tmp_path, monkeypatch
):
    import ats_vision.tap_training as training

    calls = []

    def score(*args, **kwargs):
        """Record the fraction passed to the held-out evaluation interface."""
        calls.append(args[4])
        return dict(average_jaccard=0.5, visible_position_mse=1), None, None

    data = create_manifest(tmp_path / "data")
    manifest = training.load_tap_manifest(data)
    entries = [row for row in manifest["clips"] if row["split"] == "validation"]
    config = TAPTrainConfig(data, tmp_path / "run", validation_observation_fraction=1)
    model = training.TAPATSModel(8, 8)
    monkeypatch.setattr(training, "evaluate_clip", score)
    training.validation_scores(model, manifest, entries, config)
    assert calls == [1]


def test_unverified_evaluation_is_refused_before_paper_comparison(tmp_path):
    reference = json.loads(
        (PROJECT / "references/track_on_r_davis_first.json").read_text()
    )
    directory = tmp_path / "test"
    directory.mkdir()
    write_json(directory / "manifest.json", dict(status="failed"))
    with pytest.raises(ValueError, match="Unverified evaluation"):
        audit_evaluation(
            directory, tmp_path / "absent.json", tmp_path / "absent.py", reference
        )


def test_metric_loader_refuses_unpinned_python_source(tmp_path):
    path = tmp_path / "metric.py"
    path.write_text("raise RuntimeError('must not execute')\n")
    with pytest.raises(ValueError, match="source hash"):
        official_metric_function(path)


def test_interrupted_evaluation_is_preserved_and_restarted_without_retraining(
    tmp_path, monkeypatch
):
    import ats_vision.server_experiment as server

    profile = load_profile(PROJECT / "configs/tapvid_4090.json", tmp_path / "suite")
    directory = Path(profile["output_root"]) / "seed17/test"
    directory.mkdir(parents=True)
    write_json(directory / "manifest.json", dict(status="failed"))
    (directory / "partial.txt").write_text("original partial result")
    monkeypatch.setattr(server, "validate_prepared_profile", lambda profile: {})
    monkeypatch.setattr(
        server, "freeze_suite", lambda profile: Path(profile["output_root"])
    )
    calls = []

    def evaluate(data, checkpoint, output, **kwargs):
        """Capture a fresh complete-query evaluation from the existing selected model."""
        calls.append((checkpoint, kwargs))
        output.mkdir()
        write_json(output / "manifest.json", dict(status="completed"))

    monkeypatch.setattr(server, "evaluate_tap", evaluate)
    with pytest.raises(ValueError, match="incomplete evaluation"):
        server.run_evaluation(profile, "test")
    server.run_evaluation(profile, "test", resume=True)
    preserved = list(directory.parent.glob("test_interrupted_*"))
    assert len(preserved) == 1
    assert (preserved[0] / "partial.txt").read_text() == "original partial result"
    assert len(calls) == 1
    assert calls[0][0] == directory.parent / "train/model.pt"
    assert calls[0][1]["max_queries"] == 0
    assert calls[0][1]["max_clips"] == 0


def test_interrupted_owned_preparation_staging_is_rebuilt(tmp_path, monkeypatch):
    import ats_vision.server_experiment as server

    profile = load_profile(PROJECT / "configs/tapvid_4090.json", tmp_path / "suite")
    profile["raw_root"] = str(tmp_path / "raw")
    profile["prepared_root"] = str(tmp_path / "raw/prepared")
    monkeypatch.setattr(
        server.shutil,
        "disk_usage",
        lambda path: type("Disk", (), {"free": 200 * 2**30})(),
    )
    monkeypatch.setattr(server, "validate_prepared_profile", lambda profile: {})
    attempts = 0

    def prepare(command, **kwargs):
        """Fail the first generated preparation after writing an owned partial file."""
        nonlocal attempts
        if any("prepare_tapvid_data.py" in str(arg) for arg in command):
            attempts += 1
            destination = Path(command[command.index("--output-root") + 1])
            destination.mkdir()
            if attempts == 1:
                (destination / "partial.txt").write_text("generated staging only")
                raise RuntimeError("fixture interrupted preparation")
            assert not (destination / "partial.txt").exists()
            write_json(destination / "manifest.json", {"generated_fixture": True})

    monkeypatch.setattr(server.subprocess, "run", prepare)
    with pytest.raises(RuntimeError, match="fixture interrupted preparation"):
        server.prepare_server_data(profile)
    final = server.prepare_server_data(profile)
    assert final.exists()
    assert not final.parent.with_name("prepared.building").exists()
    assert not final.parent.with_name("prepared.building.json").exists()


def generated_full_evaluation(tmp_path, profile):
    """Build explicitly generated perfect predictions to test a 30-video/650-query audit.

    The fixture mimics benchmark metadata solely to exercise the report guards;
    its images, trajectories, and scores are not real benchmark evidence.
    """
    import hashlib

    prepared = Path(profile["prepared_root"])
    prepared.mkdir()
    entries = []
    for index in range(30):
        count = 22 if index < 20 else 21
        points = np.full((count, 3, 2), 0.4, np.float32)
        clip = TapClip(
            np.full((3, 256, 256, 3), index, np.uint8),
            points,
            np.zeros((count, 3), bool),
            f"fixture:{index}",
            "tapvid_davis",
        )
        identifier = f"davis_{index:03d}"
        clip.save(prepared / f"{identifier}.npz")
        entries.append(
            dict(
                clip_id=identifier,
                path=f"{identifier}.npz",
                split="test",
                source_id=clip.source_id,
                dataset=clip.dataset,
                tracks=count,
                frames=3,
                sha256=sha256(prepared / f"{identifier}.npz"),
                frame_sha256=hashlib.sha256(clip.frames.tobytes()).hexdigest(),
            )
        )
    write_json(
        prepared / "manifest.json",
        dict(
            schema_version=1,
            image_size=256,
            train_length=60,
            seed=17,
            query_mode="first",
            official_davis_source=True,
            clips=entries,
        ),
    )
    output = freeze_suite(profile)
    directory = output / "seed17/test"
    config = dict(
        data_manifest_sha256=sha256(prepared / "manifest.json"),
        max_queries=0,
        max_clips=0,
        split="test",
        policies=["uniform", "ats"],
        fractions=[0.5, 1],
    )
    prepare_run(directory, config, "generated_test_fixture")
    (directory / "predictions").mkdir()
    rows, summary = [], []
    for entry in entries:
        clip = load_tap_clip(prepared / entry["path"])
        ids, queries = clip.queries()
        for policy in config["policies"]:
            for fraction in config["fractions"]:
                mask = np.ones(clip.occluded.shape, bool)
                if fraction == 0.5:
                    mask[:, 1] = False
                metric = tapvid_metrics(
                    queries, clip.points, clip.occluded, clip.points, clip.occluded
                )
                np.savez_compressed(
                    directory
                    / "predictions"
                    / f"{entry['clip_id']}__{policy}__{fraction:.6f}.npz",
                    points=clip.points,
                    predicted_occluded=clip.occluded,
                    acquired=mask,
                    queries=queries,
                    track_ids=ids,
                )
                rows.append(
                    dict(
                        clip_id=entry["clip_id"],
                        policy=policy,
                        fraction=fraction,
                        queries=len(ids),
                        observations=int(mask.sum()),
                        **metric,
                    )
                )
    for policy in config["policies"]:
        for fraction in config["fractions"]:
            summary.append(
                dict(
                    dataset="tapvid_davis",
                    policy=policy,
                    fraction=fraction,
                    **{"mean_" + key: 1 for key in METRICS},
                )
            )
    write_csv(directory / "metrics.csv", rows)
    write_csv(directory / "summary.csv", summary)
    write_json(
        directory / "protocol.json",
        dict(
            complete_official_davis=True,
            query_mode="first",
            input_resolution=256,
            metric_resolution=256,
            causal=True,
        ),
    )
    complete_run(directory)
    return directory


def test_official_audit_and_report_are_consistent_for_generated_fixture(tmp_path):
    source = PROJECT / "data/external/tapvid/official_evaluation_datasets.py"
    if not source.exists():
        pytest.skip(
            "Download the pinned official metric source to run this integration test."
        )
    profile = load_profile(PROJECT / "configs/tapvid_4090.json", tmp_path / "suite")
    profile["prepared_root"] = str(tmp_path / "prepared")
    profile["raw_root"] = str(source.parent)
    profile["evaluation"]["fractions"] = [0.5, 1]
    directory = generated_full_evaluation(tmp_path, profile)
    reference = json.loads(Path(profile["reference"]).read_text())
    audit = audit_evaluation(
        directory, Path(profile["prepared_root"]) / "manifest.json", source, reference
    )
    assert audit["maximum_official_metric_difference"] == 0
    assert audit["query_rollouts"] == 650 * 4
    report = build_report(profile)
    assert verify_run(report, check_current_source=True) == []
    assert (report / "budget_comparison.png").is_file()
    with (report / "comparison.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["difference_vs_published_average_jaccard_pp"] == ""
    assert len(rows) == 5
    assert build_report(profile) == report
