"""RTX 4090 experiment orchestration and auditable comparison with one published tracker.

The published Track-On-R row is a full-observation literature reference. Sparse
ATS evaluation remains a separate protocol with a matched uniform control.
"""

from __future__ import annotations

import ast
from dataclasses import fields
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from ats.artifacts import complete_run, prepare_run, source_fingerprints, verify_run
from ats_vision.experiment import write_csv, write_json
from ats_vision.tap_data import load_tap_clip, load_tap_manifest, sha256
from ats_vision.tap_model import TAPATSModel
from ats_vision.tap_rollout import point_training_loss, query_budget
from ats_vision.tap_training import (
    TAPTrainConfig,
    configure_runtime,
    evaluate_tap,
    train_tap,
    validate_training_config,
)

PROJECT = Path(__file__).resolve().parents[2]
EVALUATOR_SHA256 = "90cd01e53e23f6d489d3a6cd840cfd93fed4c1a6f4a0dfd373933cc164f0e570"
METRICS = ("average_jaccard", "average_pts_within_thresh", "occlusion_accuracy")


def load_profile(path: Path, output_root=None, seeds=None, max_hours=None) -> dict:
    """Validate a declared server plan without downloading data or allocating a GPU."""
    profile = json.loads(Path(path).read_text())
    required = {
        "schema_version",
        "raw_root",
        "prepared_root",
        "output_root",
        "reference",
        "kubric_revision",
        "scene_indices",
        "image_size",
        "train_length",
        "validation_fraction",
        "split_seed",
        "minimum_prepare_free_gib",
        "seeds",
        "training",
        "evaluation",
    }
    if set(profile) != required or profile["schema_version"] != 1:
        raise ValueError("Unknown or incomplete server profile.")
    if output_root is not None:
        profile["output_root"] = str(output_root)
    if seeds is not None:
        profile["seeds"] = list(seeds)
    if max_hours is not None:
        profile["training"]["max_hours"] = max_hours
    for key in ("raw_root", "prepared_root", "output_root", "reference"):
        profile[key] = str((PROJECT / profile[key]).resolve())
    start, stop = profile["scene_indices"]
    if (
        not isinstance(start, int)
        or not isinstance(stop, int)
        or not 0 <= start < stop <= 6000
        or stop - start < 2
    ):
        raise ValueError(
            "scene_indices must be a half-open range containing >=2 scenes."
        )
    if (
        profile["image_size"] != 256
        or not 2 <= profile["train_length"] <= 120
        or 120 % profile["train_length"]
        or not 0 < profile["validation_fraction"] < 1
        or profile["minimum_prepare_free_gib"] <= 0
        or not profile["seeds"]
        or any(not isinstance(seed, int) or seed < 0 for seed in profile["seeds"])
        or len(set(profile["seeds"])) != len(profile["seeds"])
    ):
        raise ValueError("Invalid resolution, windows, split, disk budget, or seeds.")
    allowed = {field.name for field in fields(TAPTrainConfig)} - {
        "data_manifest",
        "output_root",
        "seed",
    }
    if (
        set(profile["training"]) - allowed
        or profile["training"].get("device") != "cuda"
    ):
        raise ValueError(
            "Use explicit CUDA training settings, without overriding dataset/output/seed."
        )
    if profile["training"].get("validation_observation_fraction") != 1.0:
        raise ValueError(
            "Published-score comparison requires full-observation validation selection."
        )
    validate_training_config(
        TAPTrainConfig(
            Path(profile["prepared_root"]) / "manifest.json",
            Path(profile["output_root"]),
            seed=profile["seeds"][0],
            **profile["training"],
        )
    )
    fractions = profile["evaluation"]["fractions"]
    if set(profile["evaluation"]) != {"fractions", "policies"} or set(
        profile["evaluation"]["policies"]
    ) != {"uniform", "ats"}:
        raise ValueError(
            "This experiment uses only our model, with uniform and ATS acquisition."
        )
    if (
        len(set(fractions)) != len(fractions)
        or 1.0 not in fractions
        or any(not 0 < value <= 1 for value in fractions)
    ):
        raise ValueError(
            "Distinct observation fractions must include the full-observation reference."
        )
    reference = json.loads(Path(profile["reference"]).read_text())
    if reference["model"] != "Track-On-R" or reference["observation_fraction"] != 1.0:
        raise ValueError(
            "This profile has exactly one published competitor: Track-On-R."
        )
    return profile


def dataset_manifest(profile: dict) -> Path:
    """Resolve the single immutable prepared dataset used by all seeds and stages."""
    return Path(profile["prepared_root"]) / "manifest.json"


def validate_prepared_profile(profile: dict) -> dict:
    """Reject accidental scene subsets, changed splits, or a differently prepared dataset."""
    manifest = load_tap_manifest(dataset_manifest(profile))
    for key in ("image_size", "train_length"):
        if manifest[key] != profile[key]:
            raise ValueError(f"Prepared {key} differs from the experiment plan.")
    if manifest["seed"] != profile["split_seed"] or manifest["query_mode"] != "first":
        raise ValueError("Prepared split seed/query mode differs from the plan.")
    start, stop = profile["scene_indices"]
    expected = {f"cotracker3_kubric:{index:04d}" for index in range(start, stop)}
    actual = {row["source_id"] for row in manifest["clips"] if row["split"] != "test"}
    validation = {
        row["source_id"] for row in manifest["clips"] if row["split"] == "validation"
    }
    if actual != expected or len(validation) != int(
        np.ceil(len(expected) * profile["validation_fraction"])
    ):
        raise ValueError(
            "Prepared scene population or validation fraction differs from the plan."
        )
    test = [row for row in manifest["clips"] if row["split"] == "test"]
    if (
        not manifest.get("official_davis_source")
        or len(test) != 30
        or sum(row["tracks"] for row in test) != 650
    ):
        raise ValueError("Need the complete official 30-video/650-track DAVIS source.")
    return manifest


def prepare_server_data(profile: dict) -> Path:
    """Download the fixed bounded scene range and prepare a fresh source-disjoint split."""
    if dataset_manifest(profile).exists():
        validate_prepared_profile(profile)
        return dataset_manifest(profile)
    raw = Path(profile["raw_root"])
    raw.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(raw).free / 2**30
    present_gib = (
        sum(path.stat().st_size for path in raw.glob("kubric/*.tar.gz")) / 2**30
    )
    required_free = max(20.0, profile["minimum_prepare_free_gib"] - present_gib)
    if free < required_free:
        raise RuntimeError(
            f"Only {free:.1f} GiB free; this preparation stage requires {required_free:.1f} GiB of remaining free-space allowance."
        )
    start, stop = profile["scene_indices"]
    present = {path.name for path in (raw / "kubric").glob("*.tar.gz")}
    expected = {f"{index:04d}.tar.gz" for index in range(start, stop)}
    if present - expected:
        raise ValueError(
            "Raw scene directory contains archives outside the declared scene range."
        )
    subprocess.run(
        [
            sys.executable,
            str(PROJECT / "scripts/download_tapvid_data.py"),
            "--root",
            str(raw),
            "--davis",
            "--kubric-revision",
            profile["kubric_revision"],
            "--kubric-scenes",
            *map(str, range(start, stop)),
        ],
        cwd=PROJECT,
        check=True,
    )
    prepared = Path(profile["prepared_root"])
    prepared.parent.mkdir(parents=True, exist_ok=True)
    staging = prepared.with_name(prepared.name + ".building")
    marker = prepared.with_name(prepared.name + ".building.json")
    ownership = {
        key: profile[key]
        for key in (
            "raw_root",
            "prepared_root",
            "kubric_revision",
            "scene_indices",
            "train_length",
            "image_size",
            "validation_fraction",
            "split_seed",
        )
    }
    if staging.exists():
        if not marker.exists() or json.loads(marker.read_text()) != ownership:
            raise ValueError(
                "An unrecognized preparation staging directory exists; preserve it and choose a new prepared_root."
            )
        shutil.rmtree(staging)
    if prepared.exists():
        raise FileExistsError(
            "Prepared destination exists without a completed manifest; choose a fresh prepared_root."
        )
    write_json(marker, ownership)
    subprocess.run(
        [
            sys.executable,
            str(PROJECT / "scripts/prepare_tapvid_data.py"),
            "--kubric-root",
            str(raw / "kubric"),
            "--davis",
            str(raw / "tapvid_davis.zip"),
            "--output-root",
            str(staging),
            "--image-size",
            "256",
            "--train-length",
            str(profile["train_length"]),
            "--validation-fraction",
            str(profile["validation_fraction"]),
            "--seed",
            str(profile["split_seed"]),
        ],
        cwd=PROJECT,
        check=True,
    )
    staging.replace(prepared)
    marker.unlink()
    validate_prepared_profile(profile)
    return dataset_manifest(profile)


def freeze_suite(profile: dict) -> Path:
    """Freeze settings, reference row, source hashes, and dataset before fitting/testing."""
    output = Path(profile["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(json.dumps(profile))
    frozen["training"].pop("max_hours", None)
    identity = dict(
        profile=frozen,
        data_manifest_sha256=sha256(dataset_manifest(profile)),
        reference_sha256=sha256(Path(profile["reference"])),
        sources=source_fingerprints(PROJECT),
        runtime_requirements_sha256=sha256(PROJECT / "requirements/tapvid-4090.txt"),
    )
    path = output / "experiment_identity.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError(
                "Experiment settings, dataset, reference, or source files changed; use a new output root."
            )
    else:
        if any(output.iterdir()):
            raise FileExistsError("Choose a fresh experiment root.")
        write_json(path, identity)
        write_json(output / "profile.json", profile)
        shutil.copyfile(profile["reference"], output / "published_reference.json")
    return output


def gpu_preflight(profile: dict) -> Path:
    """Measure one-query FP32 backward memory and rough throughput on the assigned 4090.

    This is a disposable model on one training clip, never a selected checkpoint
    or a model-quality experiment. All frames are observed to stress memory.
    """
    output = freeze_suite(profile)
    config = profile["training"]
    configure_runtime("cuda", config["threads"], profile["seeds"][0])
    properties = torch.cuda.get_device_properties(0)
    if "4090" not in properties.name or properties.total_memory < 20 * 2**30:
        raise RuntimeError(
            f"Expected an RTX 4090 with >=20 GiB VRAM; got {properties.name}."
        )
    manifest = load_tap_manifest(dataset_manifest(profile), verify=False)
    entry = next(row for row in manifest["clips"] if row["split"] == "train")
    clip = load_tap_clip(Path(manifest["root"]) / entry["path"])
    ids, queries = clip.queries()
    eligible = [
        index for index in range(len(ids)) if (~clip.occluded[ids[index]]).sum() > 1
    ]
    if not eligible:
        raise ValueError(
            "Training preflight clip has no supervised future-visible query."
        )
    choice = min(eligible, key=lambda index: queries[index, 0])
    model = TAPATSModel(config["feature_dim"], config["hidden_size"]).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=1e-4
    )
    observed = np.ones(len(clip.frames), bool)
    observed[: int(queries[choice, 0])] = False
    timings = []
    torch.cuda.reset_peak_memory_stats()
    for index in range(4):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        terms = point_training_loss(
            model,
            clip.frames,
            queries[choice],
            clip.points[ids[choice]],
            clip.occluded[ids[choice]],
            observed,
        )
        if not torch.isfinite(terms["total"]):
            raise RuntimeError("Nonfinite loss in the CUDA preflight.")
        terms["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        if index:
            timings.append(time.perf_counter() - begin)
    report = dict(
        purpose="disposable full-read training-query CUDA execution/memory probe",
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        gpu=properties.name,
        total_vram_bytes=properties.total_memory,
        parameters=sum(p.numel() for p in model.parameters()),
        precision="float32; no autocast",
        image_size=256,
        frames=len(clip.frames) - int(queries[choice, 0]),
        simultaneous_queries=1,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        queries_per_optimizer_update=config["queries_per_step"],
        timed_query_seconds=timings,
        mean_query_seconds=float(np.mean(timings)),
        rough_fitting_hours_full_read=float(np.mean(timings))
        * config["queries_per_step"]
        * config["steps"]
        / 3600,
        timing_limits="single clip, full observation; excludes loading, validation, calibration, and final evaluation; not a wall-clock guarantee",
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        sources=source_fingerprints(PROJECT),
    )
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = output / "preflight" / f"cuda_{name}.json"
    destination.parent.mkdir(exist_ok=True)
    write_json(destination, report)
    print(json.dumps(report, indent=2), flush=True)
    del terms, optimizer, model
    torch.cuda.empty_cache()
    return destination


def run_training(profile: dict, resume=False) -> bool:
    """Fit only our model; stop the suite if any seed is unfinished or deferred."""
    validate_prepared_profile(profile)
    output = freeze_suite(profile)
    gpu_preflight(profile)
    for seed in profile["seeds"]:
        directory = output / f"seed{seed}" / "train"
        if (directory / "manifest.json").exists():
            state = json.loads((directory / "manifest.json").read_text())["status"]
            if state == "completed":
                failures = verify_run(directory, check_current_source=True)
                if failures:
                    raise ValueError(str(failures))
                continue
            if not resume:
                raise RuntimeError(
                    "An unfinished training run exists; pass --resume to continue it."
                )
        else:
            state = None
        config = TAPTrainConfig(
            dataset_manifest(profile), directory, seed=seed, **profile["training"]
        )
        train_tap(config, resume=state is not None)
        if (
            json.loads((directory / "manifest.json").read_text())["status"]
            != "completed"
        ):
            print(
                f"Seed {seed} deferred. Resume with the same profile and --resume; test was not run.",
                flush=True,
            )
            return False
    return True


def run_evaluation(profile: dict, split: str, resume=False) -> None:
    """Evaluate all official first-visible queries without fitting any external model."""
    if split not in {"validation", "test"}:
        raise ValueError("Only held-out validation/test roles can be evaluated.")
    validate_prepared_profile(profile)
    output = freeze_suite(profile)
    for seed in profile["seeds"]:
        directory = output / f"seed{seed}" / split
        if (directory / "manifest.json").exists():
            state = json.loads((directory / "manifest.json").read_text())["status"]
            if state == "completed":
                failures = verify_run(directory, check_current_source=True)
                if failures:
                    raise ValueError(
                        f"Completed evaluation artifacts changed: {failures}"
                    )
                continue
            if state not in {"running", "failed"} or not resume:
                raise ValueError(
                    "An incomplete evaluation exists; use --resume to preserve it and restart evaluation from the frozen checkpoint."
                )
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            directory.rename(directory.with_name(f"{split}_interrupted_{stamp}"))
        evaluate_tap(
            dataset_manifest(profile),
            output / f"seed{seed}/train/model.pt",
            directory,
            split=split,
            fractions=profile["evaluation"]["fractions"],
            policies=profile["evaluation"]["policies"],
            max_queries=(
                0 if split == "test" else profile["training"]["validation_queries"]
            ),
            max_clips=0 if split == "test" else profile["training"]["validation_clips"],
            device="cuda",
            threads=profile["training"]["threads"],
            seed=seed,
        )


def official_metric_function(source: Path):
    """Load only the known, hash-pinned NumPy metric function, without TensorFlow imports."""
    if sha256(source) != EVALUATOR_SHA256:
        raise ValueError(
            "Official metric source hash differs from the pinned evaluator."
        )
    parsed = ast.parse(source.read_text())
    definition = next(
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name == "compute_tapvid_metrics"
    )
    definition.returns = None
    for argument in definition.args.args:
        argument.annotation = None
    scope = {"np": np}
    exec(
        compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"),
        scope,
    )
    return scope["compute_tapvid_metrics"]


def audit_evaluation(
    directory: Path, data_manifest: Path, evaluator: Path, reference: dict
) -> dict:
    """Recompute every metric and budget before allowing a full-benchmark literature comparison."""
    failures = verify_run(directory, check_current_source=True)
    if failures:
        raise ValueError(f"Unverified evaluation: {failures}")
    protocol = json.loads((directory / "protocol.json").read_text())
    if (
        not protocol["complete_official_davis"]
        or protocol["query_mode"] != reference["query_mode"]
        or protocol["input_resolution"] != reference["input_resolution"]
        or protocol["metric_resolution"] != reference["metric_resolution"]
        or not protocol["causal"]
    ):
        raise ValueError(
            "Published-score comparison requires complete DAVIS, query-first, causal, matching resolution."
        )
    manifest = load_tap_manifest(data_manifest)
    saved_manifest = json.loads((directory / "manifest.json").read_text())
    config = saved_manifest["config"]
    if (
        config["data_manifest_sha256"] != sha256(data_manifest)
        or config["max_queries"]
        or config["max_clips"]
        or config["split"] != "test"
    ):
        raise ValueError("Wrong data identity or a test subset was requested.")
    entries = {
        row["clip_id"]: row for row in manifest["clips"] if row["split"] == "test"
    }
    if (
        len(entries) != reference["videos"]
        or sum(row["tracks"] for row in entries.values()) != reference["queries"]
    ):
        raise ValueError("The full official DAVIS population is required.")
    with (directory / "metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    expected = {
        (clip, policy, fraction)
        for clip in entries
        for policy in config["policies"]
        for fraction in config["fractions"]
    }
    actual = [(row["clip_id"], row["policy"], float(row["fraction"])) for row in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Missing or duplicate evaluation rows.")
    if 1.0 not in config["fractions"] or "uniform" not in config["policies"]:
        raise ValueError("Need a full-observation uniform reference.")
    official = official_metric_function(evaluator)
    largest, charged, query_rollouts = 0.0, 0, 0
    for identifier, entry in entries.items():
        clip = load_tap_clip(Path(manifest["root"]) / entry["path"])
        if clip.frames.shape[1:3] != (
            reference["input_resolution"],
            reference["input_resolution"],
        ):
            raise ValueError(
                "Saved image dimensions disagree with the published-reference protocol."
            )
        ids, queries = clip.queries()
        full = None
        for row in [item for item in rows if item["clip_id"] == identifier]:
            fraction = float(row["fraction"])
            filename = f"{identifier}__{row['policy']}__{fraction:.6f}.npz"
            with np.load(
                directory / "predictions" / filename, allow_pickle=False
            ) as arrays:
                data = {key: arrays[key] for key in arrays.files}
            np.testing.assert_array_equal(ids, data["track_ids"])
            np.testing.assert_array_equal(queries, data["queries"])
            calculated = official(
                queries[None],
                clip.occluded[ids][None],
                clip.points[ids][None] * 256,
                data["predicted_occluded"][None],
                data["points"][None] * 256,
                "first",
            )
            for key, value in calculated.items():
                difference = abs(float(row[key]) - float(value[0]))
                largest = max(largest, difference)
                if difference > 1e-12:
                    raise ValueError(f"Official metric mismatch: {identifier}/{key}.")
            masks = data["acquired"]
            if int(row["queries"]) != len(ids) or int(row["observations"]) != int(
                masks.sum()
            ):
                raise ValueError("Query/read accounting mismatch.")
            for query, mask in zip(queries, masks):
                start = int(query[0])
                if (
                    mask[:start].any()
                    or not mask[start]
                    or mask.sum() != query_budget(len(mask) - start, fraction)
                ):
                    raise ValueError(
                        "Acquisition budget or query-frame accounting failed."
                    )
            if fraction == 1.0:
                current = (data["points"], data["predicted_occluded"])
                if full is not None:
                    for previous, following in zip(full, current):
                        np.testing.assert_array_equal(previous, following)
                full = current
            charged += int(masks.sum())
            query_rollouts += len(ids)
    with (directory / "summary.csv").open() as stream:
        summary = list(csv.DictReader(stream))
    keys = [(row["policy"], float(row["fraction"])) for row in summary]
    if len(keys) != len(set(keys)) or set(keys) != {
        (policy, fraction) for _, policy, fraction in expected
    }:
        raise ValueError("Missing or repeated summary rows.")
    for item in summary:
        group = [
            row
            for row in rows
            if row["policy"] == item["policy"]
            and float(row["fraction"]) == float(item["fraction"])
        ]
        for key in METRICS:
            if not np.isclose(
                float(item["mean_" + key]),
                np.mean([float(row[key]) for row in group]),
                atol=1e-12,
                rtol=0,
            ):
                raise ValueError("Summary does not equal the video-macro metric.")
    return dict(
        passed=True,
        metric_rows=len(rows),
        query_rollouts=query_rollouts,
        charged_reads=charged,
        maximum_official_metric_difference=largest,
    )


def published_comparison_rows(summaries: list[dict], reference: dict) -> list[dict]:
    """Attach published numbers only to the same full-observation benchmark condition."""
    rows = []
    for summary in summaries:
        fraction = float(summary["fraction"])
        row = dict(
            model="ATS-CNN",
            seed=summary["seed"],
            policy=summary["policy"],
            observation_fraction=fraction,
            evidence="fresh_local_evaluation",
            comparison=(
                "published_reference"
                if fraction == 1
                else "sparse_protocol_no_external_reference"
            ),
        )
        for key in METRICS:
            score = float(summary["mean_" + key]) * 100
            row[key + "_percent"] = score
            row["difference_vs_published_" + key + "_pp"] = (
                score - reference["metrics"][key] if fraction == 1 else None
            )
        rows.append(row)
    rows.append(
        dict(
            model=reference["model"],
            seed=None,
            policy="published_full_observation",
            observation_fraction=1.0,
            evidence=reference["evidence_type"],
            comparison="published_reference",
            **{key + "_percent": value for key, value in reference["metrics"].items()},
            **{"difference_vs_published_" + key + "_pp": None for key in METRICS},
        )
    )
    return rows


def build_report(profile: dict) -> Path:
    """Save verified numbers and a figure without inventing sparse-baseline results or significance."""
    output = freeze_suite(profile)
    report = output / "report"
    if report.exists():
        failures = verify_run(report, check_current_source=True)
        if failures:
            raise ValueError(str(failures))
        return report
    reference = json.loads((output / "published_reference.json").read_text())
    audit, summaries = {}, []
    for seed in profile["seeds"]:
        directory = output / f"seed{seed}/test"
        audit[str(seed)] = audit_evaluation(
            directory,
            dataset_manifest(profile),
            Path(profile["raw_root"]) / "official_evaluation_datasets.py",
            reference,
        )
        with (directory / "summary.csv").open() as stream:
            summaries.extend(dict(seed=seed, **row) for row in csv.DictReader(stream))
    rows = published_comparison_rows(summaries, reference)
    prepare_run(
        report,
        dict(
            profile=profile,
            published_reference_sha256=sha256(output / "published_reference.json"),
        ),
        "tapvid_published_comparison",
    )
    write_csv(report / "comparison.csv", rows)
    write_json(report / "audit.json", audit)
    write_json(report / "published_reference.json", reference)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True)
    names = ("Average Jaccard", "Within-threshold accuracy", "Occlusion accuracy")
    aggregate = []
    for policy in profile["evaluation"]["policies"]:
        for fraction in profile["evaluation"]["fractions"]:
            group = [
                row
                for row in rows
                if row["model"] == "ATS-CNN"
                and row["policy"] == policy
                and row["observation_fraction"] == fraction
            ]
            item = dict(policy=policy, observation_fraction=fraction, seeds=len(group))
            for key in METRICS:
                values = [row[key + "_percent"] for row in group]
                item[key + "_mean_percent"] = float(np.mean(values))
                item[key + "_std_percent"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else None
                )
            aggregate.append(item)
        selected = sorted(
            [row for row in aggregate if row["policy"] == policy],
            key=lambda row: row["observation_fraction"],
        )
        for axis, key in zip(axes, METRICS):
            axis.errorbar(
                [row["observation_fraction"] * 100 for row in selected],
                [row[key + "_mean_percent"] for row in selected],
                yerr=[row[key + "_std_percent"] or 0 for row in selected],
                marker="o",
                capsize=3,
                label=f"Our model: {policy}",
            )
    for axis, key, title in zip(axes, METRICS, names):
        axis.scatter(
            [100],
            [reference["metrics"][key]],
            marker="D",
            s=60,
            color="black",
            label="Track-On-R (paper, full input)",
            zorder=5,
        )
        axis.set(
            title=title,
            xlabel="Requested observation budget (%)",
            ylabel="Score (%)",
            ylim=(-1, 103),
            xlim=(max(0, min(profile["evaluation"]["fractions"]) * 100 - 5), 105),
        )
        axis.set_yticks(np.arange(0, 101, 20))
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=7, loc="best")
    figure.suptitle(
        "DAVIS first-query, 256 x 256; published reference only at full observation",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(report / "budget_comparison.png", dpi=180)
    figure.savefig(report / "budget_comparison.pdf")
    plt.close(figure)
    write_csv(report / "aggregate.csv", aggregate)
    text = [
        "# ATS-CNN and one published competitor",
        "",
        "All scores below are percentages; higher is better. Fresh results cover all 30 DAVIS videos and 650 first-visible queries.",
        "",
        "| Model | Seed | Schedule | Observation budget | AJ | Within threshold | OA |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        text.append(
            f"| {row['model']} | {row['seed'] if row['seed'] is not None else 'paper'} | {row['policy']} | {100 * row['observation_fraction']:.0f}% | {row['average_jaccard_percent']:.2f} | {row['average_pts_within_thresh_percent']:.2f} | {row['occlusion_accuracy_percent']:.2f} |"
        )
    text.extend(
        [
            "",
            f"Paper evidence, {reference['table']}: “{reference['verbatim_metric_excerpt']}”.",
            f"Protocol excerpt: “{reference['verbatim_protocol_excerpt']}”. [Source]({reference['paper_url']})",
            "",
            "Only 100% observation rows share the published input-observation condition. Sparse rows have no published sparse Track-On-R reference.",
            "The uniform/ATS schedules share one fitted model. At 100% their predictions must coincide; they are not separate trained models.",
            "Error bars, if multiple seeds were requested, are sample standard deviations across seeds, not confidence intervals.",
            *reference["limits"],
            "",
            "No external model was trained or run. Paper results cannot supply matched runtime, paired uncertainty, or a controlled training-data comparison.",
        ]
    )
    (report / "results.md").write_text("\n".join(text) + "\n")
    complete_run(
        report, seeds=profile["seeds"], competitors=1, official_metric_parity=True
    )
    return report
