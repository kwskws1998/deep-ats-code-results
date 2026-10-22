"""Bounded train-only overfit and throughput probes, separate from research fitting."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import torch

from ats.artifacts import complete_run, prepare_run
from ats_vision.experiment import write_json
from ats_vision.tap_batching import point_batch_training_loss
from ats_vision.tap_data import load_tap_clip, load_tap_manifest, sha256
from ats_vision.tap_diagnostics import write_table
from ats_vision.tap_metrics import tapvid_metrics
from ats_vision.tap_model import TAPATSModel
from ats_vision.tap_fast_model import SharedTAPATSModel
from ats_vision.tap_fast_rollout import shared_point_rollouts
from ats_vision.tap_rollout import point_rollout, point_training_loss
from ats_vision.tap_training import configure_runtime


def train_probe_sample(data_manifest: Path, max_frames: int, queries: int):
    """Take only a declared training window and same-clock future-visible queries."""
    if max_frames < 2 or queries < 1:
        raise ValueError("Need >=2 frames and >=1 query.")
    manifest = load_tap_manifest(data_manifest, verify=False)
    for entry in manifest["clips"]:
        if entry["split"] != "train":
            continue
        path = Path(manifest["root"]) / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Training probe input changed: {path}")
        clip = load_tap_clip(path)
        clip.frames = clip.frames[:max_frames]
        clip.points = clip.points[:, : len(clip.frames)]
        clip.occluded = clip.occluded[:, : len(clip.frames)]
        ids, points = clip.queries()
        eligible = np.flatnonzero((~clip.occluded[ids]).sum(1) > 1)
        if not len(eligible):
            continue
        groups = [
            eligible[points[eligible, 0] == clock]
            for clock in sorted(set(points[eligible, 0]))
        ]
        selected = max(groups, key=len)[:queries]
        if len(selected) != queries:
            continue
        return entry, clip, ids[selected], points[selected]
    raise ValueError(
        "No training window has the requested same-clock supervised queries."
    )


def probe_scores(model, clip, ids, queries) -> dict:
    """Score training queries at full observation with no threshold fitting."""
    outputs = [
        point_rollout(model, clip.frames, query, 1.0, "uniform") for query in queries
    ]
    points = np.stack([item["points"] for item in outputs])
    occluded = np.stack([item["visibility_logits"] < 0 for item in outputs])
    return tapvid_metrics(
        queries, clip.points[ids], clip.occluded[ids], points, occluded
    )


def gradient_report(model, terms) -> dict:
    """Detect disconnected/nonfinite modules without populating parameter .grad buffers."""
    parameters = list(model.named_parameters())
    report = {}
    for key in ("total", "coordinate", "visibility", "measurement", "heatmap"):
        gradients = torch.autograd.grad(
            terms[key],
            [item[1] for item in parameters],
            retain_graph=True,
            allow_unused=True,
        )
        groups = {}
        for (name, _), gradient in zip(parameters, gradients):
            group = name.split(".")[0]
            row = groups.setdefault(
                group, dict(connected_parameters=0, finite=True, squared_norm=0.0)
            )
            if gradient is not None:
                row["connected_parameters"] += 1
                row["finite"] &= bool(torch.isfinite(gradient).all())
                row["squared_norm"] += float(gradient.detach().square().sum())
        report[key] = groups
    return report


def overfit_probe(
    data_manifest: Path,
    output: Path,
    steps=200,
    max_frames=16,
    feature_dim=16,
    hidden_size=24,
    learning_rate=0.003,
    device="cpu",
    threads=2,
    seed=17,
):
    """Check whether one train-only query can be fit; never claim held-out generalization."""
    if steps < 1 or not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Need a positive fitting budget and learning rate.")
    configure_runtime(device, threads, seed)
    entry, clip, ids, queries = train_probe_sample(data_manifest, max_frames, 1)
    config = dict(
        data_manifest_sha256=sha256(data_manifest),
        training_clip=entry["clip_id"],
        track_ids=ids.tolist(),
        steps=steps,
        max_frames=max_frames,
        feature_dim=feature_dim,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        device=device,
        threads=threads,
        seed=seed,
    )
    output = Path(output)
    prepare_run(output, config, "tapvid_train_only_overfit_probe")
    model = TAPATSModel(feature_dim, hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)
    observed = np.arange(len(clip.frames)) >= int(queries[0, 0])
    initial_scores = probe_scores(model, clip, ids, queries)
    rows = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        terms = point_training_loss(
            model,
            clip.frames,
            queries[0],
            clip.points[ids[0]],
            clip.occluded[ids[0]],
            observed,
        )
        if not torch.isfinite(terms["total"]):
            raise FloatingPointError(f"Nonfinite overfit loss at step {step}.")
        if step == 1:
            write_json(output / "gradients.json", gradient_report(model, terms))
        terms["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % max(1, steps // 10) == 0 or step == steps:
            row = dict(
                step=step,
                **{
                    f"loss_{key}": float(value.detach()) for key, value in terms.items()
                },
                **probe_scores(model, clip, ids, queries),
            )
            rows.append(row)
            print(
                f"overfit step={step}/{steps} loss={row['loss_total']:.4f} train_AJ={row['average_jaccard']:.4f}",
                flush=True,
            )
    write_table(output / "curve.csv", rows)
    write_json(
        output / "summary.json",
        dict(
            initial_training_metrics=initial_scores,
            final_training_metrics={key: rows[-1][key] for key in initial_scores},
            scope="One training query repeatedly supervised; this is a trainability diagnostic, not validation/test efficacy.",
            gradient_note="Forecast visibility can be disconnected in this full-read probe because no missing observations occur.",
        ),
    )
    torch.save(
        dict(
            kind="tap_ats_overfit_probe",
            model_config=dict(feature_dim=feature_dim, hidden_size=hidden_size),
            state_dict=model.state_dict(),
        ),
        output / "probe.pt",
    )
    complete_run(output)
    return output


def benchmark_queries(
    data_manifest: Path,
    output: Path,
    batch_size=4,
    repeats=3,
    max_frames=60,
    feature_dim=64,
    hidden_size=96,
    device="cuda",
    threads=4,
    seed=17,
    encoder_chunk_size=8,
):
    """Measure reference, microbatch and shared updates on identical train-only inputs."""
    if batch_size < 2 or repeats < 1:
        raise ValueError("Benchmark needs >=2 queries and >=1 measured update.")
    configure_runtime(device, threads, seed)
    entry, clip, ids, queries = train_probe_sample(
        data_manifest, max_frames, batch_size
    )
    masks = np.zeros((batch_size, len(clip.frames)), bool)
    rng = np.random.default_rng(seed)
    start = int(queries[0, 0])
    for index in range(batch_size):
        masks[index, start] = True
        budget = int(np.ceil((len(clip.frames) - start) * (0.25 + 0.25 * (index % 4))))
        if budget > 1:
            masks[
                index,
                rng.choice(
                    np.arange(start + 1, len(clip.frames)), budget - 1, replace=False
                ),
            ] = True
    initial = TAPATSModel(feature_dim, hidden_size)
    output = Path(output)
    prepare_run(
        output,
        dict(
            data_manifest_sha256=sha256(data_manifest),
            training_clip=entry["clip_id"],
            queries=batch_size,
            repeats=repeats,
            max_frames=max_frames,
            feature_dim=feature_dim,
            hidden_size=hidden_size,
            device=device,
            threads=threads,
            seed=seed,
            encoder_chunk_size=encoder_chunk_size,
        ),
        "tapvid_query_throughput_probe",
    )
    rows = []
    np.savez_compressed(
        output / "probe_inputs.npz", track_ids=ids, queries=queries, observed=masks
    )
    for mode in ("sequential", "microbatch", "shared"):
        model = (
            SharedTAPATSModel(feature_dim, hidden_size)
            if mode == "shared"
            else deepcopy(initial)
        )
        model.load_state_dict(initial.state_dict())
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
        encoded_counts, encoder_calls = [], []
        handle = model.encoder.register_forward_pre_hook(
            lambda module, args: encoded_counts.append(len(args[0]))
        )
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        timings = []
        for step in range(repeats + 1):
            encoded_counts.clear()
            if device == "cuda":
                torch.cuda.synchronize()
            begin = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            if mode == "sequential":
                for offset, query in enumerate(queries):
                    terms = point_training_loss(
                        model,
                        clip.frames,
                        query,
                        clip.points[ids[offset]],
                        clip.occluded[ids[offset]],
                        masks[offset],
                    )
                    (terms["total"] / batch_size).backward()
            else:
                terms = point_batch_training_loss(
                    model,
                    clip.frames,
                    queries,
                    clip.points[ids],
                    clip.occluded[ids],
                    masks,
                    shared_features=mode == "shared",
                    encoder_chunk_size=encoder_chunk_size,
                )
                terms["total"].mean().backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5, error_if_nonfinite=True
            )
            optimizer.step()
            if device == "cuda":
                torch.cuda.synchronize()
            if step:
                timings.append(time.perf_counter() - begin)
                encoder_calls.append((len(encoded_counts), sum(encoded_counts)))
        rows.append(
            dict(
                mode=mode,
                queries_per_update=batch_size,
                mean_update_seconds=float(np.mean(timings)),
                queries_per_second=batch_size / float(np.mean(timings)),
                median_update_seconds=float(np.median(timings)),
                encoded_images_per_update=encoder_calls[-1][1],
                encoder_calls_per_update=encoder_calls[-1][0],
                charged_image_query_reads=int(masks.sum()),
                cuda_peak_allocated_bytes=(
                    torch.cuda.max_memory_allocated() if device == "cuda" else None
                ),
                cuda_peak_reserved_bytes=(
                    torch.cuda.max_memory_reserved() if device == "cuda" else None
                ),
            )
        )
        handle.remove()
        print(
            f"benchmark mode={mode} seconds/update={np.mean(timings):.4f} encoded_images={encoder_calls[-1][1]}",
            flush=True,
        )
        del terms, optimizer, model
        if device == "cuda":
            torch.cuda.empty_cache()
    write_table(output / "benchmark.csv", rows)
    validation_rows = []
    for mode in ("sequential", "shared"):
        model = (
            SharedTAPATSModel(feature_dim, hidden_size)
            if mode == "shared"
            else deepcopy(initial)
        )
        model.load_state_dict(initial.state_dict())
        model = model.to(device).eval()
        timings = []
        for step in range(repeats + 1):
            if device == "cuda":
                torch.cuda.synchronize()
            begin = time.perf_counter()
            if mode == "shared":
                predictions = shared_point_rollouts(
                    model,
                    clip.frames,
                    queries,
                    1.0,
                    query_batch_size=batch_size,
                    encoder_chunk_size=encoder_chunk_size,
                )
            else:
                predictions = [
                    point_rollout(model, clip.frames, query, 1.0, "uniform")
                    for query in queries
                ]
            if device == "cuda":
                torch.cuda.synchronize()
            if step:
                timings.append(time.perf_counter() - begin)
        if mode == "sequential":
            reference_predictions = np.stack([row["points"] for row in predictions])
        delta = float(
            np.max(
                np.abs(
                    np.stack([row["points"] for row in predictions])
                    - reference_predictions
                )
            )
        )
        validation_rows.append(
            dict(
                mode=mode,
                mean_validation_seconds=float(np.mean(timings)),
                max_coordinate_difference=delta,
            )
        )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    write_table(output / "validation_benchmark.csv", validation_rows)
    write_json(
        output / "protocol.json",
        dict(
            same_queries_masks_initial_weights=True,
            measured_updates=repeats,
            warmup_updates=1,
            scope="Short train-only probe; hardware speedup is established only for the executed device/configuration.",
            cuda_measured=device == "cuda",
            gpu=torch.cuda.get_device_name(0) if device == "cuda" else None,
            precision="float32; no autocast",
            learning_rate=0.0003,
            optimizer="AdamW, weight decay 1e-4; identical across all arms and main training",
            encoder_chunk_size=encoder_chunk_size,
            validation="full observations; identical initial weights/queries; timed separately",
        ),
    )
    complete_run(output)
    return output
