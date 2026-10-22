"""Source-isolated TAP training, validation selection, and frozen final evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from ats.artifacts import complete_run, json_value, prepare_run, source_fingerprints
from ats_vision.experiment import write_csv, write_json
from ats_vision.tap_data import load_tap_clip, load_tap_manifest, sha256
from ats_vision.tap_metrics import tapvid_metrics
from ats_vision.tap_model import TAPATSModel
from ats_vision.tap_fast_model import SharedTAPATSModel
from ats_vision.tap_fast_rollout import shared_point_rollouts
from ats_vision.tap_rollout import point_rollout, point_training_loss, query_budget
from ats_vision.tap_batching import (
    LossAccumulator,
    point_batch_training_loss,
    query_microbatches,
)


@dataclass(frozen=True)
class TAPTrainConfig:
    """Explicit local/server training budget; validation never consumes DAVIS."""

    data_manifest: Path
    output_root: Path
    steps: int = 1000
    queries_per_step: int = 4
    validation_every: int = 100
    validation_queries: int = 16
    validation_clips: int = 16
    feature_dim: int = 32
    hidden_size: int = 48
    learning_rate: float = 0.001
    seed: int = 17
    threads: int = 2
    device: str = "cpu"
    max_gap: int = 16
    max_hours: float = 12.0
    validation_observation_fraction: float = 0.5
    learning_rate_schedule: str = "constant"
    warmup_steps: int = 0
    minimum_learning_rate: float = 0.0
    data_cache_clips: int = 0
    query_batch_size: int = 1
    execution_backend: str = "reference"
    encoder_chunk_size: int = 8
    validation_query_batch_size: int = 8


def step_learning_rate(config: TAPTrainConfig, step: int) -> float:
    """Use an update-index schedule that resumes without an independent scheduler state."""
    if config.learning_rate_schedule == "constant":
        return config.learning_rate
    if config.warmup_steps and step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
    return config.minimum_learning_rate + 0.5 * (
        config.learning_rate - config.minimum_learning_rate
    ) * (1 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def configure_runtime(device: str, threads: int, seed: int) -> None:
    """Request deterministic execution and expose unsupported accelerators as errors."""
    if device not in {"cpu", "cuda"} or threads < 1 or seed < 0:
        raise ValueError("Invalid device, thread count, or seed.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable.")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


def atomic_checkpoint(path: Path, payload: dict) -> None:
    """Keep the previous complete checkpoint if writing is interrupted."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def choose_queries(clip, maximum: int, seed: int, identifier: str):
    """Choose a declared, label-independent subset of official first-visible queries."""
    ids, queries = clip.queries()
    if maximum < 0:
        raise ValueError("Query limit must be zero (all) or positive.")
    if maximum and len(ids) > maximum:
        key = int(hashlib.sha256(identifier.encode()).hexdigest()[:8], 16)
        indices = np.sort(
            np.random.default_rng(np.random.SeedSequence([seed, key])).choice(
                len(ids), maximum, replace=False
            )
        )
        ids, queries = ids[indices], queries[indices]
    return ids, queries


def evaluate_clip(
    model,
    clip,
    identifier,
    policy,
    fraction,
    threshold,
    visibility_threshold,
    max_queries,
    max_gap,
    seed,
    fixed_q=None,
    fixed_r=None,
):
    """Evaluate independent query rollouts, then aggregate official counts per video."""
    ids, queries = choose_queries(clip, max_queries, seed, identifier)
    if getattr(model, "execution_backend", None) == "shared":
        outputs = shared_point_rollouts(
            model,
            clip.frames,
            queries,
            fraction,
            policy,
            threshold,
            max_gap,
            [seed + int(query_id) for query_id in ids],
            fixed_q,
            fixed_r,
            getattr(model, "validation_query_batch_size", 8),
            getattr(model, "encoder_chunk_size", 8),
        )
    else:
        outputs = []
        for query_id, query in zip(ids, queries):
            outputs.append(
                point_rollout(
                    model,
                    clip.frames,
                    query,
                    fraction,
                    policy,
                    threshold,
                    max_gap,
                    seed + int(query_id),
                    fixed_q,
                    fixed_r,
                )
            )
    points = np.stack([item["points"] for item in outputs])
    visibility = np.stack([item["visibility_logits"] for item in outputs])
    predicted_occluded = visibility < math.log(
        visibility_threshold / (1 - visibility_threshold)
    )
    metrics = tapvid_metrics(
        queries, clip.points[ids], clip.occluded[ids], points, predicted_occluded
    )
    metrics.update(
        queries=len(ids),
        observations=sum(item["observations"] for item in outputs),
        possible_query_frames=sum(
            len(clip.frames) - int(query[0]) for query in queries
        ),
        policy_seconds=sum(item["policy_seconds"] for item in outputs),
        wall_seconds=sum(item["wall_seconds"] for item in outputs),
    )
    arrays = dict(
        points=points,
        visibility_logits=visibility,
        predicted_occluded=predicted_occluded,
        covariance=np.stack([item["covariance"] for item in outputs]),
        acquired=np.stack([item["acquired"] for item in outputs]),
        queries=queries,
        track_ids=ids,
    )
    return metrics, arrays, outputs


def validation_scores(
    model,
    manifest,
    entries,
    config,
    policy="uniform",
    threshold=0.01,
    visibility_threshold=0.5,
    fixed_q=None,
    fixed_r=None,
):
    """Use a frozen deterministic validation subset and equal weighting per source."""
    model.eval()
    scores = []
    for entry in entries:
        clip = load_tap_clip(Path(manifest["root"]) / entry["path"])
        metrics, _, _ = evaluate_clip(
            model,
            clip,
            entry["clip_id"],
            policy,
            config.validation_observation_fraction,
            threshold,
            visibility_threshold,
            config.validation_queries,
            config.max_gap,
            config.seed + 1000,
            fixed_q,
            fixed_r,
        )
        scores.append(metrics)
    return dict(
        average_jaccard=float(np.mean([row["average_jaccard"] for row in scores])),
        visible_position_mse=float(
            np.mean([row["visible_position_mse"] for row in scores])
        ),
        average_pts_within_thresh=float(
            np.mean([row["average_pts_within_thresh"] for row in scores])
        ),
        occlusion_accuracy=float(
            np.mean([row["occlusion_accuracy"] for row in scores])
        ),
    )


def checkpoint_identity(config, manifest):
    """Freeze fitting settings, input bytes, and executed implementation sources."""
    project = Path(__file__).resolve().parents[2]
    values = json_value(asdict(config))
    values.pop("max_hours")
    return dict(
        config=values,
        data_manifest_sha256=sha256(config.data_manifest),
        sources=source_fingerprints(project),
        torch_version=str(torch.__version__),
        numpy_version=str(np.__version__),
        fitting_sources=sorted(
            {
                entry["source_id"]
                for entry in manifest["clips"]
                if entry["split"] != "test"
            }
        ),
    )


def load_tap_checkpoint(path: Path, device="cpu", verify_sources=True):
    """Load only primitive/tensor checkpoint content and verify the fitting code."""
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved.get("schema_version") != 1 or saved.get("kind") != "tap_ats":
        raise ValueError("Unsupported TAP checkpoint.")
    if verify_sources and saved["identity"]["sources"] != source_fingerprints(
        Path(__file__).resolve().parents[2]
    ):
        raise ValueError("Current sources differ from the fitted checkpoint.")
    settings = saved["identity"]["config"]
    model = execution_model(saved["model_config"], settings, device)
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.eval(), saved


def validate_training_config(config: TAPTrainConfig) -> None:
    """Reject invalid budgets before dataset downloads, model allocation, or output writes."""
    if any(
        not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1
        for value in (
            config.steps,
            config.queries_per_step,
            config.validation_every,
            config.validation_queries,
            config.validation_clips,
            config.max_gap,
            config.threads,
            config.feature_dim,
            config.hidden_size,
            config.query_batch_size,
            config.encoder_chunk_size,
            config.validation_query_batch_size,
        )
    ):
        raise ValueError("Training and validation budgets must be positive.")
    if (
        not np.isfinite(config.learning_rate)
        or config.learning_rate <= 0
        or not np.isfinite(config.max_hours)
        or config.max_hours <= 0
    ):
        raise ValueError("Learning rate and compute limit must be finite and positive.")
    if (
        not 0 < config.validation_observation_fraction <= 1
        or config.learning_rate_schedule not in {"constant", "cosine"}
        or not 0 <= config.warmup_steps < config.steps
        or not np.isfinite(config.minimum_learning_rate)
        or not 0 <= config.minimum_learning_rate <= config.learning_rate
        or config.data_cache_clips < 0
        or not isinstance(config.warmup_steps, int)
        or not isinstance(config.data_cache_clips, int)
        or config.feature_dim < 8
        or config.feature_dim % 4
        or config.hidden_size < 8
        or config.seed < 0
        or config.device not in {"cpu", "cuda"}
        or config.execution_backend not in {"reference", "shared"}
    ):
        raise ValueError(
            "Invalid validation fraction, learning-rate schedule, or cache size."
        )


def execution_model(model_config, settings, device):
    """Select a recorded backend without changing parameter names or model capacity."""
    backend = settings.get("execution_backend", "reference")
    if backend not in {"reference", "shared"}:
        raise ValueError("Unknown execution backend.")
    model = (SharedTAPATSModel if backend == "shared" else TAPATSModel)(
        **model_config
    ).to(device)
    model.encoder_chunk_size = settings.get("encoder_chunk_size", 8)
    model.validation_query_batch_size = settings.get("validation_query_batch_size", 8)
    return model


def train_tap(config: TAPTrainConfig, resume=False, fork_from=None) -> Path:
    """Fit on Kubric only; save resumable state and a validation-selected model."""
    validate_training_config(config)
    configure_runtime(config.device, config.threads, config.seed)
    manifest = load_tap_manifest(config.data_manifest)
    train_entries = [entry for entry in manifest["clips"] if entry["split"] == "train"]
    validation_entries = [
        entry for entry in manifest["clips"] if entry["split"] == "validation"
    ][: config.validation_clips]
    if not train_entries or not validation_entries:
        raise ValueError(
            "Both source-disjoint Kubric train and validation splits are required."
        )
    identity = checkpoint_identity(config, manifest)
    output = Path(config.output_root)
    model_config = dict(feature_dim=config.feature_dim, hidden_size=config.hidden_size)
    model = execution_model(model_config, asdict(config), config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-4
    )
    rng = np.random.default_rng(config.seed)
    first_step, best_key, best_step, logs = 0, (-1.0, -float("inf")), 0, []
    best_snapshot = None
    if resume and fork_from is not None:
        raise ValueError("Choose exact resume or a separately recorded fork.")
    fork_payload = None
    if fork_from is not None:
        from ats_vision.tap_fork import prepare_training_fork

        fork_payload = prepare_training_fork(config, identity, fork_from)
    if resume or fork_payload is not None:
        saved = torch.load(
            output / "last.pt", map_location=config.device, weights_only=True
        )
        if saved["identity"] != identity:
            raise ValueError(
                "Resume requires original fitting settings, data bytes, sources, and dependencies."
            )
        recorded = json.loads((output / "manifest.json").read_text())
        if recorded.get("status") not in {"failed", "running", "deferred"}:
            raise ValueError("Only unfinished training may resume.")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        rng.bit_generator.state = saved["numpy_rng"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        if config.device == "cuda":
            torch.cuda.set_rng_state_all([value.cpu() for value in saved["cuda_rng"]])
        first_step, best_key, best_step = (
            saved["step"],
            tuple(saved["best_key"]),
            saved["best_step"],
        )
        logs = saved["logs"]
        best_snapshot = saved.get("best_snapshot")
        if best_snapshot is None:
            best_snapshot = torch.load(
                output / "model.pt", map_location="cpu", weights_only=True
            )
            if best_snapshot["selected_step"] != best_step:
                raise ValueError(
                    "Legacy best checkpoint and resumable state disagree; preserve the run for recovery."
                )
        atomic_checkpoint(output / "model.pt", best_snapshot)
        recorded["status"] = "running"
        write_json(output / "manifest.json", recorded)
    else:
        prepare_run(output, config, "tapvid_train")
    last_saved = None
    fitting_seconds, validation_seconds, validation_checks = 0.0, 0.0, 0
    loading_seconds, cache_hits, executed_queries, executed_microbatches = 0.0, 0, 0, 0
    cache = OrderedDict()
    loss_log = LossAccumulator()
    if config.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(first_step + 1, config.steps + 1):
            fit_begin = time.perf_counter()
            for group in optimizer.param_groups:
                group["lr"] = step_learning_rate(config, step)
            model.train()
            entry = train_entries[int(rng.integers(len(train_entries)))]
            load_begin = time.perf_counter()
            clip = cache.pop(entry["clip_id"], None)
            if clip is None:
                clip = load_tap_clip(Path(manifest["root"]) / entry["path"])
            else:
                cache_hits += 1
            if config.data_cache_clips:
                cache[entry["clip_id"]] = clip
                while len(cache) > config.data_cache_clips:
                    cache.popitem(last=False)
            loading_seconds += time.perf_counter() - load_begin
            ids, queries = clip.queries()
            eligible = np.flatnonzero((~clip.occluded[ids]).sum(1) > 1)
            if not len(eligible):
                raise ValueError(
                    "Training clip has no supervised future-visible query."
                )
            selected = rng.choice(
                eligible, min(config.queries_per_step, len(eligible)), replace=False
            )
            selected_ids, selected_queries, masks = [], [], []
            optimizer.zero_grad(set_to_none=True)
            for choice in selected:
                index, query = ids[choice], queries[choice]
                start = int(query[0])
                fraction = float(rng.choice([0.25, 0.5, 0.75, 1.0]))
                budget = query_budget(len(clip.frames) - start, fraction)
                observed = np.zeros(len(clip.frames), bool)
                observed[start] = True
                if budget > 1:
                    observed[
                        rng.choice(
                            np.arange(start + 1, len(clip.frames)),
                            budget - 1,
                            replace=False,
                        )
                    ] = True
                selected_ids.append(index)
                selected_queries.append(query)
                masks.append(observed)
            selected_ids = np.asarray(selected_ids)
            selected_queries = np.asarray(selected_queries)
            masks = np.stack(masks)
            for group in query_microbatches(selected_queries, config.query_batch_size):
                executed_queries += len(group)
                executed_microbatches += 1
                if (
                    config.query_batch_size == 1
                    and config.execution_backend == "reference"
                ):
                    offset = int(group[0])
                    terms = point_training_loss(
                        model,
                        clip.frames,
                        selected_queries[offset],
                        clip.points[selected_ids[offset]],
                        clip.occluded[selected_ids[offset]],
                        masks[offset],
                    )
                else:
                    terms = point_batch_training_loss(
                        model,
                        clip.frames,
                        selected_queries[group],
                        clip.points[selected_ids[group]],
                        clip.occluded[selected_ids[group]],
                        masks[group],
                        shared_features=config.execution_backend == "shared",
                        encoder_chunk_size=config.encoder_chunk_size,
                    )
                if not torch.isfinite(terms["total"]).all():
                    raise FloatingPointError(
                        f"Nonfinite training objective at step {step}."
                    )
                (terms["total"].sum() / len(selected)).backward()
                loss_log.add(terms)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()
            if config.device == "cuda":
                torch.cuda.synchronize()
            fitting_seconds += time.perf_counter() - fit_begin
            if step % config.validation_every == 0 or step == 1 or step == config.steps:
                validation_begin = time.perf_counter()
                metrics = validation_scores(model, manifest, validation_entries, config)
                current_validation_seconds = time.perf_counter() - validation_begin
                validation_seconds += current_validation_seconds
                validation_checks += 1
                key = (metrics["average_jaccard"], -metrics["visible_position_mse"])
                logs.append(
                    dict(
                        step=step,
                        execution_backend=config.execution_backend,
                        learning_rate=optimizer.param_groups[0]["lr"],
                        **loss_log.summary(),
                        **metrics,
                    )
                )
                if key > best_key:
                    best_key, best_step = key, step
                    best_snapshot = dict(
                        schema_version=1,
                        kind="tap_ats",
                        identity=identity,
                        model_config=model_config,
                        state_dict={
                            name: value.detach().cpu().clone()
                            for name, value in model.state_dict().items()
                        },
                        selected_step=step,
                        validation_metrics=metrics,
                        selection_complete=False,
                    )
                    atomic_checkpoint(output / "model.pt", best_snapshot)
                last_saved = dict(
                    schema_version=1,
                    kind="tap_ats_training_snapshot",
                    model_config=model_config,
                    best_snapshot=best_snapshot,
                    identity=identity,
                    state_dict=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    numpy_rng=rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=(
                        torch.cuda.get_rng_state_all()
                        if config.device == "cuda"
                        else []
                    ),
                    step=step,
                    best_key=list(best_key),
                    best_step=best_step,
                    logs=logs,
                )
                atomic_checkpoint(output / "last.pt", last_saved)
                write_csv(output / "training.csv", logs)
                print(
                    f"step={step}/{config.steps} validation_AJ={metrics['average_jaccard']:.5f} "
                    f"within={metrics['average_pts_within_thresh']:.5f} OA={metrics['occlusion_accuracy']:.5f} "
                    f"position_MSE_px2={metrics['visible_position_mse']:.3f} loss={logs[-1]['loss']:.4f} "
                    f"train_s/update={fitting_seconds / (step - first_step):.3f} validation_s={current_validation_seconds:.2f}",
                    flush=True,
                )
                remaining_checks = sum(
                    index % config.validation_every == 0 or index == config.steps
                    for index in range(step + 1, config.steps + 1)
                )
                projected = (
                    fitting_seconds / (step - first_step) * (config.steps - step)
                )
                projected += validation_seconds / validation_checks * remaining_checks
                write_json(
                    output / "resources.json",
                    dict(
                        completed_updates_this_session=step - first_step,
                        mean_fitting_step_seconds=fitting_seconds / (step - first_step),
                        mean_validation_seconds=validation_seconds / validation_checks,
                        mean_loading_step_seconds=loading_seconds / (step - first_step),
                        mean_nonloading_step_seconds=(fitting_seconds - loading_seconds)
                        / (step - first_step),
                        cache_hit_fraction=cache_hits / (step - first_step),
                        configured_query_batch_size=config.query_batch_size,
                        execution_backend=config.execution_backend,
                        encoder_chunk_size=config.encoder_chunk_size,
                        validation_query_batch_size=config.validation_query_batch_size,
                        mean_executed_query_batch_size=executed_queries
                        / executed_microbatches,
                        timing_estimate_provisional=(step - first_step) < 20,
                        estimated_remaining_fitting_and_validation_hours=projected
                        / 3600,
                        excludes="final calibration and final test evaluation",
                        precision="float32; no autocast",
                        cuda_peak_allocated_bytes=(
                            torch.cuda.max_memory_allocated()
                            if config.device == "cuda"
                            else None
                        ),
                        cuda_peak_reserved_bytes=(
                            torch.cuda.max_memory_reserved()
                            if config.device == "cuda"
                            else None
                        ),
                    ),
                )
                if projected > config.max_hours * 3600 and step - first_step >= min(
                    20, config.steps - first_step
                ):
                    write_json(
                        output / "compute_gate.json",
                        dict(
                            deferred=True,
                            remaining_hours=projected / 3600,
                            limit_hours=config.max_hours,
                        ),
                    )
                    recorded = json.loads((output / "manifest.json").read_text())
                    recorded["status"] = "deferred"
                    write_json(output / "manifest.json", recorded)
                    return output
        model, selected = load_tap_checkpoint(output / "model.pt", config.device)
        q_values, r_values = [], []
        for entry in train_entries[: min(8, len(train_entries))]:
            clip = load_tap_clip(Path(manifest["root"]) / entry["path"])
            ids, queries = choose_queries(clip, 2, config.seed + 2000, entry["clip_id"])
            if config.execution_backend == "shared":
                results = shared_point_rollouts(
                    model,
                    clip.frames,
                    queries,
                    1.0,
                    "uniform",
                    query_batch_size=config.validation_query_batch_size,
                    encoder_chunk_size=config.encoder_chunk_size,
                )
            else:
                results = [
                    point_rollout(model, clip.frames, query, 1.0, "uniform")
                    for query in queries
                ]
            for result in results:
                q_values.extend(result["q_reads"])
                r_values.extend(result["r_reads"][1:])
        if not r_values:
            raise ValueError(
                "Training reference subset has no post-query acquisitions."
            )
        fixed_q = np.median(np.asarray(q_values), axis=0)
        fixed_r = np.median(np.asarray(r_values), axis=0)
        visibility_grid = (0.3, 0.5, 0.7)
        visibility_scores = {
            str(value): validation_scores(
                model, manifest, validation_entries, config, visibility_threshold=value
            )["average_jaccard"]
            for value in visibility_grid
        }
        visibility_threshold = max(
            visibility_grid,
            key=lambda value: (visibility_scores[str(value)], -abs(value - 0.5)),
        )
        threshold_grid = (0.001, 0.01, 0.1)
        threshold_scores = {
            str(value): validation_scores(
                model,
                manifest,
                validation_entries,
                config,
                "total_uncertainty",
                value,
                visibility_threshold,
                fixed_q,
                fixed_r,
            )["average_jaccard"]
            for value in threshold_grid
        }
        threshold = max(
            threshold_grid, key=lambda value: (threshold_scores[str(value)], -value)
        )
        selection = dict(
            fixed_q=fixed_q.tolist(),
            fixed_r=fixed_r.tolist(),
            visibility_threshold=visibility_threshold,
            uncertainty_threshold=threshold,
            validation_visibility_scores=visibility_scores,
            validation_uncertainty_scores=threshold_scores,
        )
        selected.update(selection_complete=True, selection=selection)
        atomic_checkpoint(output / "model.pt", selected)
        write_json(output / "selection.json", selection)
        write_json(
            output / "protocol.json",
            dict(
                task="causal single-query point tracking with per-query acquisition budgets",
                dataset_manifest_sha256=identity["data_manifest_sha256"],
                training_source_videos=len({row["source_id"] for row in train_entries}),
                validation_source_videos=len(
                    {row["source_id"] for row in validation_entries}
                ),
                image_size=manifest["image_size"],
                model_config=model_config,
                selection=f"maximum validation AJ at {config.validation_observation_fraction:.0%} budget, then lower visible-coordinate MSE; uniform schedule",
                supervision="visible xy and occlusion only; no velocities, clean images, or true Q/R",
                uncertainty="effective coordinate measurement/dynamics error; physical Q/R separation is not established",
                evaluation="first-visible public query; query and all earlier frames excluded from metrics",
                accounting="independent query states; every acquired image-query pair is charged; shared backend reuses query-independent encoder features only",
                execution_backend=config.execution_backend,
                limits="small CNN and approximate Gaussian planner; no SOTA/novelty claim",
            ),
        )
        complete_run(
            output,
            selected_step=best_step,
            training_steps=config.steps,
            model_parameters=sum(parameter.numel() for parameter in model.parameters()),
        )
    except Exception as error:
        recorded = json.loads((output / "manifest.json").read_text())
        recorded.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        write_json(output / "manifest.json", recorded)
        raise
    return output


def evaluate_tap(
    data_manifest: Path,
    checkpoint: Path,
    output: Path,
    split="test",
    fractions=(0.25, 0.5, 1.0),
    policies=(
        "uniform",
        "random",
        "total_uncertainty",
        "ats",
        "ats_fixed_q",
        "ats_fixed_r",
    ),
    max_queries=0,
    max_clips=0,
    device="cpu",
    threads=2,
    seed=17,
) -> Path:
    """Freeze all choices before testing; save per-video official metrics and predictions."""
    if split not in {"validation", "test"} or max_queries < 0 or max_clips < 0:
        raise ValueError(
            "Evaluation needs a held-out role and nonnegative declared limits."
        )
    if (
        not fractions
        or any(not np.isfinite(value) or not 0 < value <= 1 for value in fractions)
        or len(set(fractions)) != len(fractions)
    ):
        raise ValueError("Fractions must be distinct values in (0,1].")
    allowed = {
        "uniform",
        "random",
        "total_uncertainty",
        "ats",
        "ats_fixed_q",
        "ats_fixed_r",
    }
    if (
        not policies
        or not set(policies) <= allowed
        or len(set(policies)) != len(policies)
    ):
        raise ValueError("Unknown or repeated sampling policies.")
    configure_runtime(device, threads, seed)
    manifest = load_tap_manifest(data_manifest)
    entries = [entry for entry in manifest["clips"] if entry["split"] == split]
    entries = entries[:max_clips] if max_clips else entries
    if not entries:
        raise ValueError(
            f"No {split} clips; prepare the official evaluation data first."
        )
    model, saved = load_tap_checkpoint(checkpoint, device)
    if not saved.get("selection_complete"):
        raise ValueError(
            "Checkpoint has not completed validation-only model/threshold selection."
        )
    if split == "test" and any(
        entry["source_id"] in saved["identity"]["fitting_sources"] for entry in entries
    ):
        raise ValueError("Evaluation source was used for fitting or model selection.")
    if saved["identity"]["data_manifest_sha256"] != sha256(data_manifest):
        raise ValueError("Use the exact dataset manifest frozen before training.")
    output = Path(output)
    config = dict(
        data_manifest=str(Path(data_manifest).resolve()),
        checkpoint=str(Path(checkpoint).resolve()),
        checkpoint_sha256=sha256(checkpoint),
        data_manifest_sha256=sha256(data_manifest),
        split=split,
        fractions=list(fractions),
        policies=list(policies),
        max_queries=max_queries,
        max_clips=max_clips,
        device=device,
        seed=seed,
        threads=threads,
    )
    prepare_run(output, config, "tapvid_frozen_evaluation")
    (output / "predictions").mkdir()
    selection = saved["selection"]
    rows, traces = [], []
    try:
        for entry in entries:
            clip = load_tap_clip(Path(manifest["root"]) / entry["path"])
            for fraction in fractions:
                for policy in policies:
                    metrics, arrays, outputs = evaluate_clip(
                        model,
                        clip,
                        entry["clip_id"],
                        policy,
                        fraction,
                        selection["uncertainty_threshold"],
                        selection["visibility_threshold"],
                        max_queries,
                        saved["identity"]["config"]["max_gap"],
                        seed,
                        np.asarray(selection["fixed_q"]),
                        np.asarray(selection["fixed_r"]),
                    )
                    rows.append(
                        dict(
                            clip_id=entry["clip_id"],
                            source_id=entry["source_id"],
                            dataset=entry["dataset"],
                            policy=policy,
                            fraction=fraction,
                            **metrics,
                        )
                    )
                    filename = f"{entry['clip_id']}__{policy}__{fraction:.6f}.npz"
                    np.savez_compressed(output / "predictions" / filename, **arrays)
                    traces.append(
                        dict(
                            clip_id=entry["clip_id"],
                            policy=policy,
                            fraction=fraction,
                            predictions=filename,
                            decisions=[result["decisions"] for result in outputs],
                        )
                    )
            print(
                f"evaluated {entry['clip_id']} queries={rows[-1]['queries']}",
                flush=True,
            )
        summaries, paired = [], []
        rng = np.random.default_rng(seed + 9000)
        for dataset in sorted({entry["dataset"] for entry in entries}):
            for fraction in fractions:
                baseline = {
                    row["clip_id"]: row
                    for row in rows
                    if row["dataset"] == dataset
                    and row["fraction"] == fraction
                    and row["policy"] == "uniform"
                }
                for policy in policies:
                    group = [
                        row
                        for row in rows
                        if row["dataset"] == dataset
                        and row["fraction"] == fraction
                        and row["policy"] == policy
                    ]
                    summary = dict(
                        dataset=dataset,
                        policy=policy,
                        fraction=fraction,
                        videos=len(group),
                    )
                    for key in (
                        "average_jaccard",
                        "average_pts_within_thresh",
                        "occlusion_accuracy",
                        "visible_position_mse",
                        "observations",
                        "possible_query_frames",
                        "wall_seconds",
                        "policy_seconds",
                    ):
                        summary[f"mean_{key}"] = float(
                            np.mean([row[key] for row in group])
                        )
                    summaries.append(summary)
                    if policy != "uniform" and baseline:
                        differences = np.asarray(
                            [
                                row["average_jaccard"]
                                - baseline[row["clip_id"]]["average_jaccard"]
                                for row in group
                            ]
                        )
                        means = differences[
                            rng.integers(0, len(group), (1000, len(group)))
                        ].mean(1)
                        low, high = np.quantile(means, [0.025, 0.975])
                        paired.append(
                            dict(
                                dataset=dataset,
                                fraction=fraction,
                                policy=policy,
                                videos=len(group),
                                aj_difference_vs_uniform=float(differences.mean()),
                                exploratory_ci95_low=float(low),
                                exploratory_ci95_high=float(high),
                            )
                        )
        write_csv(output / "metrics.csv", rows)
        write_csv(output / "summary.csv", summaries)
        if paired:
            write_csv(output / "paired_comparisons.csv", paired)
        write_json(output / "decisions.json", traces)
        write_json(
            output / "protocol.json",
            dict(
                query_mode="first",
                causal=True,
                input_resolution=manifest["image_size"],
                metric_resolution=256,
                score_scale="0..1; multiply by 100 for paper percent",
                complete_official_davis=(
                    manifest.get("official_davis_source", False)
                    and split == "test"
                    and not max_queries
                    and not max_clips
                    and len(entries) == 30
                    and all(entry["dataset"] == "tapvid_davis" for entry in entries)
                ),
                modified_protocol="per-query observation budget; not directly comparable to full-observation published rankings",
                accounting="independent image-query acquisitions, including query; shared backend reuses only query-independent features from the acquisition union; no free observations",
                execution_backend=saved["identity"]["config"].get(
                    "execution_backend", "reference"
                ),
                selection=selection,
                bootstrap="whole-video exploratory paired intervals; no multiplicity correction",
                runtime="encoder and planner wall time after stored-frame decoding; not equal total compute",
            ),
        )
        complete_run(output, videos=len(entries), metric_rows=len(rows), split=split)
    except Exception as error:
        recorded = json.loads((output / "manifest.json").read_text())
        recorded.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        write_json(output / "manifest.json", recorded)
        raise
    return output
