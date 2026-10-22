"""Failure localization with immutable weights, training controls, and explicit oracles.

All data roles are train/validation. Labels enter reducers only after prediction;
oracle readouts and post-hoc thresholds are diagnostics, never benchmark scores.
This module neither replaces a trained model nor proves a server run's cause
unless that run's actual checkpoint is supplied.
"""

from pathlib import Path
import math

import numpy as np
import torch

from ats.artifacts import complete_run, prepare_run
from ats_vision.experiment import write_json
from ats_vision.tap_baselines import LKTracker, baseline_rollout
from ats_vision.tap_checkpoint import load_diagnostic_checkpoint
from ats_vision.tap_data import load_tap_clip, load_tap_manifest, sha256
from ats_vision.tap_diagnostics import (
    READOUTS,
    analyze_traces,
    full_observation_trace,
    horizon_mask,
    write_table,
)
from ats_vision.tap_metrics import tapvid_metrics
from ats_vision.tap_readout import heatmap_readouts
from ats_vision.tap_rollout import frame_tensor, point_training_loss
from ats_vision.tap_training import choose_queries, configure_runtime


def encoder_geometry(model):
    """Report convolutional geometry; GroupNorm's global statistics are not a local bound."""
    field, stride = 1, 1
    for layer in model.encoder:
        if isinstance(layer, torch.nn.Conv2d):
            field += (layer.kernel_size[0] - 1) * layer.dilation[0] * stride
            stride *= layer.stride[0]
    return dict(
        receptive_field_px=field,
        receptive_field_convention="Convolutional geometric support only; GroupNorm also depends on image-wide group statistics, so this is not a strict input-dependency bound.",
        stride=stride,
        parameters=sum(p.numel() for p in model.parameters()),
    )


def mechanism_controls():
    """Expose global-expectation failure on an explicitly constructed two-peak map."""
    logits = torch.full((1, 64, 64), -100.0)
    logits[0, 32, 16] = logits[0, 32, 48] = 0
    values = heatmap_readouts(logits)
    target = torch.tensor([[16.5 / 64, 32.5 / 64]])
    return dict(
        kind="constructed mechanism control, not participant/video/checkpoint performance",
        global_mean_error_px=float(
            torch.linalg.vector_norm((values["global_mean"] - target) * 256)
        ),
        peak_error_px=float(torch.linalg.vector_norm((values["peak"] - target) * 256)),
        local_mean_error_px=float(
            torch.linalg.vector_norm((values["local_mean"] - target) * 256)
        ),
        limitation="Equal peaks with an explicitly designated left target demonstrate a possible failure, not its frequency in the trained server model.",
    )


def sampled_data_profile(clip, ids, queries):
    """Describe exactly the selected query population, not the entire dataset."""
    future = np.arange(len(clip.frames))[None] > queries[:, :1]
    visible = ~clip.occluded[ids]
    previous_visible = visible[:, 1:] & visible[:, :-1] & future[:, 1:]
    speed = np.linalg.norm(np.diff(clip.points[ids], axis=1) * 256, axis=-1)[
        previous_visible
    ]
    displacement = np.linalg.norm(
        (clip.points[ids] - queries[:, None, [2, 1]]) * 256, axis=-1
    )[future & visible]
    return dict(
        queries=len(ids),
        frames=len(clip.frames),
        future_query_frames=int(future.sum()),
        future_visible_fraction=float(visible[future].mean()) if future.any() else None,
        query_start_median=float(np.median(queries[:, 0])),
        speed_median_px=float(np.median(speed)) if len(speed) else None,
        speed_p95_px=float(np.quantile(speed, 0.95)) if len(speed) else None,
        displacement_from_query_median_px=(
            float(np.median(displacement)) if len(displacement) else None
        ),
        adjacent_visible_speed_samples=len(speed),
    )


def heatmap_frame_statistics(logits, xy=None):
    """Measure confidence/localization after prediction, using labels only for scoring."""
    raw = logits[0].detach().cpu().numpy().astype(float)
    height, width = raw.shape
    probabilities = np.exp(raw - raw.max())
    probabilities /= probabilities.sum()
    peak = np.unravel_index(raw.argmax(), raw.shape)
    mean = np.asarray(
        [
            (probabilities * (np.arange(width)[None] + 0.5) / width).sum(),
            (probabilities * (np.arange(height)[:, None] + 0.5) / height).sum(),
        ]
    )
    row = dict(
        entropy_normalized=float(
            -(probabilities * np.log(np.maximum(probabilities, 1e-300))).sum()
            / math.log(height * width)
        ),
        peak_probability=float(probabilities.max()),
        global_to_peak_distance_px=float(
            np.linalg.norm(
                (mean - np.array([(peak[1] + 0.5) / width, (peak[0] + 0.5) / height]))
                * 256
            )
        ),
        target_cell_probability=None,
        target_cell_rank=None,
        target_cell_cross_entropy=None,
        target_cell_is_peak=None,
        mass_within_4px=None,
        mass_within_8px=None,
    )
    if xy is not None:
        column = int(np.clip(xy[0] * width, 0, width - 1))
        line = int(np.clip(xy[1] * height, 0, height - 1))
        probability = probabilities[line, column]
        row.update(
            target_cell_probability=float(probability),
            target_cell_rank=int(1 + (raw > raw[line, column]).sum()),
            target_cell_cross_entropy=float(
                raw.max() + np.log(np.exp(raw - raw.max()).sum()) - raw[line, column]
            ),
            target_cell_is_peak=bool((line, column) == peak),
        )
        yy, xx = np.meshgrid(
            (np.arange(height) + 0.5) / height,
            (np.arange(width) + 0.5) / width,
            indexing="ij",
        )
        squared = ((xx - xy[0]) * 256) ** 2 + ((yy - xy[1]) * 256) ** 2
        for radius in (4, 8):
            row[f"mass_within_{radius}px"] = float(
                probabilities[squared < radius**2].sum()
            )
    return row


@torch.no_grad()
def audited_query(model, clip, index, query):
    """Keep inference target-free, attaching only a read-only post-prediction reducer."""
    frames = []

    def collect(clock, prior, posterior, observation):
        """Compute labeled diagnostics after the state update, never feeding them back."""
        visible = not bool(clip.occluded[index, clock])
        xy = clip.points[index, clock] if visible else None
        covariance = prior.covariance[0].cpu().numpy().astype(float)
        r = observation["effective_r"][0].cpu().numpy().astype(float)
        gain = np.linalg.solve(covariance[:2, :2] + np.diag(r), covariance[:2, :]).T
        frames.append(
            dict(
                clock=clock,
                elapsed=clock - int(query[0]),
                target_visible=visible,
                gain_x=float(gain[0, 0]),
                gain_y=float(gain[1, 1]),
                effective_measurement_std_px=float(np.sqrt(r).mean() * 256),
                prior_position_std_px=float(
                    np.sqrt(np.diag(covariance)[:2]).mean() * 256
                ),
                innovation_px=float(
                    np.linalg.norm(
                        (observation["measurement"][0] - prior.mean[0, :2])
                        .cpu()
                        .numpy()
                    )
                    * 256
                ),
                visibility_probability=float(
                    observation["visibility_logit"].sigmoid()[0]
                ),
                **heatmap_frame_statistics(observation["logits"], xy),
            )
        )

    trace = full_observation_trace(model, clip.frames, query, observer=collect)
    image = frame_tensor(clip.frames, int(query[0]), model.transition.device)
    belief = model.initialize(image, image.new_tensor(query[[2, 1]])[None])
    _, observation = model.observe(belief, image, image.new_zeros(1))
    readouts = heatmap_readouts(observation["logits"])
    identity = dict(
        **heatmap_frame_statistics(observation["logits"], query[[2, 1]]),
        **{
            f"self_{key}_error_px": float(
                torch.linalg.vector_norm(
                    (readouts[key][0] - image.new_tensor(query[[2, 1]])) * 256
                )
            )
            for key in ("global_mean", "peak", "local_mean")
        },
    )
    return trace, frames, identity


def extra_component_rows(clip, ids, queries, traces, training_length):
    """Use fixed thresholds/masks and label-only controls without selecting a model."""
    rows = []
    target, occluded = clip.points[ids], clip.occluded[ids]
    for threshold in (0.1, 0.3, 0.5, 0.7, 0.9):
        rows.append(
            dict(
                condition="posthoc_visibility_threshold_DIAGNOSTIC_ONLY",
                value=threshold,
                **tapvid_metrics(
                    queries,
                    target,
                    occluded,
                    traces["filtered"],
                    traces["visibility_probability"] < threshold,
                ),
            )
        )
    grid = np.array([(clip.frames.shape[2] + 3) // 4, (clip.frames.shape[1] + 3) // 4])
    oracle = (np.minimum(np.floor(np.clip(target, 0, 1) * grid), grid - 1) + 0.5) / grid
    rows.append(
        dict(
            condition="oracle_label_cell_centres_DIAGNOSTIC_ONLY",
            value=f"{grid[1]}x{grid[0]}",
            **tapvid_metrics(queries, target, occluded, oracle, occluded),
        )
    )
    for name, low, high in (
        ("within_training_max_horizon", 1, training_length - 1),
        ("beyond_training_max_horizon", training_length, None),
    ):
        mask = horizon_mask(queries, len(clip.frames), low, high)
        if (mask & ~occluded).any():
            rows.append(
                dict(
                    condition=name,
                    value=training_length,
                    **tapvid_metrics(
                        queries,
                        target,
                        occluded,
                        traces["filtered"],
                        traces["visibility_probability"] < 0.5,
                        evaluation_mask=mask,
                    ),
                )
            )
    visible = ~occluded
    reappear = np.zeros_like(visible)
    reappear[:, 1:] = visible[:, 1:] & ~visible[:, :-1]
    speed = np.zeros_like(visible, dtype=float)
    speed[:, 1:] = np.linalg.norm(np.diff(target, axis=1) * 256, axis=-1)
    consecutive = np.zeros_like(visible)
    consecutive[:, 1:] = visible[:, 1:] & visible[:, :-1]
    masks = {"reappearance_first_visible_frame": reappear}
    masks.update(
        {
            f"visible_speed_{low}_{high}_px": consecutive
            & (speed >= low)
            & (speed < high)
            for low, high in ((0, 2), (2, 8), (8, float("inf")))
        }
    )
    for name, mask in masks.items():
        mask &= np.arange(len(clip.frames))[None] > queries[:, :1]
        if (mask & visible).any():
            rows.append(
                dict(
                    condition=name,
                    value=None,
                    **tapvid_metrics(
                        queries,
                        target,
                        occluded,
                        traces["filtered"],
                        traces["visibility_probability"] < 0.5,
                        evaluation_mask=mask,
                    ),
                )
            )
    return rows


def gradient_conflicts(model, clip, index, query, max_frames=16, fraction=1.0):
    """Measure weighted loss-gradient norms/cosines on one bounded training prefix."""
    start = int(query[0])
    stop = min(len(clip.frames), start + max_frames)
    if stop <= start + 1 or not (~clip.occluded[index, start + 1 : stop]).any():
        return dict(
            defined=False, reason="No future-visible training target in bounded prefix"
        )
    if not 0 < fraction <= 1:
        raise ValueError("Gradient observation fraction must be in (0,1].")
    observed = np.zeros(stop, bool)
    observed[start] = True
    budget = int(math.ceil((stop - start) * fraction))
    if budget > 1:
        observed[
            np.random.default_rng(17).choice(
                np.arange(start + 1, stop), budget - 1, replace=False
            )
        ] = True
    terms = point_training_loss(
        model,
        clip.frames[:stop],
        query,
        clip.points[index, :stop],
        clip.occluded[index, :stop],
        observed,
    )
    weights = dict(coordinate=1.0, visibility=1.0, measurement=0.1, heatmap=1.0)
    named = list(model.named_parameters())
    vectors, blocks, components = {}, {}, []
    for key, weight in weights.items():
        gradients = torch.autograd.grad(
            terms[key] * weight,
            [p for _, p in named],
            retain_graph=True,
            allow_unused=True,
        )
        vector, groups = [], {}
        for (name, parameter), gradient in zip(named, gradients):
            if gradient is None:
                gradient = torch.zeros_like(parameter)
            value = gradient.detach().cpu().numpy().astype(float)
            if not np.isfinite(value).all():
                raise FloatingPointError(f"Nonfinite gradient in {key}/{name}")
            vector.append(value.ravel())
            group = name.split(".")[0]
            groups[group] = groups.get(group, 0.0) + float(np.square(value).sum())
        vectors[key] = np.concatenate(vector)
        blocks[key] = {name: math.sqrt(value) for name, value in groups.items()}
        components.append(
            dict(
                component=key,
                objective_weight=weight,
                unweighted_loss=float(terms[key].detach()),
                weighted_gradient_norm=float(np.linalg.norm(vectors[key])),
            )
        )
    total = sum(vectors.values())
    cosines = []
    for a, first in vectors.items():
        for b, second in vectors.items():
            if a >= b:
                continue
            denominator = np.linalg.norm(first) * np.linalg.norm(second)
            cosines.append(
                dict(
                    first=a,
                    second=b,
                    cosine=float(first @ second / denominator) if denominator else None,
                )
            )
    return dict(
        defined=True,
        frames=stop - start,
        observation_fraction=fraction,
        components=components,
        per_module_gradient_norms=blocks,
        pairwise_cosines=cosines,
        total_unclipped_gradient_norm=float(np.linalg.norm(total)),
        would_clip_at_training_norm_5=bool(np.linalg.norm(total) > 5),
        limitation="One bounded training prefix; does not estimate historical clipping frequency or prove conflicting gradients caused the plateau. Forecast-visibility gradients are expected to be zero at full observation because that head is unused.",
    )


def failure_audit(
    data_manifest,
    checkpoint,
    output,
    max_clips=2,
    max_queries=8,
    gradient_frames=16,
    include_lk=True,
    device="cpu",
    threads=2,
    seed=17,
):
    """Audit a frozen checkpoint on train and validation with no final-test access."""
    if min(max_clips, max_queries, gradient_frames) < 1:
        raise ValueError("Audit limits must be positive.")
    configure_runtime(device, threads, seed)
    manifest = load_tap_manifest(data_manifest, verify=False)
    model, saved, provenance = load_diagnostic_checkpoint(
        checkpoint, data_manifest, device
    )
    output = Path(output).resolve()
    if output.is_relative_to(Path(checkpoint).resolve().parent):
        raise ValueError("Save audit artifacts outside the parent fitting run.")
    fitting = saved["identity"]["config"]
    selected = [
        entry
        for role in ("train", "validation")
        for entry in [r for r in manifest["clips"] if r["split"] == role][:max_clips]
    ]
    if {r["split"] for r in selected} != {"train", "validation"}:
        raise ValueError(
            "Need separate training and validation sources for a failure audit."
        )
    config = dict(
        data_manifest_sha256=sha256(data_manifest),
        checkpoint_provenance=provenance,
        max_clips_per_role=max_clips,
        max_queries=max_queries,
        gradient_frames=gradient_frames,
        include_lk=include_lk,
        device=device,
        threads=threads,
        seed=seed,
    )
    evaluator = (
        Path(data_manifest).resolve().parent.parent / "official_evaluation_datasets.py"
    )
    official = None
    if evaluator.exists():
        from ats_vision.server_experiment import official_metric_function

        official = official_metric_function(evaluator)
        config["official_evaluator_sha256"] = sha256(evaluator)
    prepare_run(output, config, "tapvid_frozen_failure_audit_not_benchmark")
    components, calibration, frames, identities, extras, data_rows = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    gradient = None
    official_difference = 0.0
    training_length = int(
        manifest.get(
            "train_length", max(r["frames"] for r in selected if r["split"] == "train")
        )
    )
    for entry in selected:
        path = Path(manifest["root"]) / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Audit input changed: {path}")
        clip = load_tap_clip(path)
        ids, queries = choose_queries(clip, max_queries, seed + 1000, entry["clip_id"])
        prefix = dict(
            clip_id=entry["clip_id"], source_id=entry["source_id"], split=entry["split"]
        )
        data_rows.append(dict(**prefix, **sampled_data_profile(clip, ids, queries)))
        traces = []
        for index, query in zip(ids, queries):
            trace, timing, identity = audited_query(model, clip, int(index), query)
            traces.append(trace)
            frames.extend(dict(**prefix, track_id=int(index), **row) for row in timing)
            identities.append(dict(**prefix, track_id=int(index), **identity))
        trace = {key: np.stack([r[key] for r in traces]) for key in traces[0]}
        rows, calibrated = analyze_traces(clip, ids, queries, trace)
        if official is not None:
            reference = official(
                queries[None],
                clip.occluded[ids][None],
                clip.points[ids][None] * 256,
                (trace["visibility_probability"] < 0.5)[None],
                trace["filtered"][None] * 256,
                "first",
            )
            current = next(
                r
                for r in rows
                if r["horizon"] == "all_future"
                and r["readout"] == "filtered"
                and r["visibility_rule"] == "predicted_0.5"
            )
            official_difference = max(
                official_difference,
                max(abs(float(v[0]) - current[k]) for k, v in reference.items()),
            )
            if official_difference > 1e-12:
                raise ValueError(
                    f"Pinned official evaluator disagrees: {official_difference}"
                )
        components.extend(dict(**prefix, **r) for r in rows)
        calibration.extend(dict(**prefix, **r) for r in calibrated)
        extras.extend(
            dict(**prefix, **r)
            for r in extra_component_rows(clip, ids, queries, trace, training_length)
        )
        if include_lk:
            controls = [
                baseline_rollout(LKTracker(), clip.frames, q, 1.0) for q in queries
            ]
            extras.append(
                dict(
                    **prefix,
                    condition="LK_DIAGNOSTIC_CONTROL",
                    value=None,
                    **tapvid_metrics(
                        queries,
                        clip.points[ids],
                        clip.occluded[ids],
                        np.stack([r["points"] for r in controls]),
                        np.stack([r["occluded"] for r in controls]),
                    ),
                )
            )
        if gradient is None and entry["split"] == "train":
            candidates = [
                offset
                for offset, (index, query) in enumerate(zip(ids, queries))
                if (
                    ~clip.occluded[
                        index, int(query[0]) + 1 : int(query[0]) + gradient_frames
                    ]
                ).any()
            ]
            if candidates:
                offset = candidates[0]
                gradient = dict(
                    **prefix,
                    track_id=int(ids[offset]),
                    **gradient_conflicts(
                        model, clip, int(ids[offset]), queries[offset], gradient_frames
                    ),
                )
                gradient["half_observation_control"] = gradient_conflicts(
                    model,
                    clip,
                    int(ids[offset]),
                    queries[offset],
                    gradient_frames,
                    fraction=0.5,
                )
        print(
            f"failure-audit {entry['split']} {entry['clip_id']} queries={len(ids)}",
            flush=True,
        )
    for name, rows in (
        ("components", components),
        ("calibration", calibration),
        ("heatmap_filter_frames", frames),
        ("self_match", identities),
        ("controls", extras),
        ("sampled_data", data_rows),
    ):
        write_table(output / f"{name}.csv", rows)
    summary = dict(
        checkpoint=provenance,
        fitting_device=fitting["device"],
        geometry=encoder_geometry(model),
        population={
            role: dict(
                source_videos=len(
                    {r["source_id"] for r in manifest["clips"] if r["split"] == role}
                ),
                clips=len([r for r in manifest["clips"] if r["split"] == role]),
            )
            for role in ("train", "validation")
        },
        fitting_steps=fitting["steps"],
        queries_per_update=fitting["queries_per_step"],
        train_window_frames=training_length,
        validation_frames=sorted(
            {r["frames"] for r in manifest["clips"] if r["split"] == "validation"}
        ),
        validation_fraction=fitting["validation_observation_fraction"],
        validation_query_limit=fitting["validation_queries"],
        validation_clip_limit=fitting["validation_clips"],
        final_test_accessed=False,
        weights_changed=False,
        cause_proven=False,
        official_evaluator_available=official is not None,
        maximum_official_metric_difference=official_difference if official else None,
        important="Alternate readouts, label oracles and posthoc thresholds are diagnostic controls; no model/threshold is selected and no benchmark score is claimed.",
    )
    write_json(output / "audit_summary.json", summary)
    write_json(output / "gradient_diagnosis.json", gradient or dict(defined=False))
    write_json(output / "mechanism_controls.json", mechanism_controls())
    write_json(
        output / "component_summary.json",
        component_summary(components, frames, identities),
    )
    torch.save(
        dict(
            schema_version=1,
            kind="tap_ats_diagnostic_snapshot",
            identity=saved["identity"],
            model_config=provenance["model_config"],
            state_dict={
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            step=provenance["step"],
            selection_complete=False,
            original_checkpoint_sha256=provenance["checkpoint_sha256"],
        ),
        output / "diagnostic_snapshot.pt",
    )
    write_failure_report(output, summary, components)
    complete_run(
        output, checkpoint_step=provenance["step"], audited_clips=len(selected)
    )
    return output


def component_summary(components, frames, identities):
    """Compute descriptive component gaps, without assigning causal percentages."""
    result = {}
    for role in ("train", "validation"):
        values = {}
        for readout in READOUTS:
            for rule in ("predicted_0.5", "oracle_visibility_DIAGNOSTIC_ONLY"):
                rows = [
                    r
                    for r in components
                    if r["split"] == role
                    and r["horizon"] == "all_future"
                    and r["readout"] == readout
                    and r["visibility_rule"] == rule
                    and r["metric_defined"]
                ]
                if rows:
                    values[f"{readout}__{rule}"] = dict(
                        aj=float(np.mean([r["average_jaccard"] for r in rows])),
                        within=float(
                            np.mean([r["average_pts_within_thresh"] for r in rows])
                        ),
                    )
        selected = [r for r in frames if r["split"] == role]
        self_matches = [r for r in identities if r["split"] == role]
        statistics = {
            key: float(np.median([r[key] for r in selected if r[key] is not None]))
            for key in (
                "gain_x",
                "gain_y",
                "effective_measurement_std_px",
                "entropy_normalized",
                "global_to_peak_distance_px",
            )
            if any(r[key] is not None for r in selected)
        }
        statistics.update(
            {
                key: float(np.median([r[key] for r in self_matches]))
                for key in (
                    "self_global_mean_error_px",
                    "self_peak_error_px",
                    "self_local_mean_error_px",
                )
                if self_matches
            }
        )
        visible = [r for r in selected if r["target_visible"]]
        if visible:
            statistics["target_cell_is_peak_fraction"] = float(
                np.mean([r["target_cell_is_peak"] for r in visible])
            )
            statistics["target_cell_rank_median"] = float(
                np.median([r["target_cell_rank"] for r in visible])
            )
        flags = []
        if values:
            base = values["filtered__predicted_0.5"]
            if (
                values["filtered__oracle_visibility_DIAGNOSTIC_ONLY"]["aj"] - base["aj"]
                > 0.05
            ):
                flags.append(
                    "Oracle visibility adds >5 AJ points; inspect visibility/calibration."
                )
            raw = values["measurement_global__predicted_0.5"]
            if raw["within"] - base["within"] > 0.05:
                flags.append(
                    "Raw localization exceeds filtering by >5 points; inspect dynamics/gain/R."
                )
            if (
                values["measurement_peak__predicted_0.5"]["within"] - raw["within"]
                > 0.05
            ):
                flags.append(
                    "Peak localization exceeds global expectation by >5 points; inspect multimodal/diffuse readouts."
                )
            if base["aj"] <= values["query_hold__predicted_0.5"]["aj"]:
                flags.append(
                    "Filtered AJ does not exceed query hold under identical predicted visibility."
                )
        result[role] = dict(
            video_macro_metrics=values,
            frame_query_weighted_median_diagnostics=statistics,
            descriptive_flags=flags,
            gap_threshold_pp=5,
            statistical_test=False,
            cause_proven=False,
        )
    return result


def write_failure_report(output, summary, components):
    """Present per-role macro-video diagnostics with explicit evidence boundaries."""
    text = [
        "# Frozen checkpoint failure audit",
        "",
        f"Checkpoint step: {summary['checkpoint']['step']}; fitting device: {summary['fitting_device']}; weights were not updated.",
        "",
        "AJ values below are percentages. The supplied checkpoint alone is audited; synthetic controls and local smoke checkpoints do not diagnose an unavailable server checkpoint.",
        "",
        "| Role | Readout | Visibility | Videos | AJ | Within threshold |",
        "|---|---|---|---:|---:|---:|",
    ]
    for role in ("train", "validation"):
        for readout in READOUTS:
            for rule in ("predicted_0.5", "oracle_visibility_DIAGNOSTIC_ONLY"):
                rows = [
                    r
                    for r in components
                    if r["split"] == role
                    and r["readout"] == readout
                    and r["visibility_rule"] == rule
                    and r["horizon"] == "all_future"
                    and r["metric_defined"]
                ]
                if rows:
                    text.append(
                        f"| {role} | {readout} | {rule} | {len(rows)} | {100*np.mean([r['average_jaccard'] for r in rows]):.2f} | {100*np.mean([r['average_pts_within_thresh'] for r in rows]):.2f} |"
                    )
    text.extend(
        [
            "",
            "## Interpretation rules",
            "",
            "- Train and validation both weak: inspect visual correspondence/readout, objective conflict and optimization. A broad generalization gap is not established by that pattern.",
            "- Train strong but validation weak: inspect source diversity, augmentation and overfitting; the small audited subset is not the whole training distribution.",
            "- Peak/local readout better than global mean: supports a heatmap ambiguity/averaging failure. These readouts reuse original logits and are not retrained alternatives.",
            "- Raw measurement better than filtered localization: inspect gain, effective R and learned dynamics. Frozen-state comparisons do not establish the efficacy of retraining without the filter.",
            "- Oracle visibility better: visibility errors cost AJ. Oracle labels are not a deployable improvement.",
            "- Accuracy drops beyond the training horizon: a descriptive horizon association; motion/visibility difficulty also changes over time.",
            "- High full-read visibility threshold sensitivity: checkpoint logs at threshold 0.5 can differ from the later validation-selected threshold. Do not choose a threshold from final-test labels.",
            "- One-hot cell-centre oracle is a quantization control, not an upper bound on continuous soft-argmax tracking.",
            "- Negative component-gradient cosines identify local objective conflict, not its historical frequency or causal contribution.",
            "",
            "## Controls",
            "",
            "See controls.csv for fixed visibility thresholds, horizon/motion/reappearance strata, label-cell oracle and optional matched LK control.",
            "See self_match.csv for same-image raw correspondence, heatmap_filter_frames.csv for entropy, target rank, gain and effective R, and gradient_diagnosis.json for objective-gradient norms/cosines.",
            "",
            "No cause is marked proven by this automated report.",
            "",
        ]
    )
    (output / "failure_report.md").write_text("\n".join(text))
