"""Full-observation training/validation diagnosis with frozen weights and explicit oracles.

Alternate readouts reuse logits from the original filter trajectory. They are
offline diagnostics, not independently executed replacement tracking models.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from ats.artifacts import complete_run, prepare_run
from ats_vision.experiment import write_json
from ats_vision.tap_calibration import position_calibration, visibility_calibration
from ats_vision.tap_checkpoint import load_diagnostic_checkpoint
from ats_vision.tap_data import load_tap_clip, load_tap_manifest, sha256
from ats_vision.tap_metrics import tapvid_metrics
from ats_vision.tap_readout import heatmap_readouts
from ats_vision.tap_rollout import frame_tensor
from ats_vision.tap_training import choose_queries, configure_runtime

HORIZONS = {
    "all_future": (1, None),
    "1_15": (1, 15),
    "16_30": (16, 30),
    "31_60": (31, 60),
    "61_plus": (61, None),
}
READOUTS = (
    "filtered",
    "measurement_global",
    "measurement_peak",
    "measurement_local",
    "query_hold",
)


def write_table(path: Path, rows: list[dict]) -> None:
    """Allow nullable diagnostic fields when an interval has no scored visible targets."""
    if not rows:
        raise ValueError("No diagnostic rows to save.")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def full_observation_trace(model, frames, query, observer=None) -> dict:
    """Trace original causal filtering; targets never enter this function."""
    start, length = int(query[0]), len(frames)
    if start != query[0] or not 0 <= start < length or not np.isfinite(query).all():
        raise ValueError("Invalid public query.")
    initial = frame_tensor(frames, start, model.transition.device)
    belief = model.initialize(initial, initial.new_tensor(query[[2, 1]])[None])
    result = {
        key: np.broadcast_to(query[[2, 1]], (length, 2)).copy().astype(np.float32)
        for key in READOUTS
    }
    result.update(
        covariance=np.zeros((length, 2, 2), np.float32),
        visibility_probability=np.zeros(length, np.float32),
        r=np.zeros((length, 2), np.float32),
        effective_r=np.zeros((length, 2), np.float32),
        entropy=np.zeros(length, np.float32),
        peak_probability=np.zeros(length, np.float32),
        measurement_valid=np.zeros(length, bool),
    )
    result["visibility_probability"][start] = 1
    result["covariance"][start] = belief.covariance[0, :2, :2].cpu().numpy()
    for clock in range(start + 1, length):
        prior = model.predict(belief)
        belief, observation = model.observe(
            prior,
            frame_tensor(frames, clock, model.transition.device),
            initial.new_ones(1),
        )
        if observer is not None:
            observer(clock, prior, belief, observation)
        readouts = heatmap_readouts(observation["logits"])
        result["filtered"][clock] = belief.mean[0, :2].cpu().numpy()
        result["measurement_global"][clock] = (
            observation["measurement"][0].cpu().numpy()
        )
        result["measurement_peak"][clock] = readouts["peak"][0].cpu().numpy()
        result["measurement_local"][clock] = readouts["local_mean"][0].cpu().numpy()
        result["covariance"][clock] = belief.covariance[0, :2, :2].cpu().numpy()
        result["visibility_probability"][clock] = (
            observation["visibility_logit"].sigmoid()[0].cpu().numpy()
        )
        for key in ("r", "effective_r"):
            result[key][clock] = observation[key][0].cpu().numpy()
        for key in ("entropy", "peak_probability"):
            result[key][clock] = readouts[key][0].cpu().numpy()
        result["measurement_valid"][clock] = True
    for key, value in result.items():
        if not np.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite diagnostic trace: {key}.")
    return result


def horizon_mask(queries, length, low, high):
    """Restrict evaluation to elapsed physical frames after each independent query."""
    elapsed = np.arange(length)[None] - queries[:, 0, None]
    mask = elapsed >= low
    if high is not None:
        mask &= elapsed <= high
    return mask


def analyze_traces(clip, ids, queries, traces) -> tuple[list[dict], list[dict]]:
    """Separate localization, visibility, temporal drift, and covariance calibration."""
    target, occluded = clip.points[ids], clip.occluded[ids]
    rows, calibration = [], []
    for horizon, (low, high) in HORIZONS.items():
        mask = horizon_mask(queries, len(clip.frames), low, high)
        visible_mask = mask & ~occluded
        p = traces["visibility_probability"]
        calibration.append(
            dict(
                horizon=horizon,
                **visibility_calibration(p, ~occluded, mask),
                **position_calibration(
                    target, traces["filtered"], traces["covariance"], visible_mask
                ),
            )
        )
        for readout in READOUTS:
            error = np.linalg.norm((traces[readout] - target) * 256, axis=-1)[
                visible_mask
            ]
            for rule, predicted_occluded in (
                ("predicted_0.5", p < 0.5),
                ("oracle_visibility_DIAGNOSTIC_ONLY", occluded),
                ("always_visible", np.zeros_like(occluded)),
            ):
                row = dict(
                    horizon=horizon,
                    readout=readout,
                    visibility_rule=rule,
                    scored_frames=int(mask.sum()),
                    visible_frames=int(visible_mask.sum()),
                    metric_defined=bool(mask.any() and visible_mask.any()),
                    error_median_px=float(np.median(error)) if len(error) else None,
                    error_p95_px=(
                        float(np.quantile(error, 0.95)) if len(error) else None
                    ),
                    error_mean_px=float(error.mean()) if len(error) else None,
                    average_jaccard=None,
                    average_pts_within_thresh=None,
                    occlusion_accuracy=None,
                )
                if row["metric_defined"]:
                    row.update(
                        tapvid_metrics(
                            queries,
                            target,
                            occluded,
                            traces[readout],
                            predicted_occluded,
                            evaluation_mask=mask,
                        )
                    )
                rows.append(row)
    return rows, calibration


def diagnostic_flags(rows) -> list[str]:
    """Use declared descriptive 10-point gaps to suggest checks, never select a model."""
    selected = {
        (row["readout"], row["visibility_rule"]): row
        for row in rows
        if row["horizon"] == "all_future" and row["metric_defined"]
    }
    if not selected:
        return ["No future-visible targets in this diagnostic subset."]
    base = selected[("filtered", "predicted_0.5")]
    flags = []
    if (
        selected[("filtered", "oracle_visibility_DIAGNOSTIC_ONLY")]["average_jaccard"]
        - base["average_jaccard"]
        > 0.1
    ):
        flags.append(
            "Oracle visibility raises AJ by >10 percentage points: inspect visibility errors/calibration."
        )
    raw = selected[("measurement_global", "predicted_0.5")]
    if raw["average_pts_within_thresh"] - base["average_pts_within_thresh"] > 0.1:
        flags.append(
            "Raw measurement localization exceeds filtering by >10 points: inspect filter gain and Q/R."
        )
    peak = selected[("measurement_peak", "predicted_0.5")]
    if peak["average_pts_within_thresh"] - raw["average_pts_within_thresh"] > 0.1:
        flags.append(
            "Peak localization exceeds global expectation by >10 points: inspect diffuse/multimodal heatmaps."
        )
    return flags or [
        "No >10-point component gap was detected; this does not establish correct calibration or a strong tracker."
    ]


def plot_diagnostic(clip, query, target, occluded, trace, output: Path):
    """Overlay fixed evenly spaced frames using labels only after causal predictions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start = int(query[0])
    clocks = np.unique(
        np.linspace(
            start, len(clip.frames) - 1, min(8, len(clip.frames) - start)
        ).astype(int)
    )
    figure, axes = plt.subplots(2, 4, figsize=(12, 6))
    for axis in axes.flat:
        axis.axis("off")
    height, width = clip.frames.shape[1:3]
    for axis, clock in zip(axes.flat, clocks):
        axis.imshow(clip.frames[clock])
        if not occluded[clock]:
            axis.scatter(
                *(target[clock] * [width, height]),
                marker="*",
                color="lime",
                s=80,
                label="target",
            )
        for key, color in (
            ("filtered", "red"),
            ("measurement_global", "cyan"),
            ("measurement_peak", "yellow"),
        ):
            axis.scatter(
                *(trace[key][clock] * [width, height]),
                marker="x",
                color=color,
                s=40,
                label=key,
            )
        axis.set_title(
            f"t={clock}; visible p={trace['visibility_probability'][clock]:.2f}",
            fontsize=9,
        )
        axis.set_xlim(0, width)
        axis.set_ylim(height, 0)
    axes.flat[0].legend(fontsize=6, loc="upper left")
    figure.suptitle("Frozen-weight diagnostic on a declared training/validation subset")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def diagnose_checkpoint(
    data_manifest: Path,
    checkpoint: Path,
    output: Path,
    split="validation",
    max_clips=2,
    max_queries=4,
    device="cpu",
    threads=2,
    seed=17,
) -> Path:
    """Read a live or selected snapshot from a separate checkout without editing fitting artifacts."""
    if split not in {"train", "validation"} or max_clips < 1 or max_queries < 1:
        raise ValueError(
            "Diagnostics require a bounded training/validation subset; final test is forbidden."
        )
    configure_runtime(device, threads, seed)
    manifest = load_tap_manifest(data_manifest, verify=False)
    model, saved, provenance = load_diagnostic_checkpoint(
        checkpoint, data_manifest, device
    )
    entries = [entry for entry in manifest["clips"] if entry["split"] == split][
        :max_clips
    ]
    if not entries:
        raise ValueError("No requested diagnostic split.")
    output = Path(output).resolve()
    if output.is_relative_to(Path(checkpoint).resolve().parent):
        raise ValueError("Save diagnostics outside the fitting run directory.")
    config = dict(
        data_manifest=str(Path(data_manifest).resolve()),
        data_manifest_sha256=sha256(data_manifest),
        split=split,
        max_clips=max_clips,
        max_queries=max_queries,
        device=device,
        threads=threads,
        seed=seed,
        checkpoint_provenance=provenance,
    )
    prepare_run(output, config, "tapvid_diagnostic_not_benchmark")
    all_rows, all_calibration, flags = [], [], {}
    for entry in entries:
        path = Path(manifest["root"]) / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Diagnostic input changed: {path}")
        clip = load_tap_clip(path)
        ids, queries = choose_queries(clip, max_queries, seed + 1000, entry["clip_id"])
        outputs = [
            full_observation_trace(model, clip.frames, query) for query in queries
        ]
        traces = {key: np.stack([item[key] for item in outputs]) for key in outputs[0]}
        rows, calibration = analyze_traces(clip, ids, queries, traces)
        all_rows.extend(
            dict(clip_id=entry["clip_id"], split=split, **row) for row in rows
        )
        all_calibration.extend(
            dict(clip_id=entry["clip_id"], split=split, **row) for row in calibration
        )
        flags[entry["clip_id"]] = diagnostic_flags(rows)
        np.savez_compressed(
            output / f"{entry['clip_id']}_trace.npz",
            queries=queries,
            track_ids=ids,
            **traces,
        )
        plot_diagnostic(
            clip,
            queries[0],
            clip.points[ids[0]],
            clip.occluded[ids[0]],
            outputs[0],
            output / f"{entry['clip_id']}_overlay.png",
        )
        print(f"diagnosed {entry['clip_id']} queries={len(ids)}", flush=True)
    write_table(output / "components.csv", all_rows)
    write_table(output / "calibration.csv", all_calibration)
    write_json(
        output / "diagnosis.json",
        dict(flags=flags, heuristic_gap_threshold_pp=10, cause_proven=False),
    )
    write_json(
        output / "protocol.json",
        dict(
            split=split,
            final_test_accessed=False,
            weights_changed=False,
            observation_fraction=1.0,
            readout_limit="Alternate readouts reuse logits under original global-mean filter states; not end-to-end alternate models.",
            oracle_limit="Oracle visibility uses labels after predictions and is diagnostic only, never a deployable or benchmark score.",
            horizon_limit="Each horizon is a separately masked diagnostic; all_future alone uses the full first-query evaluation mask.",
            aggregation="Per-video rows; no pooled-video ranking or model selection is performed.",
        ),
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
    text = [
        "# Frozen checkpoint diagnosis",
        "",
        f"Split: {split}; checkpoint step: {provenance['step']}; source checkpoint SHA256: {provenance['checkpoint_sha256']}.",
        "",
        "Scores below are percentages. Oracle visibility is diagnostic only. Alternate readouts reuse the original filter's logits; they are not end-to-end replacement models.",
        "",
        "| Clip | Readout | Visibility | AJ | Within threshold | OA | Median position error (px) |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in all_rows:
        if row["horizon"] == "all_future" and row["metric_defined"]:
            text.append(
                f"| {row['clip_id']} | {row['readout']} | {row['visibility_rule']} | {100 * row['average_jaccard']:.2f} | {100 * row['average_pts_within_thresh']:.2f} | {100 * row['occlusion_accuracy']:.2f} | {row['error_median_px']:.2f} |"
            )
    text.extend(["", "Heuristic flags suggest checks and do not prove causes:", ""])
    for clip_id, messages in flags.items():
        text.extend(f"- {clip_id}: {message}" for message in messages)
    (output / "diagnostic_report.md").write_text("\n".join(text) + "\n")
    complete_run(
        output, diagnostic_clips=len(entries), checkpoint_step=provenance["step"]
    )
    return output
