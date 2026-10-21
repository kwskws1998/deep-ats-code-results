"""Observed-frame-only wrappers for LK and an optional official TAPNext++ model."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from ats.artifacts import prepare_run, complete_run
from ats_vision.experiment import write_csv, write_json
from ats_vision.sampling import uniform_times
from ats_vision.tap_data import load_tap_clip, load_tap_manifest, sha256
from ats_vision.tap_metrics import tapvid_metrics
from ats_vision.tap_rollout import query_budget
from ats_vision.tap_training import choose_queries


class LKTracker:
    """Pyramidal Lucas-Kanade with a forward/backward consistency visibility check."""

    def initialize(self, frame, xy):
        """Store only the public query image and coordinate."""
        import cv2

        self.previous = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        self.point = np.asarray(xy, np.float32).reshape(1, 1, 2)
        return self.point.reshape(2).copy(), True

    def observe(self, frame):
        """Estimate flow between consecutive acquired images only."""
        import cv2

        current = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        point, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous, current, self.point, None, winSize=(21, 21), maxLevel=3
        )
        visible = False
        if (
            point is not None
            and status is not None
            and status[0, 0]
            and np.isfinite(point).all()
        ):
            backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                current, self.previous, point, None, winSize=(21, 21), maxLevel=3
            )
            visible = bool(
                backward is not None
                and backward_status[0, 0]
                and np.linalg.norm(backward - self.point) < 1.5
            )
            self.point = point
        self.previous = current
        return self.point.reshape(2).copy(), visible


class TAPNextPPTracker:
    """Use the official online API with a supplied checkpoint; no implicit download."""

    def __init__(self, checkpoint: Path, device="cuda", model_resolution=256):
        try:
            from tapnet.tapnextpp.votsp2026.model import TAPNextPP
        except ImportError as error:
            raise ImportError(
                "Install the pinned optional tapnet[torch] package described in docs/tapvid_ats.md."
            ) from error
        self.model = TAPNextPP.from_checkpoint(
            str(checkpoint), device=device, input_resolution=model_resolution
        )
        self.state = None

    def initialize(self, frame, xy):
        """Run one public query independently from other benchmark annotations."""
        self.state = None
        positions, visible, self.state = self.model.track_frame(
            frame[..., ::-1].copy(),
            query_points_xy=np.asarray(xy, np.float32).reshape(1, 2),
        )
        return np.asarray(positions).reshape(-1, 2)[0], bool(
            np.asarray(visible).reshape(-1)[0]
        )

    def observe(self, frame):
        """Advance the official recurrent model only when an image is acquired."""
        positions, visible, self.state = self.model.track_frame(
            frame[..., ::-1].copy(), state=self.state
        )
        return np.asarray(positions).reshape(-1, 2)[0], bool(
            np.asarray(visible).reshape(-1)[0]
        )


def baseline_rollout(
    tracker,
    frames,
    query,
    fraction,
    schedule="uniform",
    seed=17,
    extrapolation="velocity",
):
    """Adapt an online tracker to sparse reads using declared causal gap filling.

    Sparse observations are fed as consecutive inputs to the external tracker;
    this changes its usual cadence. Missing frames use hold or constant velocity,
    never interpolation from a future acquired frame.
    """
    if schedule not in {"uniform", "random"} or extrapolation not in {
        "hold",
        "velocity",
    }:
        raise ValueError("Unknown baseline schedule or extrapolation rule.")
    start = int(query[0])
    length = len(frames) - start
    budget = query_budget(length, fraction)
    clocks = uniform_times(length, budget)
    if schedule == "random":
        clocks = np.concatenate(
            (
                [0],
                np.sort(
                    np.random.default_rng(seed).choice(
                        np.arange(1, length), budget - 1, replace=False
                    )
                ),
            )
        )
    acquired = np.zeros(len(frames), bool)
    scale = np.array([frames.shape[2], frames.shape[1]])
    point = query[[2, 1]].astype(float)
    predictions = np.broadcast_to(point, (len(frames), 2)).copy()
    occluded = np.ones(len(frames), bool)
    last_read = start
    velocity = np.zeros(2)
    visible = True
    began = time.perf_counter()
    for clock in range(start, len(frames)):
        if clock - start in clocks:
            old = point.copy()
            if clock == start:
                xy, visible = tracker.initialize(frames[clock], point * scale)
            else:
                xy, visible = tracker.observe(frames[clock])
            point = np.asarray(xy) / scale
            if point.shape != (2,) or not np.isfinite(point).all():
                raise FloatingPointError(
                    "External tracker returned invalid coordinates."
                )
            if clock > start:
                velocity = (point - old) / (clock - last_read)
            last_read = clock
            acquired[clock] = True
        prediction = (
            point + (clock - last_read) * velocity
            if extrapolation == "velocity"
            else point
        )
        predictions[clock] = prediction
        occluded[clock] = not visible or np.any((prediction < 0) | (prediction > 1))
    return dict(
        points=predictions.astype(np.float32),
        occluded=occluded,
        acquired=acquired,
        wall_seconds=time.perf_counter() - began,
    )


def evaluate_baseline(
    data_manifest,
    output,
    backend="lk",
    checkpoint=None,
    split="test",
    fractions=(0.25, 0.5, 1.0),
    schedules=("uniform", "random"),
    extrapolation="velocity",
    max_queries=0,
    max_clips=0,
    device="cuda",
    model_resolution=256,
    seed=17,
):
    """Save official metrics under matched independent query acquisition budgets."""
    if (
        backend not in {"lk", "tapnextpp"}
        or split not in {"validation", "test"}
        or max_queries < 0
        or max_clips < 0
    ):
        raise ValueError("Invalid backend, split, or declared subset limit.")
    if backend == "tapnextpp" and (
        checkpoint is None or not Path(checkpoint).is_file()
    ):
        raise ValueError("Supply a local official TAPNext++ checkpoint.")
    if len(set(fractions)) != len(fractions) or any(
        not 0 < value <= 1 for value in fractions
    ):
        raise ValueError("Fractions must be distinct values in (0,1].")
    manifest = load_tap_manifest(data_manifest)
    entries = [entry for entry in manifest["clips"] if entry["split"] == split]
    entries = entries[:max_clips] if max_clips else entries
    if not entries:
        raise ValueError("No requested evaluation clips.")
    tracker = (
        LKTracker()
        if backend == "lk"
        else TAPNextPPTracker(checkpoint, device, model_resolution)
    )
    config = dict(
        data_manifest_sha256=sha256(data_manifest),
        backend=backend,
        split=split,
        fractions=list(fractions),
        schedules=list(schedules),
        extrapolation=extrapolation,
        max_queries=max_queries,
        max_clips=max_clips,
        seed=seed,
        device=device,
        model_resolution=model_resolution,
        checkpoint_sha256=sha256(checkpoint) if checkpoint else None,
    )
    output = Path(output)
    prepare_run(output, config, "tapvid_external_baseline")
    (output / "predictions").mkdir()
    rows = []
    for entry in entries:
        clip = load_tap_clip(Path(manifest["root"]) / entry["path"])
        ids, queries = choose_queries(clip, max_queries, seed, entry["clip_id"])
        for fraction in fractions:
            for schedule in schedules:
                results = [
                    baseline_rollout(
                        tracker,
                        clip.frames,
                        query,
                        fraction,
                        schedule,
                        seed + int(index),
                        extrapolation,
                    )
                    for index, query in zip(ids, queries)
                ]
                points = np.stack([value["points"] for value in results])
                occluded = np.stack([value["occluded"] for value in results])
                acquired = np.stack([value["acquired"] for value in results])
                metrics = tapvid_metrics(
                    queries, clip.points[ids], clip.occluded[ids], points, occluded
                )
                rows.append(
                    dict(
                        clip_id=entry["clip_id"],
                        dataset=entry["dataset"],
                        backend=backend,
                        schedule=schedule,
                        fraction=fraction,
                        queries=len(ids),
                        observations=int(acquired.sum()),
                        wall_seconds=sum(value["wall_seconds"] for value in results),
                        **metrics,
                    )
                )
                np.savez_compressed(
                    output
                    / "predictions"
                    / f"{entry['clip_id']}__{schedule}__{fraction:.6f}.npz",
                    points=points,
                    predicted_occluded=occluded,
                    acquired=acquired,
                    queries=queries,
                    track_ids=ids,
                )
    write_csv(output / "metrics.csv", rows)
    write_json(
        output / "protocol.json",
        dict(
            modified_protocol="sparse observed frames with causal hold/CV extrapolation; distinct from published full-observation numbers",
            per_query_independent=True,
            future_interpolation=False,
            input_resolution=manifest["image_size"],
            support_points=0,
            pretraining="external pretrained weights have a different training budget; report provenance separately",
            backend_code=(
                "official TAPNext++ API, optional and not locally weight-validated"
                if backend == "tapnextpp"
                else "OpenCV LK forward/backward check"
            ),
        ),
    )
    complete_run(output, metric_rows=len(rows), videos=len(entries))
    return output
