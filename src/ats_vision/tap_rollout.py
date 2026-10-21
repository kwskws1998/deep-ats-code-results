"""Causal single-query inference and label-supervised point-tracking objectives."""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as functional

from ats_vision.model import gaussian_nll
from ats_vision.sampling import SamplingPolicy
from ats_vision.tap_model import TAPATSModel


def frame_tensor(frames: np.ndarray, clock: int, device) -> Tensor:
    """Convert only one acquired RGB image to a network input."""
    return (
        torch.as_tensor(np.ascontiguousarray(frames[clock]), device=device)
        .permute(2, 0, 1)[None]
        .float()
        / 255
    )


def query_budget(length: int, fraction: float) -> int:
    """Charge the query frame and round a fractional budget up, with no free read."""
    if length < 1 or not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("Need a nonempty horizon and 0 < fraction <= 1.")
    return max(1, int(math.ceil(length * fraction)))


@torch.no_grad()
def point_rollout(
    model: TAPATSModel,
    frames: np.ndarray,
    query: np.ndarray,
    fraction: float,
    policy_name: str = "ats",
    threshold: float = 0.01,
    max_gap: int = 16,
    seed: int = 17,
    fixed_q=None,
    fixed_r=None,
) -> dict:
    """Track using a public [t,y,x] query; no targets, masks, or other queries enter."""
    if query.shape != (3,) or not np.isfinite(query).all():
        raise ValueError("Expected a finite [t,y,x] query.")
    start = int(query[0])
    if (
        start != query[0]
        or not 0 <= start < len(frames)
        or np.any(query[1:] < 0)
        or np.any(query[1:] > 1)
    ):
        raise ValueError("Invalid public query.")
    length = len(frames) - start
    budget = query_budget(length, fraction)
    policy = SamplingPolicy(
        policy_name,
        length,
        budget,
        model.transition.detach().cpu().numpy().astype(float),
        model.acceleration_map.detach().cpu().numpy().astype(float),
        seed,
        max_gap,
        threshold,
        fixed_q,
        fixed_r,
    )
    device = model.transition.device
    initial_frame = frame_tensor(frames, start, device)
    query_xy = initial_frame.new_tensor(query[[2, 1]])[None]
    begin = time.perf_counter()
    belief = model.initialize(initial_frame, query_xy)
    means, covariances, logits = [], [], []
    acquired = np.zeros(len(frames), bool)
    acquired[start] = True
    used, last_read = 1, 0
    last_r = np.asarray(fixed_r).copy() if fixed_r is not None else np.full(2, 1e-3)
    q_reads, r_reads = [], []
    decisions = []
    policy_seconds = 0.0
    next_read = None
    for clock in range(length):
        if clock:
            belief = model.predict(belief)
        elapsed = belief.mean.new_tensor([clock - last_read])
        if clock == next_read:
            belief, observation = model.observe(
                belief, frame_tensor(frames, start + clock, device), elapsed
            )
            last_r = observation["effective_r"][0].detach().cpu().numpy().astype(float)
            visibility = observation["visibility_logit"]
            acquired[start + clock] = True
            last_read = clock
            used += 1
        else:
            visibility = (
                belief.mean.new_full((1,), 20.0)
                if not clock
                else model.visibility(belief, elapsed)
            )
        if acquired[start + clock]:
            q_reads.append(model.process_variance(belief)[0].detach().cpu().numpy())
            r_reads.append(last_r.copy())
            policy_start = time.perf_counter()
            next_read, scores = policy.next_time(
                clock,
                used,
                belief.covariance[0].detach().cpu().numpy().astype(float),
                model.process_variance(belief)[0].detach().cpu().numpy().astype(float),
                last_r,
            )
            policy_seconds += time.perf_counter() - policy_start
            decisions.append(
                dict(
                    time=start + clock,
                    next_time=None if next_read is None else start + next_read,
                    scores={str(start + key): value for key, value in scores.items()},
                )
            )
        means.append(belief.mean[0])
        covariances.append(belief.covariance[0])
        logits.append(visibility[0])
    if used != budget:
        raise AssertionError("Acquisition budget violated.")
    means = torch.stack(means)
    covariance = torch.stack(covariances)
    visibility = torch.stack(logits)
    full_points = (
        np.broadcast_to(query[[2, 1]], (len(frames), 2)).copy().astype(np.float32)
    )
    full_logits = np.full(len(frames), -20, np.float32)
    full_covariance = np.zeros((len(frames), 2, 2), np.float32)
    full_points[start:] = means[:, :2].detach().cpu().numpy()
    full_logits[start:] = visibility.detach().cpu().numpy()
    full_covariance[start:] = covariance[:, :2, :2].detach().cpu().numpy()
    return dict(
        points=full_points,
        visibility_logits=full_logits,
        covariance=full_covariance,
        acquired=acquired,
        observations=used,
        q=model.process_variance(belief)[0].detach().cpu().numpy(),
        last_r=last_r,
        q_reads=np.asarray(q_reads),
        r_reads=np.asarray(r_reads),
        decisions=decisions,
        policy_seconds=policy_seconds,
        wall_seconds=time.perf_counter() - begin,
    )


def point_training_loss(
    model: TAPATSModel,
    frames: np.ndarray,
    query: np.ndarray,
    target: np.ndarray,
    occluded: np.ndarray,
    observed: np.ndarray,
) -> dict[str, Tensor]:
    """Supervise visible coordinates/heatmaps and all-frame occlusion, excluding query.

    No exact velocities, acceleration labels, clean images, or true Q/R enter.
    Point labels after missing observations supervise prediction, not inference.
    """
    start = int(query[0])
    if start != query[0] or not observed[start] or observed[:start].any():
        raise ValueError("Training mask must start with the public query observation.")
    device = model.transition.device
    initial = frame_tensor(frames, start, device)
    belief = model.initialize(initial, initial.new_tensor(query[[2, 1]])[None])
    last_read = start
    coordinate_terms, visibility_terms, measurement_terms, heatmap_terms = (
        [],
        [],
        [],
        [],
    )
    for clock in range(start + 1, len(frames)):
        belief = model.predict(belief)
        elapsed = belief.mean.new_tensor([clock - last_read])
        label = belief.mean.new_tensor(target[clock])[None]
        visible = not bool(occluded[clock])
        if observed[clock]:
            belief, observation = model.observe(
                belief, frame_tensor(frames, clock, device), elapsed
            )
            logit = observation["visibility_logit"]
            last_read = clock
            if visible:
                measurement_terms.append(
                    gaussian_nll(
                        label,
                        observation["measurement"],
                        torch.diag_embed(observation["r"]),
                    ).mean()
                )
                height, width = observation["logits"].shape[-2:]
                column = int(np.clip(target[clock, 0] * width, 0, width - 1))
                row = int(np.clip(target[clock, 1] * height, 0, height - 1))
                cell = torch.tensor([row * width + column], device=device)
                heatmap_terms.append(
                    functional.cross_entropy(observation["logits"].flatten(1), cell)
                )
        else:
            logit = model.visibility(belief, elapsed)
        visibility_terms.append(
            functional.binary_cross_entropy_with_logits(
                logit, logit.new_tensor([float(visible)])
            )
        )
        if visible:
            coordinate_terms.append(
                0.05
                * gaussian_nll(
                    label, belief.mean[:, :2], belief.covariance[:, :2, :2]
                ).mean()
                + 10 * functional.smooth_l1_loss(belief.mean[:, :2], label, beta=0.02)
            )
    zero = next(model.parameters()).sum() * 0
    mean_or_zero = lambda terms: torch.stack(terms).mean() if terms else zero
    coordinate = mean_or_zero(coordinate_terms)
    visibility = mean_or_zero(visibility_terms)
    measurement = mean_or_zero(measurement_terms)
    heatmap = mean_or_zero(heatmap_terms)
    return dict(
        total=coordinate + visibility + 0.1 * measurement + heatmap,
        coordinate=coordinate,
        visibility=visibility,
        measurement=measurement,
        heatmap=heatmap,
    )
