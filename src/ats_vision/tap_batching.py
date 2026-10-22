"""True query microbatches with independent masks, beliefs, and loss denominators.

Queries in a microbatch share a video and query clock. Unacquired images never
enter that query's encoder. Loss vectors retain the sequential per-query means.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as functional

from ats_vision.model import gaussian_nll
from ats_vision.tap_model import PointBelief, TAPATSModel
from ats_vision.tap_rollout import frame_tensor
from ats_vision.tap_features import EncodedFrames
from ats_vision.tap_fast_math import gaussian_nll2

LOSS_NAMES = ("total", "coordinate", "visibility", "measurement", "heatmap")


def subset_belief(belief: PointBelief, indices: Tensor) -> PointBelief:
    """Select independent query states without changing the temporal graph."""
    return PointBelief(
        *(
            getattr(belief, key).index_select(0, indices)
            for key in ("mean", "covariance", "context", "descriptor")
        )
    )


def replace_belief(
    belief: PointBelief, indices: Tensor, update: PointBelief
) -> PointBelief:
    """Scatter unique observed-query states while preserving unobserved forecasts."""
    return PointBelief(
        *(
            getattr(belief, key).index_copy(0, indices, getattr(update, key))
            for key in ("mean", "covariance", "context", "descriptor")
        )
    )


def query_microbatches(queries: np.ndarray, maximum: int) -> list[np.ndarray]:
    """Group by public query clock, retaining original query indices and a size cap."""
    if maximum < 1 or queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError("Need positive microbatch size and [N,3] queries.")
    if maximum == 1:
        return [np.asarray([index]) for index in range(len(queries))]
    groups = []
    for start in sorted(set(queries[:, 0])):
        indices = np.flatnonzero(queries[:, 0] == start)
        groups.extend(
            indices[offset : offset + maximum]
            for offset in range(0, len(indices), maximum)
        )
    return groups


def point_batch_training_loss(
    model: TAPATSModel,
    frames: np.ndarray,
    queries: np.ndarray,
    targets: np.ndarray,
    occluded: np.ndarray,
    observed: np.ndarray,
    shared_features: bool = False,
    encoder_chunk_size: int = 8,
) -> dict[str, Tensor]:
    """Return one loss per query, equivalent to separate scalar training calls."""
    count, length = len(queries), len(frames)
    if (
        count < 1
        or queries.shape != (count, 3)
        or not np.isfinite(queries).all()
        or targets.shape != (count, length, 2)
        or not np.isfinite(targets).all()
        or occluded.shape != (count, length)
        or occluded.dtype != bool
        or observed.shape != (count, length)
        or observed.dtype != bool
        or np.any(queries[:, 0] != queries[0, 0])
        or queries[0, 0] != int(queries[0, 0])
        or not 0 <= queries[0, 0] < length
        or np.any((queries[:, 1:] < 0) | (queries[:, 1:] > 1))
    ):
        raise ValueError("Invalid same-clock query microbatch or supervision arrays.")
    start = int(queries[0, 0])
    if not observed[:, start].all() or observed[:, :start].any():
        raise ValueError("Every mask must begin with its public query observation.")
    device = model.transition.device
    if shared_features:
        if getattr(model, "execution_backend", None) != "shared":
            raise ValueError("Shared features require the compatible shared backend.")
        features = EncodedFrames(model, frames, observed.any(0), encoder_chunk_size)
        initial = features[start].expand(count, -1, -1, -1)
        belief = model.initialize_encoded(
            initial, initial.new_tensor(queries[:, [2, 1]])
        )
    else:
        initial = frame_tensor(frames, start, device).expand(count, -1, -1, -1)
        belief = model.initialize(initial, initial.new_tensor(queries[:, [2, 1]]))
    nll = gaussian_nll2 if shared_features else gaussian_nll
    sums = {key: initial.new_zeros(count) for key in LOSS_NAMES[1:]}
    denominators = {key: np.zeros(count, np.int64) for key in LOSS_NAMES[1:]}
    labels_table = initial.new_tensor(targets)
    visibility_table = initial.new_tensor(~occluded)
    previous = np.maximum.accumulate(
        np.where(observed, np.arange(length), start), axis=1
    )
    previous = np.column_stack((np.full(count, start), previous[:, :-1]))
    elapsed_table = initial.new_tensor(np.arange(length)[None, :] - previous)
    for clock in range(start + 1, length):
        belief = model.predict(belief)
        elapsed = elapsed_table[:, clock]
        labels = labels_table[:, clock]
        visible = np.flatnonzero(~occluded[:, clock])
        reads = np.flatnonzero(observed[:, clock])
        missing = np.flatnonzero(~observed[:, clock])
        logits = initial.new_zeros(count)
        if len(reads):
            read_ids = torch.as_tensor(reads, device=device)
            state = subset_belief(belief, read_ids)
            read_elapsed = elapsed.index_select(0, read_ids)
            if shared_features:
                update, observation = model.observe_encoded(
                    state, features[clock].expand(len(reads), -1, -1, -1), read_elapsed
                )
            else:
                update, observation = model.observe(
                    state,
                    frame_tensor(frames, clock, device).expand(len(reads), -1, -1, -1),
                    read_elapsed,
                )
            belief = replace_belief(belief, read_ids, update)
            logits = logits.index_copy(0, read_ids, observation["visibility_logit"])
            local_visible = np.flatnonzero(~occluded[reads, clock])
            if len(local_visible):
                local = torch.as_tensor(local_visible, device=device)
                global_ids = read_ids.index_select(0, local)
                losses = nll(
                    labels.index_select(0, global_ids),
                    observation["measurement"].index_select(0, local),
                    torch.diag_embed(observation["r"].index_select(0, local)),
                )
                sums["measurement"] = sums["measurement"].index_add(
                    0, global_ids, losses
                )
                height, width = observation["logits"].shape[-2:]
                xy = targets[reads[local_visible], clock]
                cells = np.clip(
                    (xy[:, 1] * height).astype(np.int64), 0, height - 1
                ) * width + np.clip((xy[:, 0] * width).astype(np.int64), 0, width - 1)
                losses = functional.cross_entropy(
                    observation["logits"].index_select(0, local).flatten(1),
                    torch.as_tensor(cells, device=device),
                    reduction="none",
                )
                sums["heatmap"] = sums["heatmap"].index_add(0, global_ids, losses)
                for key in ("measurement", "heatmap"):
                    denominators[key][reads[local_visible]] += 1
        if len(missing):
            missing_ids = torch.as_tensor(missing, device=device)
            missing_logits = model.visibility(
                subset_belief(belief, missing_ids), elapsed.index_select(0, missing_ids)
            )
            logits = logits.index_copy(0, missing_ids, missing_logits)
        sums["visibility"] = sums[
            "visibility"
        ] + functional.binary_cross_entropy_with_logits(
            logits, visibility_table[:, clock], reduction="none"
        )
        denominators["visibility"] += 1
        if len(visible):
            indices = torch.as_tensor(visible, device=device)
            losses = 0.05 * nll(
                labels.index_select(0, indices),
                belief.mean.index_select(0, indices)[:, :2],
                belief.covariance.index_select(0, indices)[:, :2, :2],
            )
            losses = losses + 10 * functional.smooth_l1_loss(
                belief.mean.index_select(0, indices)[:, :2],
                labels.index_select(0, indices),
                beta=0.02,
                reduction="none",
            ).mean(-1)
            sums["coordinate"] = sums["coordinate"].index_add(0, indices, losses)
            denominators["coordinate"][visible] += 1
    zero = next(model.parameters()).sum() * 0
    means = {
        key: value / initial.new_tensor(np.maximum(denominators[key], 1)) + zero
        for key, value in sums.items()
    }
    means["total"] = (
        means["coordinate"]
        + means["visibility"]
        + 0.1 * means["measurement"]
        + means["heatmap"]
    )
    return means


class LossAccumulator:
    """Record per-query loss components over an entire logging interval."""

    def __init__(self):
        self.count = 0
        self.sums = None

    def add(self, terms: dict[str, Tensor]) -> None:
        """Detach only diagnostics; never change the graph used for fitting."""
        count = terms["total"].numel()
        values = torch.stack([terms[key].detach().double().sum() for key in LOSS_NAMES])
        self.count += count
        self.sums = values if self.sums is None else self.sums + values

    def summary(self) -> dict:
        """Report query-weighted means and clear the completed interval."""
        if not self.count:
            raise ValueError("No training losses have been accumulated.")
        values = self.sums.cpu().numpy()
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite training loss component.")
        result = {
            "loss_" + key: float(value / self.count)
            for key, value in zip(LOSS_NAMES, values)
        }
        result.update(loss=result["loss_total"], training_queries=self.count)
        self.count = 0
        self.sums = None
        return result
