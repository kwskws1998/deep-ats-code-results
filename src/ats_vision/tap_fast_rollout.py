"""Causal query batches with shared encoder work and deferred result transfers.

Static schedules can encode their acquired-frame union in temporal chunks.
Adaptive schedules encode only the current acquired image, then transfer one
packed belief/noise tensor per decision time. No target arrays enter this API.
Every image-query acquisition is still charged, even when computation is shared.
"""

import time

import numpy as np
import torch

from ats_vision.sampling import SamplingPolicy
from ats_vision.tap_batching import query_microbatches, replace_belief, subset_belief
from ats_vision.tap_features import EncodedFrames
from ats_vision.tap_rollout import frame_tensor, query_budget


@torch.no_grad()
def shared_point_rollouts(
    model,
    frames,
    queries,
    fraction,
    policy_name="uniform",
    threshold=0.01,
    max_gap=16,
    seeds=None,
    fixed_q=None,
    fixed_r=None,
    query_batch_size=8,
    encoder_chunk_size=8,
):
    """Return the reference per-query records, with amortized batch wall times."""
    queries = np.asarray(queries)
    count, length = len(queries), len(frames)
    if (
        getattr(model, "execution_backend", None) != "shared"
        or queries.shape != (count, 3)
        or count < 1
        or not np.isfinite(queries).all()
        or np.any(queries[:, 0] != np.floor(queries[:, 0]))
        or np.any((queries[:, 0] < 0) | (queries[:, 0] >= length))
        or np.any((queries[:, 1:] < 0) | (queries[:, 1:] > 1))
        or query_batch_size < 1
        or encoder_chunk_size < 1
    ):
        raise ValueError("Invalid public queries or shared execution settings.")
    seeds = [17] * count if seeds is None else list(seeds)
    if len(seeds) != count:
        raise ValueError("Need one independent policy seed per query.")
    begin = time.perf_counter()
    transition = model.transition.cpu().numpy().astype(float)
    acceleration = model.acceleration_map.cpu().numpy().astype(float)
    policies = [
        SamplingPolicy(
            policy_name,
            length - int(query[0]),
            query_budget(length - int(query[0]), fraction),
            transition,
            acceleration,
            seed,
            max_gap,
            threshold,
            fixed_q,
            fixed_r,
        )
        for query, seed in zip(queries, seeds)
    ]
    static = policy_name in {"uniform", "random"} or fraction == 1
    masks = np.zeros((count, length), bool)
    if static:
        for index, (query, policy) in enumerate(zip(queries, policies)):
            masks[index, int(query[0]) + policy._times] = True
        cache = EncodedFrames(model, frames, masks.any(0), encoder_chunk_size)
    else:
        cache = None
    setup_seconds = time.perf_counter() - begin
    outputs = [None] * count
    for group in query_microbatches(queries, query_batch_size):
        rows = _group_rollout(
            model,
            frames,
            queries[group],
            [policies[i] for i in group],
            masks[group] if static else None,
            cache,
            fixed_r,
        )
        for index, row in zip(group, rows):
            row["wall_seconds"] += setup_seconds / count
            outputs[int(index)] = row
    return outputs


def _group_rollout(model, frames, queries, policies, static_masks, cache, fixed_r):
    """Advance same-clock independent states; synchronize only adaptive decisions."""
    begin = time.perf_counter()
    count, total = len(queries), len(frames)
    start = int(queries[0, 0])
    length = total - start
    first = (
        cache[start]
        if cache is not None
        else model.frame_features(frame_tensor(frames, start, model.transition.device))
    )
    belief = model.initialize_encoded(
        first.expand(count, -1, -1, -1), first.new_tensor(queries[:, [2, 1]])
    )
    used = np.ones(count, np.int64)
    last_read = np.zeros(count, np.int64)
    next_read = np.full(count, -1, np.int64)
    initial_r = np.asarray(fixed_r).copy() if fixed_r is not None else np.full(2, 1e-3)
    last_r = np.broadcast_to(initial_r, (count, 2)).copy()
    r_tensor = first.new_tensor(last_r)
    acquired = np.zeros((count, total), bool)
    acquired[:, start] = True
    decisions = [[] for _ in range(count)]
    means, covariances, logits, noise_rows, r_rows = [], [], [], [], []
    policy_seconds = 0.0
    for clock in range(length):
        if clock:
            belief = model.predict(belief)
        elapsed = first.new_tensor(clock - last_read)
        reads = (
            (
                np.flatnonzero(static_masks[:, start + clock])
                if static_masks is not None
                else np.flatnonzero(next_read == clock)
            )
            if clock
            else np.arange(count)
        )
        visibility = first.new_full((count,), 20.0)
        if clock and len(reads):
            read_ids = torch.as_tensor(reads, device=first.device)
            features = (
                cache[start + clock]
                if cache is not None
                else model.frame_features(
                    frame_tensor(frames, start + clock, first.device)
                )
            )
            update, observation = model.observe_encoded(
                subset_belief(belief, read_ids),
                features.expand(len(reads), -1, -1, -1),
                elapsed.index_select(0, read_ids),
            )
            belief = replace_belief(belief, read_ids, update)
            visibility = visibility.index_copy(
                0, read_ids, observation["visibility_logit"]
            )
            r_tensor = r_tensor.index_copy(0, read_ids, observation["effective_r"])
            acquired[reads, start + clock] = True
            last_read[reads] = clock
            used[reads] += 1
        if clock:
            missing = np.setdiff1d(np.arange(count), reads, assume_unique=True)
            if len(missing):
                missing_ids = torch.as_tensor(missing, device=first.device)
                visibility = visibility.index_copy(
                    0,
                    missing_ids,
                    model.visibility(
                        subset_belief(belief, missing_ids), elapsed[missing_ids]
                    ),
                )
        q = model.process_variance(belief)
        noise_rows.append(q)
        r_rows.append(r_tensor)
        if len(reads):
            policy_begin = time.perf_counter()
            if static_masks is None:
                read_ids = torch.as_tensor(reads, device=first.device)
                packed = (
                    torch.cat(
                        (
                            belief.covariance.index_select(0, read_ids).flatten(1),
                            q.index_select(0, read_ids),
                            r_tensor.index_select(0, read_ids),
                        ),
                        -1,
                    )
                    .cpu()
                    .numpy()
                    .astype(float)
                )
            for local, index in enumerate(reads):
                if static_masks is not None:
                    future = np.flatnonzero(static_masks[index, start + clock + 1 :])
                    next_time = clock + 1 + int(future[0]) if len(future) else None
                    scores = {}
                else:
                    values = packed[local]
                    if clock:
                        last_r[index] = values[18:20]
                    next_time, scores = policies[index].next_time(
                        clock,
                        int(used[index]),
                        values[:16].reshape(4, 4),
                        values[16:18],
                        last_r[index],
                    )
                next_read[index] = -1 if next_time is None else next_time
                decisions[index].append(
                    dict(
                        time=start + clock,
                        next_time=None if next_time is None else start + next_time,
                        scores={
                            str(start + key): value for key, value in scores.items()
                        },
                    )
                )
            policy_seconds += time.perf_counter() - policy_begin
        means.append(belief.mean[:, :2])
        covariances.append(belief.covariance[:, :2, :2])
        logits.append(visibility)
    if any(int(value) != policy.budget for value, policy in zip(used, policies)):
        raise AssertionError("Acquisition budget violated.")
    arrays = [
        torch.stack(values).cpu().numpy()
        for values in (means, covariances, logits, noise_rows, r_rows)
    ]
    if any(not np.isfinite(value).all() for value in arrays):
        raise FloatingPointError("Nonfinite shared rollout; inspect covariance pivots.")
    wall_seconds = time.perf_counter() - begin
    outputs = []
    for index, query in enumerate(queries):
        points = np.broadcast_to(query[[2, 1]], (total, 2)).copy().astype(np.float32)
        covariance = np.zeros((total, 2, 2), np.float32)
        visibility = np.full(total, -20, np.float32)
        points[start:], covariance[start:], visibility[start:] = (
            arrays[0][:, index],
            arrays[1][:, index],
            arrays[2][:, index],
        )
        times = np.flatnonzero(acquired[index, start:])
        r_reads = arrays[4][times, index].astype(float)
        r_reads[0] = initial_r
        outputs.append(
            dict(
                points=points,
                covariance=covariance,
                visibility_logits=visibility,
                acquired=acquired[index],
                observations=int(used[index]),
                q=arrays[3][-1, index],
                last_r=r_reads[-1].copy(),
                q_reads=arrays[3][times, index],
                r_reads=r_reads,
                decisions=decisions[index],
                policy_seconds=policy_seconds / count,
                wall_seconds=wall_seconds / count,
                runtime_accounting="shared encoder; group/setup wall time amortized over queries",
            )
        )
    return outputs
