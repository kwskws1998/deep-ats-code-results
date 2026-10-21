"""Independent TAP-Vid first-query metrics in the official 256-pixel convention."""

from __future__ import annotations

import numpy as np


def tapvid_metrics(
    query_points: np.ndarray,
    target_points: np.ndarray,
    target_occluded: np.ndarray,
    predicted_points: np.ndarray,
    predicted_occluded: np.ndarray,
) -> dict[str, float]:
    """Aggregate counts over queries/time within one video, excluding query/past frames.

    Coordinates use fractions of image dimensions, so multiply by 256 before
    applying strict distance thresholds [1,2,4,8,16]. Average resulting metrics
    equally over videos, not over individual tracks or windows.
    """
    n, length = target_occluded.shape
    if (
        query_points.shape != (n, 3)
        or target_points.shape != (n, length, 2)
        or predicted_points.shape != target_points.shape
        or predicted_occluded.shape != (n, length)
    ):
        raise ValueError("Incompatible TAP metric dimensions.")
    if target_occluded.dtype != bool or predicted_occluded.dtype != bool:
        raise ValueError("Occlusion masks must be boolean.")
    if not np.isfinite(predicted_points).all() or not np.isfinite(target_points).all():
        raise ValueError("Point predictions and targets must be finite.")
    clock = query_points[:, 0]
    if (
        not np.isfinite(query_points).all()
        or np.any(clock != np.round(clock))
        or np.any((clock < 0) | (clock >= length))
    ):
        raise ValueError("Invalid query clock.")
    evaluated = np.arange(length)[None, :] > clock[:, None]
    visible = ~target_occluded
    predicted_visible = ~predicted_occluded
    count = int(evaluated.sum())
    positives = int((visible & evaluated).sum())
    if not count or not positives:
        raise ValueError(
            "Video has no evaluated visible point; official ratios are undefined."
        )
    result = dict(
        occlusion_accuracy=float(
            ((predicted_occluded == target_occluded) & evaluated).sum() / count
        )
    )
    squared_error = np.sum(
        ((predicted_points.astype(float) - target_points) * 256) ** 2, -1
    )
    jaccards, within = [], []
    for threshold in (1, 2, 4, 8, 16):
        close = squared_error < threshold**2
        correct = close & visible
        fraction = float((correct & evaluated).sum() / positives)
        tp = int((correct & predicted_visible & evaluated).sum())
        fp = int(((~visible | ~close) & predicted_visible & evaluated).sum())
        jaccard = float(tp / (positives + fp))
        result[f"pts_within_{threshold}"] = fraction
        result[f"jaccard_{threshold}"] = jaccard
        within.append(fraction)
        jaccards.append(jaccard)
    result["average_jaccard"] = float(np.mean(jaccards))
    result["average_pts_within_thresh"] = float(np.mean(within))
    result["evaluated_frames"] = count
    result["visible_evaluated_frames"] = positives
    result["visible_position_mse"] = float(
        squared_error[visible & evaluated].mean() / 2
    )
    return result
