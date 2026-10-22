"""Alternate heatmap readouts used for diagnosis without changing the trained model."""

from __future__ import annotations

import torch


def heatmap_readouts(logits: torch.Tensor, radius: int = 2) -> dict[str, torch.Tensor]:
    """Compare global expectation, peak cell, and expectation near the strongest peak."""
    if logits.ndim != 3 or radius < 0 or not torch.isfinite(logits).all():
        raise ValueError("Need finite [B,H,W] logits and nonnegative local radius.")
    batch, height, width = logits.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=logits.device),
        torch.arange(width, device=logits.device),
        indexing="ij",
    )
    positions = (
        torch.stack(((xx + 0.5) / width, (yy + 0.5) / height), -1)
        .reshape(-1, 2)
        .to(logits.dtype)
    )
    flat = logits.flatten(1)
    probabilities = flat.softmax(-1)
    peak = flat.argmax(-1)
    px, py = peak % width, peak // width
    neighborhood = (xx.flatten()[None] - px[:, None]).abs() <= radius
    neighborhood &= (yy.flatten()[None] - py[:, None]).abs() <= radius
    local = flat.masked_fill(~neighborhood, -torch.inf).softmax(-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-10).log()).sum(-1)
    return dict(
        global_mean=probabilities @ positions,
        peak=positions[peak],
        local_mean=local @ positions,
        entropy=entropy,
        peak_probability=probabilities.max(-1).values,
    )
