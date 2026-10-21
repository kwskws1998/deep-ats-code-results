"""Query-conditioned spatial decoder and learned causal Gaussian point dynamics.

No clean images, velocities, or simulator parameters are required. Q and R are
effective predictive errors in anchored image coordinates, not identified
physical noise sources. The decoder predicts a point heatmap and visibility.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as functional


@dataclass
class PointBelief:
    """Independent state for one query: position/velocity, covariance, and memory."""

    mean: Tensor
    covariance: Tensor
    context: Tensor
    descriptor: Tensor


class TAPATSModel(nn.Module):
    """Small trainable point tracker; all queries are processed independently."""

    def __init__(self, feature_dim: int = 32, hidden_size: int = 48):
        super().__init__()
        if feature_dim < 8 or feature_dim % 4 or hidden_size < 8:
            raise ValueError("Feature and context dimensions must be >=8.")
        self.feature_dim, self.hidden_size = feature_dim, hidden_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, feature_dim, 5, stride=2, padding=2),
            nn.GroupNorm(4, feature_dim),
            nn.SiLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, stride=2, padding=1),
            nn.GroupNorm(4, feature_dim),
            nn.SiLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(2 * feature_dim + 1, feature_dim, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(feature_dim, 1, 1),
        )
        self.log_temperature = nn.Parameter(torch.tensor(-2.0))
        self.measurement_noise_head = nn.Sequential(
            nn.Linear(6, 32), nn.SiLU(), nn.Linear(32, 2)
        )
        self.observed_visibility_head = nn.Sequential(
            nn.Linear(6, 32), nn.SiLU(), nn.Linear(32, 1)
        )
        self.context_update = nn.GRUCell(8, hidden_size)
        self.acceleration_head = nn.Linear(hidden_size, 2)
        self.process_noise_head = nn.Linear(hidden_size, 2)
        self.forecast_visibility_head = nn.Sequential(
            nn.Linear(hidden_size + 6, 32), nn.SiLU(), nn.Linear(32, 1)
        )
        identity = torch.eye(2)
        transition = torch.eye(4)
        transition[:2, 2:] = identity
        self.register_buffer("transition", transition)
        self.register_buffer("acceleration_map", torch.cat((0.5 * identity, identity)))
        self.register_buffer(
            "observation_map", torch.cat((identity, torch.zeros(2, 2)), 1)
        )
        self.register_buffer("identity", torch.eye(4))
        nn.init.normal_(self.acceleration_head.weight, std=0.001)
        nn.init.zeros_(self.acceleration_head.bias)
        nn.init.normal_(self.process_noise_head.weight, std=0.01)
        nn.init.zeros_(self.process_noise_head.bias)

    def frame_features(self, frames: Tensor) -> Tensor:
        """Encode acquired float RGB frames in [0,1] without temporal batch mixing."""
        return self.encoder(2 * frames - 1)

    @staticmethod
    def sample_descriptor(features: Tensor, query_xy: Tensor) -> Tensor:
        """Match border-padded bilinear sampling with deterministic gather operations.

        This avoids the nondeterministic CUDA grid_sample backward kernel while
        preserving align_corners=False image coordinates.
        """
        batch, channels, height, width = features.shape
        x = (query_xy[:, 0] * width - 0.5).clamp(0, width - 1)
        y = (query_xy[:, 1] * height - 0.5).clamp(0, height - 1)
        x0, y0 = x.floor().long(), y.floor().long()
        x1, y1 = (x0 + 1).clamp_max(width - 1), (y0 + 1).clamp_max(height - 1)
        flat = features.flatten(2)

        def gather(column, row):
            """Gather one independent feature vector per batch element."""
            index = (
                (row * width + column).reshape(batch, 1, 1).expand(batch, channels, 1)
            )
            return flat.gather(2, index).squeeze(-1)

        dx, dy = (x - x0).unsqueeze(-1), (y - y0).unsqueeze(-1)
        top = gather(x0, y0) * (1 - dx) + gather(x1, y0) * dx
        bottom = gather(x0, y1) * (1 - dx) + gather(x1, y1) * dx
        return top * (1 - dy) + bottom * dy

    def initialize(self, frame: Tensor, query_xy: Tensor) -> PointBelief:
        """Read the public query image/position and initialize zero velocity."""
        features = self.frame_features(frame)
        descriptor = self.sample_descriptor(features, query_xy)
        descriptor = functional.normalize(descriptor, dim=-1)
        mean = torch.cat((query_xy, torch.zeros_like(query_xy)), -1)
        variance = frame.new_tensor([1e-6, 1e-6, 1e-3, 1e-3])
        covariance = torch.diag(variance).expand(len(frame), -1, -1).clone()
        context = frame.new_zeros(len(frame), self.hidden_size)
        return PointBelief(mean, covariance, context, descriptor)

    def process_variance(self, belief: PointBelief) -> Tensor:
        """Predict per-frame acceleration uncertainty from causal context only."""
        return (-16 + 11 * self.process_noise_head(belief.context).sigmoid()).exp()

    def predict(self, belief: PointBelief) -> PointBelief:
        """Advance one physical frame using learned mean acceleration and Q."""
        acceleration = 0.03 * self.acceleration_head(belief.context).tanh()
        mean = belief.mean @ self.transition.T + acceleration @ self.acceleration_map.T
        q = self.process_variance(belief)
        process = self.acceleration_map @ torch.diag_embed(q) @ self.acceleration_map.T
        covariance = self.transition @ belief.covariance @ self.transition.T + process
        covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
        return PointBelief(mean, covariance, belief.context, belief.descriptor)

    def visibility(self, belief: PointBelief, elapsed: Tensor) -> Tensor:
        """Forecast visibility even when no new frame has been acquired."""
        inputs = torch.cat(
            (
                belief.context,
                belief.mean,
                belief.covariance[:, :2, :2]
                .diagonal(dim1=-2, dim2=-1)
                .clamp_min(1e-8)
                .log(),
            ),
            -1,
        )
        return (
            self.forecast_visibility_head(inputs).squeeze(-1) - 0.05 * elapsed.log1p()
        )

    def observe(
        self, belief: PointBelief, frames: Tensor, elapsed: Tensor
    ) -> tuple[PointBelief, dict[str, Tensor]]:
        """Decode a query-conditioned heatmap, measurement R, and soft visibility."""
        features = functional.normalize(self.frame_features(frames), dim=1)
        descriptor = belief.descriptor[:, :, None, None].expand_as(features)
        cosine = (features * descriptor).sum(1, keepdim=True)
        temperature = self.log_temperature.exp().clamp(0.03, 1.0)
        logits = (
            cosine[:, 0] / temperature
            + self.decoder(torch.cat((features, descriptor, cosine), 1))[:, 0]
        )
        height, width = logits.shape[-2:]
        yy, xx = torch.meshgrid(
            (torch.arange(height, device=frames.device, dtype=frames.dtype) + 0.5)
            / height,
            (torch.arange(width, device=frames.device, dtype=frames.dtype) + 0.5)
            / width,
            indexing="ij",
        )
        positions = torch.stack((xx, yy), -1).reshape(-1, 2)
        probabilities = logits.flatten(1).softmax(-1)
        measurement = probabilities @ positions
        spread = (
            probabilities[:, :, None]
            * (positions[None] - measurement[:, None]).square()
        ).sum(1)
        entropy = -(probabilities * probabilities.clamp_min(1e-10).log()).sum(-1) / max(
            1.0, math.log(height * width)
        )
        statistics = torch.cat(
            (
                spread,
                measurement - belief.mean[:, :2],
                cosine.flatten(1).max(-1).values[:, None],
                entropy[:, None],
            ),
            -1,
        )
        r = (
            spread * (0.1 + 3.9 * self.measurement_noise_head(statistics).sigmoid())
            + 1e-6
        )
        visibility_logit = self.observed_visibility_head(statistics).squeeze(-1)
        effective_r = r / visibility_logit.sigmoid().clamp_min(0.02)[:, None]
        cross = belief.covariance @ self.observation_map.T
        innovation_covariance = self.observation_map @ cross + torch.diag_embed(
            effective_r
        )
        gain = torch.linalg.solve(
            innovation_covariance, cross.transpose(-1, -2)
        ).transpose(-1, -2)
        innovation = measurement - belief.mean[:, :2]
        mean = belief.mean + (gain @ innovation.unsqueeze(-1)).squeeze(-1)
        residual = self.identity - gain @ self.observation_map
        covariance = residual @ belief.covariance @ residual.transpose(
            -1, -2
        ) + gain @ torch.diag_embed(effective_r) @ gain.transpose(-1, -2)
        covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
        inputs = torch.cat(
            (
                measurement,
                innovation,
                r.log(),
                visibility_logit.sigmoid()[:, None],
                elapsed.log1p()[:, None],
            ),
            -1,
        )
        context = self.context_update(inputs, belief.context)
        result = PointBelief(mean, covariance, context, belief.descriptor)
        return result, dict(
            measurement=measurement,
            r=r,
            effective_r=effective_r,
            visibility_logit=visibility_logit,
            logits=logits,
        )
