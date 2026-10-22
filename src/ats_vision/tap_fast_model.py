"""State-dict-compatible execution backend for shared visual features and 2D algebra.

The reference TAPATSModel sources remain unchanged for checkpoint verification.
This backend has identical parameters, state transitions, and observation-budget
semantics. Floating-point reduction order can differ; it is a separately recorded
execution choice, not bitwise replay of a reference run.
"""

import torch
from torch import Tensor
import torch.nn.functional as functional
import math

from ats_vision.tap_fast_math import solve_spd2
from ats_vision.tap_model import PointBelief, TAPATSModel


class SharedTAPATSModel(TAPATSModel):
    """Reuse a frame's encoder output across independently gated queries."""

    execution_backend = "shared"

    def spatial_positions(self, height, width, prototype):
        """Cache a parameter-free spatial grid by shape, device and dtype."""
        key = (height, width, prototype.device, prototype.dtype)
        if getattr(self, "_position_key", None) != key:
            yy, xx = torch.meshgrid(
                (
                    torch.arange(height, device=prototype.device, dtype=prototype.dtype)
                    + 0.5
                )
                / height,
                (
                    torch.arange(width, device=prototype.device, dtype=prototype.dtype)
                    + 0.5
                )
                / width,
                indexing="ij",
            )
            self._position_values = torch.stack((xx, yy), -1).reshape(-1, 2)
            self._position_key = key
        return self._position_values

    def initialize_encoded(self, features, query_xy):
        """Initialize the reference state from already acquired image features."""
        descriptor = functional.normalize(
            self.sample_descriptor(features, query_xy), dim=-1
        )
        mean = torch.cat((query_xy, torch.zeros_like(query_xy)), -1)
        variance = features.new_tensor([1e-6, 1e-6, 1e-3, 1e-3])
        covariance = torch.diag(variance).expand(len(query_xy), -1, -1).clone()
        context = features.new_zeros(len(query_xy), self.hidden_size)
        return PointBelief(mean, covariance, context, descriptor)

    def observe(self, belief, frames, elapsed):
        """Retain the scalar API while using synchronization-free 2D algebra."""
        return self.observe_encoded(belief, self.frame_features(frames), elapsed)

    def observe_encoded(
        self, belief: PointBelief, features: Tensor, elapsed: Tensor
    ) -> tuple[PointBelief, dict[str, Tensor]]:
        """Decode a query-conditioned heatmap, measurement R, and soft visibility."""
        if len(features) > 1 and features.stride(0) == 0:
            features = functional.normalize(features[:1], dim=1).expand_as(features)
        else:
            features = functional.normalize(features, dim=1)
        descriptor = belief.descriptor[:, :, None, None].expand_as(features)
        cosine = (features * descriptor).sum(1, keepdim=True)
        temperature = self.log_temperature.exp().clamp(0.03, 1.0)
        logits = (
            cosine[:, 0] / temperature
            + self.decoder(torch.cat((features, descriptor, cosine), 1))[:, 0]
        )
        height, width = logits.shape[-2:]
        positions = self.spatial_positions(height, width, features)
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
        gain = solve_spd2(innovation_covariance, cross.transpose(-1, -2)).transpose(
            -1, -2
        )
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
