"""Differentiable image encoder, Gaussian motion filter, and clean-image decoder.

The state is [x, y, vx, vy] in normalized image coordinates. Q is acceleration
variance per frame. R is the effective error variance of the encoder's coordinate
measurement, not pixel noise or variational posterior variance. Training requires
coordinate supervision; this is not an unsupervised identification claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass
class Belief:
    """Gaussian physical state and a causal context for forecasting process noise."""

    mean: Tensor
    covariance: Tensor
    context: Tensor


def gaussian_nll(target: Tensor, mean: Tensor, covariance: Tensor) -> Tensor:
    """Return joint Gaussian NLL per example with a declared numerical jitter."""
    dimension = mean.shape[-1]
    identity = torch.eye(dimension, device=mean.device, dtype=mean.dtype)
    factor = torch.linalg.cholesky(covariance + 1e-7 * identity)
    error = (target - mean).unsqueeze(-1)
    whitened = torch.linalg.solve_triangular(factor, error, upper=False)
    return 0.5 * (
        dimension * math.log(2 * math.pi)
        + 2 * factor.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        + whitened.square().sum(dim=(-2, -1))
    )


class VideoATSModel(nn.Module):
    """Learn visual measurements and noise forecasts around a fixed motion law."""

    def __init__(
        self,
        image_size: int = 24,
        channels: int = 1,
        hidden_size: int = 32,
        damping: float = 0.9,
        spring: float = 0.02,
    ):
        super().__init__()
        if image_size < 8 or channels < 1 or hidden_size < 4:
            raise ValueError(
                "Require image_size >= 8, channels >= 1, hidden_size >= 4."
            )
        if not 0 < damping <= 1 or not 0 <= spring < 1:
            raise ValueError("Invalid motion coefficients.")
        self.image_size = image_size
        self.channels = channels
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 12, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(12, 24, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(24 * ((image_size + 3) // 4) ** 2, hidden_size),
            nn.SiLU(),
        )
        self.measurement_head = nn.Linear(hidden_size, 2)
        self.measurement_noise_head = nn.Linear(hidden_size, 2)
        self.context_update = nn.GRUCell(7, hidden_size)
        self.process_noise_head = nn.Linear(hidden_size, 2)
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, channels * image_size * image_size),
            nn.Sigmoid(),
        )
        identity = torch.eye(2)
        transition = torch.zeros(4, 4)
        transition[:2, :2] = (1 - 0.5 * spring) * identity
        transition[:2, 2:] = 0.5 * (1 + damping) * identity
        transition[2:, :2] = -spring * identity
        transition[2:, 2:] = damping * identity
        self.register_buffer("transition", transition)
        self.register_buffer("acceleration_map", torch.cat((0.5 * identity, identity)))
        self.register_buffer(
            "observation_map", torch.cat((identity, torch.zeros(2, 2)), 1)
        )
        self.register_buffer("identity", torch.eye(4))
        nn.init.zeros_(self.measurement_noise_head.weight)
        nn.init.zeros_(self.measurement_noise_head.bias)
        nn.init.normal_(self.process_noise_head.weight, std=0.01)
        nn.init.zeros_(self.process_noise_head.bias)

    def initial_belief(self, batch_size: int) -> Belief:
        """Use the same prior for all conditions without reading any frame."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        prototype = self.transition
        mean = prototype.new_zeros(batch_size, 4)
        variance = prototype.new_tensor([0.25, 0.25, 0.01, 0.01])
        covariance = torch.diag(variance).expand(batch_size, -1, -1).clone()
        return Belief(
            mean, covariance, prototype.new_zeros(batch_size, self.hidden_size)
        )

    def encode(self, frames: Tensor) -> tuple[Tensor, Tensor]:
        """Encode only frames actually acquired; R describes coordinate error."""
        if frames.ndim != 4 or tuple(frames.shape[1:]) != (
            self.channels,
            self.image_size,
            self.image_size,
        ):
            raise ValueError("Frame tensor does not match the configured C,H,W.")
        features = self.encoder(frames)
        measurement = 1.5 * self.measurement_head(features).tanh()
        log_variance = -12 + 12 * self.measurement_noise_head(features).sigmoid()
        return measurement, log_variance.exp()

    def process_variance(self, belief: Belief) -> Tensor:
        """Forecast positive per-axis acceleration variance from observed history."""
        log_variance = -16 + 10 * self.process_noise_head(belief.context).sigmoid()
        return log_variance.exp()

    def predict(self, belief: Belief) -> tuple[Belief, Tensor]:
        """Advance exactly one physical frame, including unobserved frames."""
        q = self.process_variance(belief)
        process_covariance = (
            self.acceleration_map
            @ torch.diag_embed(q)
            @ self.acceleration_map.transpose(-1, -2)
        )
        mean = belief.mean @ self.transition.T
        covariance = (
            self.transition @ belief.covariance @ self.transition.T + process_covariance
        )
        covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
        return Belief(mean, covariance, belief.context), q

    def update(
        self,
        belief: Belief,
        measurement: Tensor,
        variance: Tensor,
        elapsed: Tensor,
    ) -> Belief:
        """Apply a Joseph-form update and update context using observed data only."""
        h = self.observation_map
        noise = torch.diag_embed(variance)
        innovation = measurement - belief.mean[:, :2]
        cross = belief.covariance @ h.T
        innovation_covariance = h @ cross + noise
        gain = torch.linalg.solve(
            innovation_covariance, cross.transpose(-1, -2)
        ).transpose(-1, -2)
        mean = belief.mean + (gain @ innovation.unsqueeze(-1)).squeeze(-1)
        residual = self.identity - gain @ h
        covariance = residual @ belief.covariance @ residual.transpose(
            -1, -2
        ) + gain @ noise @ gain.transpose(-1, -2)
        covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
        context_input = torch.cat(
            (measurement, innovation, variance.log(), elapsed.reshape(-1, 1).log1p()),
            -1,
        )
        context = self.context_update(context_input, belief.context)
        return Belief(mean, covariance, context)

    def decode(self, mean: Tensor) -> Tensor:
        """Decode the position-anchored state into a clean image estimate."""
        output = self.decoder(mean[..., :2])
        return output.reshape(
            *mean.shape[:-1], self.channels, self.image_size, self.image_size
        )

    def observed_subset(
        self,
        belief: Belief,
        indices: Tensor,
        frames: Tensor,
        elapsed: Tensor,
    ) -> tuple[Belief, Tensor, Tensor]:
        """Update selected batch elements without encoding any unselected image."""
        measurement, variance = self.encode(frames)
        subset = Belief(
            belief.mean[indices], belief.covariance[indices], belief.context[indices]
        )
        updated = self.update(subset, measurement, variance, elapsed)
        return (
            Belief(
                belief.mean.index_copy(0, indices, updated.mean),
                belief.covariance.index_copy(0, indices, updated.covariance),
                belief.context.index_copy(0, indices, updated.context),
            ),
            measurement,
            variance,
        )
