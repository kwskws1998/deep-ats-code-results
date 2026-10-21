"""Independent moving-blob clips and a strict labeled NPZ interchange format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np


@dataclass
class VideoData:
    """Frames are N,T,C,H,W; states are normalized [x,y,vx,vy]."""

    frames: np.ndarray
    clean_frames: np.ndarray
    states: np.ndarray
    conditions: np.ndarray
    clip_ids: np.ndarray
    process_variances: np.ndarray | None = None

    def validate(self) -> None:
        """Reject malformed data rather than guessing axes or silently clipping labels."""
        if self.frames.ndim != 5 or min(self.frames.shape) < 1:
            raise ValueError("frames must have nonempty N,T,C,H,W dimensions.")
        n, length, channels, height, width = self.frames.shape
        if length < 3 or height != width or height < 8 or channels < 1:
            raise ValueError("Require T >= 3 and square frames of size >= 8.")
        if self.clean_frames.shape != self.frames.shape or self.states.shape != (
            n,
            length,
            4,
        ):
            raise ValueError("clean_frames or states have incompatible dimensions.")
        for array in (self.frames, self.clean_frames, self.states):
            if array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError("Frames and states must be finite float32 arrays.")
        if any(
            array.min() < 0 or array.max() > 1
            for array in (self.frames, self.clean_frames)
        ):
            raise ValueError("Images must already be normalized to [0,1].")
        if self.conditions.shape != (n,) or self.clip_ids.shape != (n,):
            raise ValueError("conditions and clip_ids must have shape (N,).")
        if (
            self.conditions.dtype.kind not in "US"
            or self.clip_ids.dtype.kind not in "US"
        ):
            raise ValueError(
                "conditions and clip_ids must use non-object string arrays."
            )
        if len(set(self.clip_ids.tolist())) != n:
            raise ValueError("clip_ids must be unique within each split.")
        if self.process_variances is not None and (
            self.process_variances.shape != (n, 2)
            or not np.isfinite(self.process_variances).all()
            or (self.process_variances <= 0).any()
        ):
            raise ValueError("process_variances must be positive finite N,2 values.")

    def save(self, path: Path) -> None:
        """Save portable arrays without Python objects or executable pickle data."""
        self.validate()
        values = {key: value for key, value in vars(self).items() if value is not None}
        np.savez_compressed(path, **values)


def load_data(path: Path) -> VideoData:
    """Load one labeled split; velocities and clean images are explicit targets."""
    with np.load(path, allow_pickle=False) as contents:
        data = VideoData(**{key: contents[key] for key in contents.files})
    data.validate()
    return data


def validate_splits(splits: dict[str, VideoData]) -> None:
    """Prevent cross-split clip reuse and require compatible sequence shapes."""
    identifiers: set[str] = set()
    fingerprints: set[str] = set()
    shapes = set()
    for data in splits.values():
        data.validate()
        shapes.add(data.frames.shape[1:])
        for index, identifier in enumerate(data.clip_ids.tolist()):
            digest = hashlib.sha256(data.frames[index].tobytes()).hexdigest()
            if identifier in identifiers or digest in fingerprints:
                raise ValueError(
                    "Repeated clip identity or exact frame sequence across splits."
                )
            identifiers.add(identifier)
            fingerprints.add(digest)
    if len(shapes) != 1:
        raise ValueError("All splits must share T,C,H,W.")


def moving_blobs(
    count: int,
    length: int = 24,
    image_size: int = 24,
    seed: int = 1,
    prefix: str = "train",
    damping: float = 0.9,
    spring: float = 0.02,
) -> VideoData:
    """Cross low/high acceleration variance with independent pixel noise.

    All methods receive the same pre-generated frames. No latent state, regime
    label, or generator variance is exposed to a sampling policy.
    """
    if count < 1 or length < 3 or image_size < 8 or seed < 0:
        raise ValueError("Invalid synthetic data dimensions or seed.")
    frames = np.empty((count, length, 1, image_size, image_size), np.float32)
    clean = np.empty_like(frames)
    states = np.empty((count, length, 4), np.float32)
    q_values = np.empty((count, 2), np.float32)
    conditions = []
    coordinates = np.linspace(-1, 1, image_size)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    for clip in range(count):
        rng = np.random.default_rng(np.random.SeedSequence([seed, clip]))
        motion = (clip // 2) % 2
        noise = clip % 2
        q = (2e-6, 8e-5)[motion]
        pixel_sd = (0.025, 0.2)[noise]
        q_values[clip] = q
        conditions.append(f"q{motion}_pixels{noise}")
        state = np.concatenate((rng.uniform(-0.4, 0.4, 2), rng.normal(0, 0.02, 2)))
        for time in range(length):
            if time:
                acceleration = (
                    -spring * state[:2]
                    + (damping - 1) * state[2:]
                    + rng.normal(0, np.sqrt(q), 2)
                )
                state = np.concatenate(
                    (
                        state[:2] + state[2:] + 0.5 * acceleration,
                        state[2:] + acceleration,
                    )
                )
            states[clip, time] = state
            image = 0.03 + 0.9 * np.exp(
                -((xx - state[0]) ** 2 + (yy - state[1]) ** 2) / (2 * 0.11**2)
            )
            clean[clip, time, 0] = image
            frames[clip, time, 0] = np.clip(
                image + rng.normal(0, pixel_sd, image.shape), 0, 1
            )
    data = VideoData(
        frames,
        clean,
        states,
        np.asarray(conditions),
        np.asarray([f"{prefix}_{seed}_{i}" for i in range(count)]),
        q_values,
    )
    data.validate()
    return data
