"""Prospective fixed-budget schedulers that never receive an unread frame.

The ATS extension minimizes forecast integrated position variance. It is a
task-risk objective, not the scalar package's information gain about Q/R grid
parameters. Forecast Q and R stay fixed within each planning call; replanning
occurs only after a real observation. Uniform suffixes approximate continuation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def uniform_times(length: int, budget: int) -> np.ndarray:
    """Spread exactly budget acquisitions over length frames, always including 0."""
    if (
        any(
            not isinstance(value, (int, np.integer)) or isinstance(value, bool)
            for value in (length, budget)
        )
        or not 1 <= budget <= length
    ):
        raise ValueError("Require 1 <= budget <= length.")
    return np.arange(budget, dtype=np.int64) * length // budget


def covariance_step(
    covariance: np.ndarray, transition: np.ndarray, g: np.ndarray, q: np.ndarray
) -> np.ndarray:
    """Propagate one frame under the same physical law used by the neural filter."""
    return transition @ covariance @ transition.T + (g * q) @ g.T


def covariance_update(covariance: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Expected Joseph update for a direct 2D coordinate observation."""
    gain = np.linalg.solve(covariance[:2, :2] + np.diag(r), covariance[:2, :]).T
    residual = np.eye(4)
    residual[:, :2] -= gain
    result = residual @ covariance @ residual.T + (gain * r) @ gain.T
    return 0.5 * (result + result.T)


@dataclass
class SamplingPolicy:
    """Causal next-observation scheduler with an exact per-clip acquisition budget."""

    name: str
    length: int
    budget: int
    transition: np.ndarray
    acceleration_map: np.ndarray
    seed: int = 1
    max_gap: int = 8
    threshold: float = 0.01
    fixed_q: np.ndarray | None = None
    fixed_r: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate public timing metadata and initialize data-independent controls."""
        if self.name not in {
            "uniform",
            "random",
            "total_uncertainty",
            "ats",
            "ats_fixed_q",
            "ats_fixed_r",
        }:
            raise ValueError(f"Unknown policy: {self.name}")
        uniform_times(self.length, self.budget)
        if (
            not isinstance(self.max_gap, int)
            or isinstance(self.max_gap, bool)
            or self.max_gap < 1
            or not np.isfinite(self.threshold)
            or self.threshold <= 0
        ):
            raise ValueError("max_gap and threshold must be positive.")
        if self.name == "ats_fixed_q" and self.fixed_q is None:
            raise ValueError("ats_fixed_q requires training-only fixed_q.")
        if self.name == "ats_fixed_r" and self.fixed_r is None:
            raise ValueError("ats_fixed_r requires training-only fixed_r.")
        if self.transition.shape != (4, 4) or self.acceleration_map.shape != (4, 2):
            raise ValueError("Invalid transition or acceleration map shape.")
        if (
            not np.isfinite(self.transition).all()
            or not np.isfinite(self.acceleration_map).all()
        ):
            raise ValueError("Motion matrices must be finite.")
        for value in (self.fixed_q, self.fixed_r):
            if value is not None and (
                value.shape != (2,)
                or not np.isfinite(value).all()
                or np.any(value <= 0)
            ):
                raise ValueError(
                    "Fixed noise forecasts must contain two positive finite values."
                )
        self._times = uniform_times(self.length, self.budget)
        if self.name == "random":
            rng = np.random.default_rng(self.seed)
            self._times = np.concatenate(
                (
                    [0],
                    np.sort(
                        rng.choice(
                            np.arange(1, self.length), self.budget - 1, replace=False
                        )
                    ),
                )
            )

    def next_time(
        self,
        time: int,
        used: int,
        covariance: np.ndarray,
        q: np.ndarray,
        r: np.ndarray,
    ) -> tuple[int | None, dict[int, float]]:
        """Choose using current belief/noise forecasts and the remaining budget only."""
        if not 0 <= time < self.length or not 1 <= used <= self.budget:
            raise ValueError("Invalid clock or number of acquisitions.")
        remaining = self.budget - used
        if remaining == 0:
            return None, {}
        if self.length - time - 1 < remaining:
            raise ValueError("The acquisition budget is infeasible at this clock.")
        if self.name in {"uniform", "random"}:
            return int(self._times[used]), {}
        for values, shape in ((covariance, (4, 4)), (q, (2,)), (r, (2,))):
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError("Invalid covariance or noise forecast.")
        if np.any(q <= 0) or np.any(r <= 0):
            raise ValueError("Noise forecasts must be positive.")
        if (
            not np.allclose(covariance, covariance.T, atol=1e-6)
            or np.linalg.eigvalsh(covariance).min() < -1e-8
        ):
            raise ValueError(
                "State covariance must be symmetric positive semidefinite."
            )
        last = min(time + self.max_gap, self.length - remaining)
        candidates = range(time + 1, last + 1)
        if self.name == "total_uncertainty":
            predicted = covariance.copy()
            scores = {}
            for candidate in candidates:
                predicted = covariance_step(
                    predicted, self.transition, self.acceleration_map, q
                )
                scores[candidate] = float(np.trace(predicted[:2, :2]) + r.sum())
                if scores[candidate] >= self.threshold:
                    return candidate, scores
            return last, scores
        q = self.fixed_q if self.name == "ats_fixed_q" else q
        r = self.fixed_r if self.name == "ats_fixed_r" else r
        scores = {}
        for candidate in candidates:
            suffix = uniform_times(self.length - candidate, remaining)
            suffix = np.minimum(suffix, np.arange(remaining) * self.max_gap)
            schedule = set((candidate + suffix).tolist())
            predicted = covariance.copy()
            risk = 0.0
            for future in range(time + 1, self.length):
                predicted = covariance_step(
                    predicted, self.transition, self.acceleration_map, q
                )
                if future in schedule:
                    predicted = covariance_update(predicted, r)
                risk += float(np.trace(predicted[:2, :2]))
            scores[candidate] = risk
        return min(scores, key=lambda candidate: (scores[candidate], candidate)), scores
