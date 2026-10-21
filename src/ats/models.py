"""Perceptual model baselines for temporal noise-source inference."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ats.beliefs import GridBelief
from ats.config import ExperimentCondition, SimulationConfig


def normal_nll(observation: float, mean: float, variance: float) -> float:
    if not np.isfinite([observation, mean, variance]).all() or variance <= 0:
        raise FloatingPointError(
            "Cannot score a nonfinite or nonpositive-variance Gaussian prediction."
        )
    error = float(observation) - float(mean)
    if not np.isfinite(error):
        raise FloatingPointError("Nonfinite prediction error.")
    return float(0.5 * (np.log(2.0 * np.pi * variance) + (error**2) / variance))


def safe_log_ratio(volatility: float, stochasticity: float) -> float:
    if not np.isfinite(volatility) or not np.isfinite(stochasticity):
        return float("nan")
    if volatility <= 0.0 or stochasticity <= 0.0:
        return float("nan")
    return float(np.log(volatility / stochasticity))


@dataclass(frozen=True)
class ModelPrediction:
    mean: float
    variance: float


@dataclass(frozen=True)
class ModelEstimate:
    volatility: float
    stochasticity: float
    log_ratio: float
    learning_rate: float


class BasePerceptualModel:
    name: str

    def reset(self, initial_observation: float) -> None:
        raise NotImplementedError

    def predict(self, dt: int) -> ModelPrediction:
        raise NotImplementedError

    def update(self, observation: float, dt: int) -> ModelEstimate:
        raise NotImplementedError

    def estimate(self) -> ModelEstimate:
        raise NotImplementedError

    def negative_log_likelihood(self, observation: float, dt: int) -> float:
        """Score the model's predictive Gaussian before assimilating this observation."""
        prediction = self.predict(dt)
        return normal_nll(observation, prediction.mean, prediction.variance)


class KalmanModel(BasePerceptualModel):
    """Scalar Kalman filter with fixed process and observation noise."""

    def __init__(
        self,
        name: str,
        volatility: float,
        stochasticity: float,
        initial_variance: float,
    ):
        self.name = name
        self.volatility = float(volatility)
        self.stochasticity = float(stochasticity)
        self.initial_variance = float(initial_variance)
        self.mean = 0.0
        self.variance = self.initial_variance
        self.learning_rate = 0.0

    def reset(self, initial_observation: float) -> None:
        self.mean = float(initial_observation)
        self.variance = self.initial_variance
        self.learning_rate = 0.0

    def predict(self, dt: int) -> ModelPrediction:
        prior_variance = self.variance + self.volatility * int(dt)
        return ModelPrediction(
            mean=self.mean,
            variance=prior_variance + self.stochasticity,
        )

    def update(self, observation: float, dt: int) -> ModelEstimate:
        prior_variance = self.variance + self.volatility * int(dt)
        predictive_variance = prior_variance + self.stochasticity
        prediction_error = float(observation) - self.mean
        self.learning_rate = float(prior_variance / predictive_variance)
        self.mean += self.learning_rate * prediction_error
        self.variance = max((1.0 - self.learning_rate) * prior_variance, 1e-12)
        return self.estimate()

    def estimate(self) -> ModelEstimate:
        return ModelEstimate(
            volatility=self.volatility,
            stochasticity=self.stochasticity,
            log_ratio=safe_log_ratio(self.volatility, self.stochasticity),
            learning_rate=self.learning_rate,
        )


class GridPerceptualModel(BasePerceptualModel):
    """Bayesian grid model over one or both noise dimensions."""

    def __init__(
        self,
        name: str,
        volatility_grid: np.ndarray,
        stochasticity_grid: np.ndarray,
        initial_variance: float,
    ):
        self.name = name
        self.belief = GridBelief(
            volatility_grid=volatility_grid,
            stochasticity_grid=stochasticity_grid,
            initial_variance=initial_variance,
        )
        self.last_estimate = ModelEstimate(
            volatility=float("nan"),
            stochasticity=float("nan"),
            log_ratio=float("nan"),
            learning_rate=0.0,
        )

    def reset(self, initial_observation: float) -> None:
        self.belief.reset(initial_observation)
        self.last_estimate = self.estimate()

    def predict(self, dt: int) -> ModelPrediction:
        return ModelPrediction(
            mean=self.belief.predictive_mean(),
            variance=self.belief.mixture_predictive_variance(dt),
        )

    def negative_log_likelihood(self, observation: float, dt: int) -> float:
        """Score the full predictive mixture rather than a moment-matched Gaussian."""
        return -self.belief.predictive_log_density(observation, dt)

    def update(self, observation: float, dt: int) -> ModelEstimate:
        summary = self.belief.update(observation, dt)
        self.last_estimate = ModelEstimate(
            volatility=summary.volatility_mean,
            stochasticity=summary.stochasticity_mean,
            log_ratio=summary.expected_log_ratio,
            learning_rate=summary.learning_rate,
        )
        return self.last_estimate

    def estimate(self) -> ModelEstimate:
        summary = self.belief.summary()
        return ModelEstimate(
            volatility=summary.volatility_mean,
            stochasticity=summary.stochasticity_mean,
            log_ratio=summary.expected_log_ratio,
            learning_rate=summary.learning_rate,
        )


class DeltaRuleModel(BasePerceptualModel):
    """Fixed learning-rate prediction model without explicit noise decomposition."""

    def __init__(
        self,
        name: str = "delta_rule",
        learning_rate: float = 0.25,
        initial_error_variance: float = 9.0,
        variance_rate: float = 0.05,
    ):
        self.name = name
        self.learning_rate = float(learning_rate)
        self.initial_error_variance = float(initial_error_variance)
        self.variance_rate = float(variance_rate)
        self.mean = 0.0
        self.error_variance = self.initial_error_variance

    def reset(self, initial_observation: float) -> None:
        self.mean = float(initial_observation)
        self.error_variance = self.initial_error_variance

    def predict(self, dt: int) -> ModelPrediction:
        return ModelPrediction(
            mean=self.mean,
            variance=max(self.error_variance + 0.05 * int(dt), 1e-9),
        )

    def update(self, observation: float, dt: int) -> ModelEstimate:
        prediction_error = float(observation) - self.mean
        self.mean += self.learning_rate * prediction_error
        self.error_variance = max(
            (1.0 - self.variance_rate) * self.error_variance
            + self.variance_rate * prediction_error**2,
            1e-9,
        )
        return self.estimate()

    def estimate(self) -> ModelEstimate:
        return ModelEstimate(
            volatility=float("nan"),
            stochasticity=float("nan"),
            log_ratio=float("nan"),
            learning_rate=self.learning_rate,
        )


class PearceHallModel(BasePerceptualModel):
    """Associability model with prediction-error-driven learning rate."""

    def __init__(
        self,
        name: str = "pearce_hall",
        initial_learning_rate: float = 0.25,
        associability_rate: float = 0.12,
        initial_error_variance: float = 9.0,
        variance_rate: float = 0.05,
    ):
        self.name = name
        self.initial_learning_rate = float(initial_learning_rate)
        self.associability_rate = float(associability_rate)
        self.initial_error_variance = float(initial_error_variance)
        self.variance_rate = float(variance_rate)
        self.mean = 0.0
        self.learning_rate = self.initial_learning_rate
        self.error_variance = self.initial_error_variance

    def reset(self, initial_observation: float) -> None:
        self.mean = float(initial_observation)
        self.learning_rate = self.initial_learning_rate
        self.error_variance = self.initial_error_variance

    def predict(self, dt: int) -> ModelPrediction:
        return ModelPrediction(
            mean=self.mean,
            variance=max(self.error_variance + 0.05 * int(dt), 1e-9),
        )

    def update(self, observation: float, dt: int) -> ModelEstimate:
        prediction = self.predict(dt)
        prediction_error = float(observation) - prediction.mean
        scaled_error = min(
            1.0, abs(prediction_error) / (np.sqrt(prediction.variance) + 1e-9)
        )
        self.learning_rate = float(
            np.clip(
                (1.0 - self.associability_rate) * self.learning_rate
                + self.associability_rate * scaled_error,
                0.02,
                0.95,
            )
        )
        self.mean += self.learning_rate * prediction_error
        self.error_variance = max(
            (1.0 - self.variance_rate) * self.error_variance
            + self.variance_rate * prediction_error**2,
            1e-9,
        )
        return self.estimate()

    def estimate(self) -> ModelEstimate:
        return ModelEstimate(
            volatility=float("nan"),
            stochasticity=float("nan"),
            log_ratio=float("nan"),
            learning_rate=self.learning_rate,
        )


class HGFStyleVolatilityModel(BasePerceptualModel):
    """Kalman model with surprise-driven log-volatility adaptation."""

    def __init__(
        self,
        name: str = "hgf_style_volatility",
        initial_volatility: float = 1.0,
        fixed_stochasticity: float = 3.0,
        initial_variance: float = 12.0,
        volatility_learning_rate: float = 0.08,
        volatility_bounds: tuple[float, float] = (0.08, 8.0),
    ):
        self.name = name
        self.initial_volatility = float(initial_volatility)
        self.fixed_stochasticity = float(fixed_stochasticity)
        self.initial_variance = float(initial_variance)
        self.volatility_learning_rate = float(volatility_learning_rate)
        self.log_volatility_bounds = (
            float(np.log(volatility_bounds[0])),
            float(np.log(volatility_bounds[1])),
        )
        self.mean = 0.0
        self.variance = self.initial_variance
        self.log_volatility = float(np.log(self.initial_volatility))
        self.learning_rate = 0.0

    def reset(self, initial_observation: float) -> None:
        self.mean = float(initial_observation)
        self.variance = self.initial_variance
        self.log_volatility = float(np.log(self.initial_volatility))
        self.learning_rate = 0.0

    @property
    def volatility(self) -> float:
        return float(np.exp(self.log_volatility))

    def predict(self, dt: int) -> ModelPrediction:
        prior_variance = self.variance + self.volatility * int(dt)
        return ModelPrediction(
            mean=self.mean,
            variance=prior_variance + self.fixed_stochasticity,
        )

    def update(self, observation: float, dt: int) -> ModelEstimate:
        prior_variance = self.variance + self.volatility * int(dt)
        predictive_variance = prior_variance + self.fixed_stochasticity
        prediction_error = float(observation) - self.mean
        self.learning_rate = float(prior_variance / predictive_variance)
        self.mean += self.learning_rate * prediction_error
        self.variance = max((1.0 - self.learning_rate) * prior_variance, 1e-12)

        normalized_surprise = prediction_error**2 / max(predictive_variance, 1e-9) - 1.0
        self.log_volatility = float(
            np.clip(
                self.log_volatility
                + self.volatility_learning_rate * normalized_surprise,
                self.log_volatility_bounds[0],
                self.log_volatility_bounds[1],
            )
        )
        return self.estimate()

    def estimate(self) -> ModelEstimate:
        return ModelEstimate(
            volatility=self.volatility,
            stochasticity=self.fixed_stochasticity,
            log_ratio=safe_log_ratio(self.volatility, self.fixed_stochasticity),
            learning_rate=self.learning_rate,
        )


class PyHGFContinuousModel(BasePerceptualModel):
    """Three-level PyHGF on unit time steps, with missing observations between samples.

    Propagating unit steps preserves the observation equation y=x+noise. Passing
    a long time step directly to generalized HGF also scales its value coupling.
    Invalid internal states raise instead of being replaced by clipped forecasts.
    """

    def __init__(
        self,
        name: str = "pyhgf_full_3level",
        fixed_stochasticity: float = 3.0,
        initial_volatility: float = 1.0,
        initial_variance: float = 12.0,
        update_type: str = "unbounded",
    ):
        if not np.isfinite(
            [fixed_stochasticity, initial_volatility, initial_variance]
        ).all():
            raise ValueError("HGF variances must be finite.")
        if min(fixed_stochasticity, initial_volatility, initial_variance) <= 0:
            raise ValueError("HGF variances must be strictly positive.")
        self.name = name
        self.fixed_stochasticity = float(fixed_stochasticity)
        self.initial_volatility = float(initial_volatility)
        self.initial_variance = float(initial_variance)
        self.update_type = update_type
        self.model = None
        self.learning_rate = 0.0
        self._prediction_cache: dict[int, ModelPrediction] = {}

    def reset(self, initial_observation: float) -> None:
        """Initialize the actual process variance to the requested value."""
        os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib_cache"))
        try:
            from pyhgf.model import HGF
        except ImportError as exc:
            raise RuntimeError("pyhgf is required for PyHGFContinuousModel.") from exc
        self.model = HGF(
            n_levels=3,
            model_type="continuous",
            update_type=self.update_type,
            initial_mean={
                "1": float(initial_observation),
                "2": float(np.log(self.initial_volatility) + 3.0),
                "3": 0.0,
            },
            initial_precision={"1": 1.0 / self.initial_variance, "2": 1.0, "3": 1.0},
            continuous_precision=1.0 / self.fixed_stochasticity,
            tonic_volatility={"1": -3.0, "2": -3.0, "3": -3.0},
            volatility_coupling={"1": 1.0, "2": 1.0},
            tonic_drift={"1": 0.0, "2": 0.0, "3": 0.0},
        )
        self.learning_rate = 0.0
        self._prediction_cache.clear()

    def _state(self) -> dict:
        if self.model is None:
            raise RuntimeError("Model must be reset before use.")
        return (
            self.model.last_attributes
            if self.model.last_attributes is not None
            else self.model.attributes
        )

    def _latent_moments(self, state=None) -> tuple[float, float, float]:
        """Read finite latent moments without truncating the HGF's inferred variance."""
        state = self._state() if state is None else state
        for index in (1, 2, 3):
            mean = float(state[index]["mean"])
            precision = float(state[index]["precision"])
            if not np.isfinite([mean, precision]).all() or precision <= 0:
                raise FloatingPointError(
                    f"{self.name}: invalid HGF state at level {index}."
                )
        omega = float(state[1]["tonic_volatility"])
        kappa = float(np.asarray(state[1]["volatility_coupling_parents"])[0])
        with np.errstate(over="raise", invalid="raise"):
            volatility = float(np.exp(omega + kappa * float(state[2]["mean"])))
        if not np.isfinite(volatility) or volatility <= 0:
            raise FloatingPointError(f"{self.name}: invalid inferred process variance.")
        return float(state[1]["mean"]), 1.0 / float(state[1]["precision"]), volatility

    def _advance(self, state, dt: int, observation: float | None = None):
        """Reuse the jitted unit transition without recompiling a scan for every interval."""
        for index in range(int(dt)):
            observed = observation is not None and index == int(dt) - 1
            value = float(observation) if observed else 0.0
            inputs = (
                (np.array([value], dtype=np.float32),),
                (np.int32(observed),),
                np.float32(1.0),
                None,
            )
            state, _ = self.model.scan_fn(state, inputs)
        return state

    def predict(self, dt: int) -> ModelPrediction:
        """Propagate an observation-free copy, leaving the live posterior unchanged."""
        if int(dt) != dt or dt <= 0:
            raise ValueError("dt must be a positive integer.")
        if self.model is None:
            raise RuntimeError("Model must be reset before use.")
        if int(dt) not in self._prediction_cache:
            predicted = self._advance(self._state(), int(dt))
            mean, variance, _ = self._latent_moments(predicted)
            self._prediction_cache[int(dt)] = ModelPrediction(
                mean, variance + self.fixed_stochasticity
            )
        return self._prediction_cache[int(dt)]

    def update(self, observation: float, dt: int) -> ModelEstimate:
        """Commit dt unit transitions, observing only at the final transition."""
        if not np.isfinite(observation):
            raise ValueError("observation must be finite.")
        prediction = self.predict(dt)
        updated = self._advance(self._state(), int(dt), float(observation))
        self._latent_moments(updated)
        self.model.attributes = updated
        self.model.last_attributes = updated
        self.model.node_trajectories = None
        self._prediction_cache.clear()
        prior_variance = prediction.variance - self.fixed_stochasticity
        self.learning_rate = prior_variance / prediction.variance
        return self.estimate()

    def estimate(self) -> ModelEstimate:
        _, _, volatility = self._latent_moments()
        return ModelEstimate(
            volatility,
            self.fixed_stochasticity,
            safe_log_ratio(volatility, self.fixed_stochasticity),
            self.learning_rate,
        )


def make_model_suite(
    condition: ExperimentCondition,
    config: SimulationConfig,
) -> tuple[BasePerceptualModel, ...]:
    volatility_grid, stochasticity_grid = config.grid.make_grids()
    fixed_volatility = 1.0
    fixed_stochasticity = 3.0
    return (
        KalmanModel(
            name="oracle_kalman",
            volatility=condition.volatility,
            stochasticity=condition.stochasticity,
            initial_variance=config.agent.initial_variance,
        ),
        KalmanModel(
            name="fixed_kalman",
            volatility=fixed_volatility,
            stochasticity=fixed_stochasticity,
            initial_variance=config.agent.initial_variance,
        ),
        GridPerceptualModel(
            name="joint_grid",
            volatility_grid=volatility_grid,
            stochasticity_grid=stochasticity_grid,
            initial_variance=config.agent.initial_variance,
        ),
        GridPerceptualModel(
            name="volatility_only_grid",
            volatility_grid=volatility_grid,
            stochasticity_grid=np.array([fixed_stochasticity]),
            initial_variance=config.agent.initial_variance,
        ),
        GridPerceptualModel(
            name="stochasticity_only_grid",
            volatility_grid=np.array([fixed_volatility]),
            stochasticity_grid=stochasticity_grid,
            initial_variance=config.agent.initial_variance,
        ),
        PyHGFContinuousModel(
            name="pyhgf_full_3level",
            fixed_stochasticity=fixed_stochasticity,
            initial_volatility=fixed_volatility,
            initial_variance=config.agent.initial_variance,
        ),
        PyHGFContinuousModel(
            name="pyhgf_full_3level_oracle_s",
            fixed_stochasticity=condition.stochasticity,
            initial_volatility=fixed_volatility,
            initial_variance=config.agent.initial_variance,
        ),
        HGFStyleVolatilityModel(
            initial_volatility=fixed_volatility,
            fixed_stochasticity=fixed_stochasticity,
            initial_variance=config.agent.initial_variance,
            volatility_bounds=(config.grid.volatility_min, config.grid.volatility_max),
        ),
        PearceHallModel(),
        DeltaRuleModel(),
    )


def default_model_names() -> tuple[str, ...]:
    return (
        "oracle_kalman",
        "joint_grid",
        "fixed_kalman",
        "pyhgf_full_3level",
        "pyhgf_full_3level_oracle_s",
        "hgf_style_volatility",
        "volatility_only_grid",
        "stochasticity_only_grid",
        "pearce_hall",
        "delta_rule",
    )
