"""Competitive active-sampling model baselines."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from ats.agents import make_agent
from ats.beliefs import GridBelief
from ats.config import ExperimentCondition, SimulationConfig, default_conditions
from ats.envs import TemporalSamplingEnv
from ats.policies import sampling_cost
from ats.models import (
    BasePerceptualModel,
    DeltaRuleModel,
    KalmanModel,
    ModelEstimate,
    ModelPrediction,
    PearceHallModel,
    PyHGFContinuousModel,
    normal_nll,
)


def stable_seed(base_seed: int, *parts: object) -> int:
    text = "|".join([str(base_seed), *(str(part) for part in parts)])
    value = 2166136261
    for char in text:
        value = (value ^ ord(char)) * 16777619
        value = value % (2**32)
    return int(value)


def finite_mean(values: Iterable[float]) -> float:
    finite_values = [float(value) for value in values if np.isfinite(float(value))]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))


def uncertainty_sampling_score(
    prediction: ModelPrediction,
    dt: int,
    wait_cost: float,
    observation_rate_cost: float,
    cost_mode: str = "per_observation",
) -> float:
    return float(
        np.log(max(prediction.variance, 1e-9))
        - sampling_cost(dt, wait_cost, observation_rate_cost, cost_mode)
    )


@dataclass
class BaseCompetitiveModel:
    name: str

    def reset(self, initial_observation: float) -> None:
        raise NotImplementedError

    def choose_dt(self, step: int, action_space: tuple[int, ...] | None = None) -> int:
        raise NotImplementedError

    def negative_log_likelihood(self, observation: float, dt: int) -> float:
        """Score the model's observation forecast before its posterior update."""
        prediction = self.predict(dt)
        return normal_nll(observation, prediction.mean, prediction.variance)

    def predict(self, dt: int) -> ModelPrediction:
        raise NotImplementedError

    def update(self, observation: float, dt: int) -> ModelEstimate:
        raise NotImplementedError


class ATSJointActiveModel(BaseCompetitiveModel):
    def __init__(self, config: SimulationConfig, seed: int):
        self.name = "ats_joint_active"
        volatility_grid, stochasticity_grid = config.grid.make_grids()
        self.belief = GridBelief(
            volatility_grid=volatility_grid,
            stochasticity_grid=stochasticity_grid,
            initial_variance=config.agent.initial_variance,
        )
        self.agent = make_agent("active_temporal", config.agent, seed)

    def reset(self, initial_observation: float) -> None:
        self.belief.reset(initial_observation)
        self.agent.reset()

    def choose_dt(self, step: int, action_space: tuple[int, ...] | None = None) -> int:
        original = self.agent.config
        try:
            if action_space is not None:
                self.agent.config = replace(original, action_space=action_space)
            return self.agent.choose_dt(self.belief, step)
        finally:
            self.agent.config = original

    def predict(self, dt: int) -> ModelPrediction:
        return ModelPrediction(
            mean=self.belief.predictive_mean(),
            variance=self.belief.mixture_predictive_variance(dt),
        )

    def negative_log_likelihood(self, observation: float, dt: int) -> float:
        return -self.belief.predictive_log_density(observation, dt)

    def update(self, observation: float, dt: int) -> ModelEstimate:
        summary = self.belief.update(observation, dt)
        return ModelEstimate(
            volatility=summary.volatility_mean,
            stochasticity=summary.stochasticity_mean,
            log_ratio=summary.expected_log_ratio,
            learning_rate=summary.learning_rate,
        )


class FixedScheduleJointModel(ATSJointActiveModel):
    def __init__(self, name: str, config: SimulationConfig, seed: int, mode: str):
        super().__init__(config=config, seed=seed)
        self.name = name
        self.action_space = config.agent.action_space
        self.mode = mode

    def choose_dt(self, step: int, action_space: tuple[int, ...] | None = None) -> int:
        actions = self.action_space if action_space is None else action_space
        if self.mode == "short":
            return min(actions)
        if self.mode == "long":
            return max(actions)
        if self.mode.startswith("dt_"):
            requested = int(self.mode.removeprefix("dt_"))
            return max((dt for dt in actions if dt <= requested), default=min(actions))
        if self.mode == "passive_grid":
            return int(actions[step % len(actions)])
        raise ValueError(f"Unknown fixed schedule mode: {self.mode}")


class UncertaintyActiveModel(BaseCompetitiveModel):
    def __init__(
        self,
        name: str,
        model: BasePerceptualModel,
        action_space: tuple[int, ...],
        wait_cost: float,
        observation_rate_cost: float,
        cost_mode: str = "per_observation",
    ):
        self.name = name
        self.model = model
        self.action_space = action_space
        self.wait_cost = wait_cost
        self.observation_rate_cost = observation_rate_cost
        self.cost_mode = cost_mode

    def reset(self, initial_observation: float) -> None:
        self.model.reset(initial_observation)

    def choose_dt(self, step: int, action_space: tuple[int, ...] | None = None) -> int:
        scores = {
            int(dt): uncertainty_sampling_score(
                self.model.predict(int(dt)),
                int(dt),
                self.wait_cost,
                self.observation_rate_cost,
                self.cost_mode,
            )
            for dt in (self.action_space if action_space is None else action_space)
        }
        return max(scores, key=lambda key: (scores[key], -key))

    def predict(self, dt: int) -> ModelPrediction:
        return self.model.predict(dt)

    def negative_log_likelihood(self, observation: float, dt: int) -> float:
        return self.model.negative_log_likelihood(observation, dt)

    def update(self, observation: float, dt: int) -> ModelEstimate:
        return self.model.update(observation, dt)


def make_competitive_models(
    condition: ExperimentCondition,
    config: SimulationConfig,
    seed: int,
) -> tuple[BaseCompetitiveModel, ...]:
    return tuple(
        make_competitive_model(name, condition, config, stable_seed(seed, name))
        for name in competitive_model_names()
    )


def make_competitive_model(
    name: str,
    condition: ExperimentCondition,
    config: SimulationConfig,
    seed: int,
) -> BaseCompetitiveModel:
    fixed_volatility = 1.0
    fixed_stochasticity = 3.0
    if name == "ats_joint_active":
        return ATSJointActiveModel(config=config, seed=stable_seed(seed, "ats"))
    if name == "joint_fixed_short":
        return FixedScheduleJointModel(
            name="joint_fixed_short",
            config=config,
            seed=stable_seed(seed, "short"),
            mode="short",
        )
    if name == "joint_passive_grid":
        return FixedScheduleJointModel(
            name="joint_passive_grid",
            config=config,
            seed=stable_seed(seed, "passive"),
            mode="passive_grid",
        )
    if name in {"joint_fixed_dt_2", "joint_fixed_dt_4", "joint_fixed_long"}:
        mode = "long" if name == "joint_fixed_long" else "dt_" + name.rsplit("_", 1)[-1]
        return FixedScheduleJointModel(name, config, seed, mode)
    if name == "kalman_fixed_active":
        return UncertaintyActiveModel(
            name="kalman_fixed_active",
            model=KalmanModel(
                name="kalman_fixed_active",
                volatility=fixed_volatility,
                stochasticity=fixed_stochasticity,
                initial_variance=config.agent.initial_variance,
            ),
            action_space=config.agent.action_space,
            wait_cost=config.agent.wait_cost,
            observation_rate_cost=config.agent.observation_rate_cost,
            cost_mode=config.agent.cost_mode,
        )
    if name == "kalman_oracle_active":
        return UncertaintyActiveModel(
            name="kalman_oracle_active",
            model=KalmanModel(
                name="kalman_oracle_active",
                volatility=condition.volatility,
                stochasticity=condition.stochasticity,
                initial_variance=config.agent.initial_variance,
            ),
            action_space=config.agent.action_space,
            wait_cost=config.agent.wait_cost,
            observation_rate_cost=config.agent.observation_rate_cost,
            cost_mode=config.agent.cost_mode,
        )
    if name == "pyhgf_full_active":
        return UncertaintyActiveModel(
            name="pyhgf_full_active",
            model=PyHGFContinuousModel(
                name="pyhgf_full_active",
                fixed_stochasticity=fixed_stochasticity,
                initial_volatility=fixed_volatility,
                initial_variance=config.agent.initial_variance,
            ),
            action_space=config.agent.action_space,
            wait_cost=config.agent.wait_cost,
            observation_rate_cost=config.agent.observation_rate_cost,
            cost_mode=config.agent.cost_mode,
        )
    if name == "pyhgf_full_oracle_s_active":
        return UncertaintyActiveModel(
            name="pyhgf_full_oracle_s_active",
            model=PyHGFContinuousModel(
                name="pyhgf_full_oracle_s_active",
                fixed_stochasticity=condition.stochasticity,
                initial_volatility=fixed_volatility,
                initial_variance=config.agent.initial_variance,
            ),
            action_space=config.agent.action_space,
            wait_cost=config.agent.wait_cost,
            observation_rate_cost=config.agent.observation_rate_cost,
            cost_mode=config.agent.cost_mode,
        )
    if name == "pearce_hall_active":
        return UncertaintyActiveModel(
            name="pearce_hall_active",
            model=PearceHallModel(name="pearce_hall_active"),
            action_space=config.agent.action_space,
            wait_cost=config.agent.wait_cost,
            observation_rate_cost=config.agent.observation_rate_cost,
            cost_mode=config.agent.cost_mode,
        )
    if name == "delta_rule_active":
        return UncertaintyActiveModel(
            name="delta_rule_active",
            model=DeltaRuleModel(name="delta_rule_active"),
            action_space=config.agent.action_space,
            wait_cost=config.agent.wait_cost,
            observation_rate_cost=config.agent.observation_rate_cost,
            cost_mode=config.agent.cost_mode,
        )
    raise ValueError(f"Unknown competitive model name: {name}")


def run_competitive_model_comparison(
    config: SimulationConfig,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition in default_conditions():
        true_log_ratio = float(np.log(condition.volatility / condition.stochasticity))
        for template_name in competitive_model_names():
            for episode in range(config.competitive_episodes):
                seed = stable_seed(config.seed, "competitive", condition.name, episode)
                env = TemporalSamplingEnv(
                    condition,
                    np.random.default_rng(stable_seed(seed, "env")),
                    max_time=config.time_budget
                    or config.steps * max(config.agent.action_space),
                )
                model = make_competitive_model(
                    template_name,
                    condition,
                    config,
                    stable_seed(seed, "model", template_name),
                )
                initial = env.reset()
                model.reset(initial.observation)
                cumulative_squared_error = 0.0

                cumulative_tracking_loss = 0.0
                cumulative_cost = 0.0
                for step in range(
                    config.steps if config.time_budget is None else config.time_budget
                ):
                    if (
                        config.time_budget is not None
                        and env.time >= config.time_budget
                    ):
                        break
                    feasible = tuple(
                        dt
                        for dt in config.agent.action_space
                        if config.time_budget is None
                        or dt <= config.time_budget - env.time
                    )
                    dt = model.choose_dt(step, feasible)
                    prediction = model.predict(dt)
                    result = env.step(dt)
                    prediction_error = result.observation - prediction.mean
                    squared_error = prediction_error**2
                    cumulative_squared_error += squared_error
                    predictive_nll = model.negative_log_likelihood(
                        result.observation, dt
                    )
                    cumulative_tracking_loss += float(
                        np.sum((np.asarray(result.latent_path) - prediction.mean) ** 2)
                    )
                    cumulative_cost += sampling_cost(
                        dt,
                        config.agent.wait_cost,
                        config.agent.observation_rate_cost,
                        config.agent.cost_mode,
                    )
                    estimate = model.update(result.observation, dt)
                    if np.isfinite(estimate.log_ratio):
                        ratio_abs_error = abs(estimate.log_ratio - true_log_ratio)
                    else:
                        ratio_abs_error = float("nan")

                    rows.append(
                        {
                            "condition": condition.name,
                            "model": model.name,
                            "episode": episode,
                            "step": step,
                            "time": result.time,
                            "dt": dt,
                            "true_volatility": condition.volatility,
                            "true_stochasticity": condition.stochasticity,
                            "latent_state": result.latent_state,
                            "observation": result.observation,
                            "prediction_before": prediction.mean,
                            "predictive_variance": prediction.variance,
                            "prediction_error": prediction_error,
                            "squared_error": squared_error,
                            "cumulative_squared_error": cumulative_squared_error,
                            "negative_log_likelihood": predictive_nll,
                            "cumulative_tracking_squared_error": cumulative_tracking_loss,
                            "cumulative_sampling_cost": cumulative_cost,
                            "estimated_volatility": estimate.volatility,
                            "estimated_stochasticity": estimate.stochasticity,
                            "estimated_log_ratio": estimate.log_ratio,
                            "true_log_ratio": true_log_ratio,
                            "ratio_abs_error": ratio_abs_error,
                            "learning_rate": estimate.learning_rate,
                        }
                    )
    return rows


def summarize_competitive_model_comparison(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["condition"]), str(row["model"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    for (condition, model), group in sorted(groups.items()):
        final_by_episode = {}
        cumulative_by_episode = {}
        for row in group:
            episode = int(row["episode"])
            final_by_episode[episode] = row
            cumulative_by_episode[episode] = float(row["cumulative_squared_error"])
        final_rows = list(final_by_episode.values())

        summary_rows.append(
            {
                "condition": condition,
                "model": model,
                "mean_mse": float(
                    np.mean([float(row["squared_error"]) for row in group])
                ),
                "mean_negative_log_likelihood": float(
                    np.mean([float(row["negative_log_likelihood"]) for row in group])
                ),
                "mean_cumulative_loss": float(
                    np.mean(list(cumulative_by_episode.values()))
                ),
                "mean_final_ratio_abs_error": finite_mean(
                    float(row["ratio_abs_error"]) for row in final_rows
                ),
                "mean_final_volatility": finite_mean(
                    float(row["estimated_volatility"]) for row in final_rows
                ),
                "mean_final_stochasticity": finite_mean(
                    float(row["estimated_stochasticity"]) for row in final_rows
                ),
                "mean_dt": float(np.mean([float(row["dt"]) for row in group])),
                "mean_elapsed_time": float(
                    np.mean([float(row["time"]) for row in final_rows])
                ),
                "mean_sampling_cost": float(
                    np.mean(
                        [float(row["cumulative_sampling_cost"]) for row in final_rows]
                    )
                ),
                "mean_time_tracking_mse": float(
                    np.mean(
                        [
                            float(row["cumulative_tracking_squared_error"])
                            / float(row["time"])
                            for row in final_rows
                        ]
                    )
                ),
                "mean_learning_rate": finite_mean(
                    float(row["learning_rate"]) for row in group
                ),
                "episodes": len(final_rows),
                "steps": len(group),
            }
        )
    return summary_rows


def competitive_model_names() -> tuple[str, ...]:
    return (
        "ats_joint_active",
        "joint_fixed_short",
        "joint_fixed_dt_2",
        "joint_fixed_dt_4",
        "joint_fixed_long",
        "joint_passive_grid",
        "kalman_fixed_active",
        "kalman_oracle_active",
        "pyhgf_full_active",
        "pyhgf_full_oracle_s_active",
        "pearce_hall_active",
        "delta_rule_active",
    )
