"""Experiment runner and CSV serialization for active temporal sampling."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np

from ats.agents import default_agent_names, make_agent
from ats.artifacts import prepare_run, complete_run, record_run_failures
from ats.policies import sampling_cost
from ats.beliefs import GridBelief
from ats.competitors import (
    run_competitive_model_comparison,
    summarize_competitive_model_comparison,
)
from ats.config import ExperimentCondition, SimulationConfig, default_conditions
from ats.envs import TemporalSamplingEnv
from ats.models import make_model_suite


def stable_seed(base_seed: int, *parts: object) -> int:
    text = "|".join([str(base_seed), *(str(part) for part in parts)])
    value = 2166136261
    for char in text:
        value = (value ^ ord(char)) * 16777619
        value = value % (2**32)
    return int(value)


def write_csv(
    path: Path, rows: list[dict[str, object]], fieldnames: Iterable[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def run_episode(
    condition: ExperimentCondition,
    agent_name: str,
    config: SimulationConfig,
    episode: int,
    seed: int,
    schedule: list[int] | None = None,
) -> list[dict[str, object]]:
    if schedule is not None:
        if any(dt not in config.agent.action_space for dt in schedule):
            raise ValueError("Replay schedule contains an unsupported interval.")
        if config.time_budget is None and len(schedule) != config.steps:
            raise ValueError("Replay schedule must use the full observation budget.")
        if config.time_budget is not None and sum(schedule) != config.time_budget:
            raise ValueError("Replay schedule must use the full physical-time budget.")
    volatility_grid, stochasticity_grid = config.grid.make_grids()
    horizon = config.time_budget or config.steps * max(config.agent.action_space)
    env = TemporalSamplingEnv(
        condition, np.random.default_rng(stable_seed(seed, "env")), max_time=horizon
    )
    agent = make_agent(
        "passive_grid" if schedule is not None else agent_name,
        config.agent,
        stable_seed(seed, "agent", agent_name),
    )
    belief = GridBelief(
        volatility_grid=volatility_grid,
        stochasticity_grid=stochasticity_grid,
        initial_variance=config.agent.initial_variance,
    )
    initial = env.reset()
    belief.reset(initial.observation)
    agent.reset()

    rows: list[dict[str, object]] = []
    true_log_ratio = float(np.log(condition.volatility / condition.stochasticity))
    cumulative_squared_error = 0.0
    cumulative_tracking_loss = 0.0
    cumulative_cost = 0.0
    decision_limit = config.steps if config.time_budget is None else config.time_budget

    for step in range(decision_limit):
        prediction_before = belief.predictive_mean()
        if config.time_budget is not None and env.time >= config.time_budget:
            break
        feasible = tuple(
            dt
            for dt in config.agent.action_space
            if config.time_budget is None or dt <= config.time_budget - env.time
        )
        agent.config = replace(config.agent, action_space=feasible)
        if schedule is not None:
            if step >= len(schedule):
                break
            dt = int(schedule[step])
            if dt not in feasible:
                raise ValueError("Replay schedule violates the sampling budget.")
        else:
            dt = agent.choose_dt(belief, step)
        predictive_variance = belief.mixture_predictive_variance(dt)
        result = env.step(dt)
        prediction_error = result.observation - prediction_before
        squared_error = prediction_error**2
        cumulative_squared_error += squared_error
        predictive_nll = -belief.predictive_log_density(result.observation, dt)
        tracking_loss = float(
            np.sum((np.asarray(result.latent_path) - prediction_before) ** 2)
        )
        cumulative_tracking_loss += tracking_loss
        cost = sampling_cost(
            dt,
            config.agent.wait_cost,
            config.agent.observation_rate_cost,
            config.agent.cost_mode,
        )
        cumulative_cost += cost
        summary = belief.update(result.observation, dt)
        v_interval = belief.marginal_interval("volatility")
        s_interval = belief.marginal_interval("stochasticity")
        ratio_abs_error = abs(summary.expected_log_ratio - true_log_ratio)

        rows.append(
            {
                "condition": condition.name,
                "agent": agent_name,
                "environment_seed": stable_seed(seed, "env"),
                "policy_seed": stable_seed(seed, "agent", agent_name),
                "episode": episode,
                "step": step,
                "time": result.time,
                "dt": dt,
                "true_volatility": condition.volatility,
                "true_stochasticity": condition.stochasticity,
                "latent_state": result.latent_state,
                "observation": result.observation,
                "prediction_before": prediction_before,
                "prediction_error": prediction_error,
                "squared_error": squared_error,
                "cumulative_squared_error": cumulative_squared_error,
                "predictive_variance": predictive_variance,
                "negative_log_likelihood": predictive_nll,
                "interval_tracking_squared_error": tracking_loss,
                "cumulative_tracking_squared_error": cumulative_tracking_loss,
                "cumulative_sampling_cost": cumulative_cost,
                "sampling_cost": cost,
                "volatility_log_abs_error": abs(
                    np.log(summary.volatility_mean / condition.volatility)
                ),
                "stochasticity_log_abs_error": abs(
                    np.log(summary.stochasticity_mean / condition.stochasticity)
                ),
                "volatility_ci_low": v_interval[0],
                "volatility_ci_high": v_interval[1],
                "stochasticity_ci_low": s_interval[0],
                "stochasticity_ci_high": s_interval[1],
                "volatility_covered_95": int(
                    v_interval[0] <= condition.volatility <= v_interval[1]
                ),
                "stochasticity_covered_95": int(
                    s_interval[0] <= condition.stochasticity <= s_interval[1]
                ),
                "posterior_volatility": summary.volatility_mean,
                "posterior_stochasticity": summary.stochasticity_mean,
                "posterior_volatility_sd": summary.volatility_sd,
                "posterior_stochasticity_sd": summary.stochasticity_sd,
                "posterior_log_ratio": summary.expected_log_ratio,
                "true_log_ratio": true_log_ratio,
                "ratio_abs_error": ratio_abs_error,
                "posterior_entropy": summary.entropy,
                "effective_hypotheses": summary.effective_n,
                "learning_rate": summary.learning_rate,
                "active_score_dt1": agent.last_scores.get(1, ""),
                "active_score_dt2": agent.last_scores.get(2, ""),
                "active_score_dt4": agent.last_scores.get(4, ""),
                "active_score_dt8": agent.last_scores.get(8, ""),
            }
        )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    true_volatilities = sorted({float(row["true_volatility"]) for row in rows})
    true_stochasticities = sorted({float(row["true_stochasticity"]) for row in rows})
    volatility_threshold = float(np.sqrt(true_volatilities[0] * true_volatilities[-1]))
    stochasticity_threshold = float(
        np.sqrt(true_stochasticities[0] * true_stochasticities[-1])
    )

    for row in rows:
        groups[(str(row["condition"]), str(row["agent"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    for (condition, agent), group in sorted(groups.items()):
        final_by_episode = {}
        cumulative_by_episode = {}
        for row in group:
            episode = int(row["episode"])
            final_by_episode[episode] = row
            cumulative_by_episode[episode] = float(row["cumulative_squared_error"])

        final_rows = list(final_by_episode.values())
        mean_final_ratio_error = float(
            np.mean([float(row["ratio_abs_error"]) for row in final_rows])
        )
        mean_final_v = float(
            np.mean([float(row["posterior_volatility"]) for row in final_rows])
        )
        mean_final_s = float(
            np.mean([float(row["posterior_stochasticity"]) for row in final_rows])
        )
        mean_cumulative_loss = float(np.mean(list(cumulative_by_episode.values())))
        mean_mse = float(np.mean([float(row["squared_error"]) for row in group]))
        mean_dt = float(np.mean([float(row["dt"]) for row in group]))
        volatility_class_accuracy = float(
            np.mean(
                [
                    (float(row["posterior_volatility"]) > volatility_threshold)
                    == (float(row["true_volatility"]) > volatility_threshold)
                    for row in final_rows
                ]
            )
        )
        stochasticity_class_accuracy = float(
            np.mean(
                [
                    (float(row["posterior_stochasticity"]) > stochasticity_threshold)
                    == (float(row["true_stochasticity"]) > stochasticity_threshold)
                    for row in final_rows
                ]
            )
        )
        joint_class_accuracy = float(
            np.mean(
                [
                    (
                        (float(row["posterior_volatility"]) > volatility_threshold)
                        == (float(row["true_volatility"]) > volatility_threshold)
                    )
                    and (
                        (
                            float(row["posterior_stochasticity"])
                            > stochasticity_threshold
                        )
                        == (float(row["true_stochasticity"]) > stochasticity_threshold)
                    )
                    for row in final_rows
                ]
            )
        )
        summary_rows.append(
            {
                "condition": condition,
                "agent": agent,
                "mean_mse": mean_mse,
                "mean_cumulative_loss": mean_cumulative_loss,
                "mean_final_ratio_abs_error": mean_final_ratio_error,
                "mean_final_volatility": mean_final_v,
                "mean_final_stochasticity": mean_final_s,
                "mean_dt": mean_dt,
                "volatility_class_accuracy": volatility_class_accuracy,
                "stochasticity_class_accuracy": stochasticity_class_accuracy,
                "joint_class_accuracy": joint_class_accuracy,
                "mean_elapsed_time": float(
                    np.mean([float(row["time"]) for row in final_rows])
                ),
                "mean_observation_count": len(group) / len(final_rows),
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
                "mean_final_volatility_log_abs_error": float(
                    np.mean(
                        [float(row["volatility_log_abs_error"]) for row in final_rows]
                    )
                ),
                "mean_final_stochasticity_log_abs_error": float(
                    np.mean(
                        [
                            float(row["stochasticity_log_abs_error"])
                            for row in final_rows
                        ]
                    )
                ),
                "volatility_coverage_95": float(
                    np.mean([float(row["volatility_covered_95"]) for row in final_rows])
                ),
                "stochasticity_coverage_95": float(
                    np.mean(
                        [float(row["stochasticity_covered_95"]) for row in final_rows]
                    )
                ),
                "episodes": len(final_rows),
                "steps": len(group),
            }
        )
    return summary_rows


def run_wait_cost_ablation(config: SimulationConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for wait_cost in config.ablation_wait_costs:
        ablation_agent = replace(config.agent, wait_cost=wait_cost)
        ablation_config = replace(
            config,
            agent=ablation_agent,
            episodes=config.ablation_episodes,
            include_ablation=False,
        )
        for condition in default_conditions():
            for episode in range(ablation_config.episodes):
                seed = stable_seed(config.seed, "ablation", condition.name, episode)
                episode_rows = run_episode(
                    condition, "active_temporal", ablation_config, episode, seed
                )
                for row in episode_rows:
                    row = dict(row)
                    row["wait_cost"] = wait_cost
                    rows.append(row)
    return rows


def summarize_wait_cost_ablation(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[float, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[float(row["wait_cost"])].append(row)

    summary_rows: list[dict[str, object]] = []
    for wait_cost, group in sorted(groups.items()):
        final_by_episode_condition = {}
        for row in group:
            key = (str(row["condition"]), int(row["episode"]))
            final_by_episode_condition[key] = row
        final_rows = list(final_by_episode_condition.values())
        summary_rows.append(
            {
                "wait_cost": wait_cost,
                "mean_final_ratio_abs_error": float(
                    np.mean([float(row["ratio_abs_error"]) for row in final_rows])
                ),
                "mean_mse": float(
                    np.mean([float(row["squared_error"]) for row in group])
                ),
                "mean_dt": float(np.mean([float(row["dt"]) for row in group])),
            }
        )
    return summary_rows


def finite_mean(values: Iterable[float]) -> float:
    finite_values = [float(value) for value in values if np.isfinite(float(value))]
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))


def run_model_comparison(config: SimulationConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    action_space = config.agent.action_space

    for condition in default_conditions():
        true_log_ratio = float(np.log(condition.volatility / condition.stochasticity))
        for episode in range(config.model_comparison_episodes):
            seed = stable_seed(config.seed, "model_comparison", condition.name, episode)
            env = TemporalSamplingEnv(
                condition,
                np.random.default_rng(seed),
                max_time=config.time_budget or config.steps * max(action_space),
            )
            initial = env.reset()
            models = make_model_suite(condition, config)
            for model in models:
                model.reset(initial.observation)

            for step in range(
                config.steps if config.time_budget is None else config.time_budget
            ):
                if config.time_budget is not None and env.time >= config.time_budget:
                    break
                feasible = tuple(
                    dt
                    for dt in action_space
                    if config.time_budget is None or dt <= config.time_budget - env.time
                )
                requested = action_space[step % len(action_space)]
                dt = max(
                    (value for value in feasible if value <= requested),
                    default=min(feasible),
                )
                result = env.step(dt)
                for model in models:
                    prediction = model.predict(dt)
                    prediction_error = result.observation - prediction.mean
                    squared_error = prediction_error**2
                    predictive_nll = model.negative_log_likelihood(
                        result.observation, dt
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
                            "negative_log_likelihood": predictive_nll,
                            "estimated_volatility": estimate.volatility,
                            "estimated_stochasticity": estimate.stochasticity,
                            "estimated_log_ratio": estimate.log_ratio,
                            "true_log_ratio": true_log_ratio,
                            "ratio_abs_error": ratio_abs_error,
                            "learning_rate": estimate.learning_rate,
                        }
                    )
    return rows


def summarize_model_comparison(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    true_volatilities = sorted({float(row["true_volatility"]) for row in rows})
    true_stochasticities = sorted({float(row["true_stochasticity"]) for row in rows})
    volatility_threshold = float(np.sqrt(true_volatilities[0] * true_volatilities[-1]))
    stochasticity_threshold = float(
        np.sqrt(true_stochasticities[0] * true_stochasticities[-1])
    )

    for row in rows:
        groups[(str(row["condition"]), str(row["model"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    for (condition, model), group in sorted(groups.items()):
        final_by_episode = {}
        for row in group:
            final_by_episode[int(row["episode"])] = row
        final_rows = list(final_by_episode.values())

        finite_ratio_rows = [
            row for row in final_rows if np.isfinite(float(row["ratio_abs_error"]))
        ]
        finite_volatility_rows = [
            row for row in final_rows if np.isfinite(float(row["estimated_volatility"]))
        ]
        finite_stochasticity_rows = [
            row
            for row in final_rows
            if np.isfinite(float(row["estimated_stochasticity"]))
        ]

        volatility_class_accuracy = float("nan")
        stochasticity_class_accuracy = float("nan")
        joint_class_accuracy = float("nan")
        if finite_volatility_rows:
            volatility_class_accuracy = float(
                np.mean(
                    [
                        (float(row["estimated_volatility"]) > volatility_threshold)
                        == (float(row["true_volatility"]) > volatility_threshold)
                        for row in finite_volatility_rows
                    ]
                )
            )
        if finite_stochasticity_rows:
            stochasticity_class_accuracy = float(
                np.mean(
                    [
                        (
                            float(row["estimated_stochasticity"])
                            > stochasticity_threshold
                        )
                        == (float(row["true_stochasticity"]) > stochasticity_threshold)
                        for row in finite_stochasticity_rows
                    ]
                )
            )
        if finite_ratio_rows:
            joint_class_accuracy = float(
                np.mean(
                    [
                        (
                            (float(row["estimated_volatility"]) > volatility_threshold)
                            == (float(row["true_volatility"]) > volatility_threshold)
                        )
                        and (
                            (
                                float(row["estimated_stochasticity"])
                                > stochasticity_threshold
                            )
                            == (
                                float(row["true_stochasticity"])
                                > stochasticity_threshold
                            )
                        )
                        for row in finite_ratio_rows
                    ]
                )
            )

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
                "mean_final_ratio_abs_error": finite_mean(
                    float(row["ratio_abs_error"]) for row in final_rows
                ),
                "mean_final_volatility": finite_mean(
                    float(row["estimated_volatility"]) for row in final_rows
                ),
                "mean_final_stochasticity": finite_mean(
                    float(row["estimated_stochasticity"]) for row in final_rows
                ),
                "mean_learning_rate": finite_mean(
                    float(row["learning_rate"]) for row in group
                ),
                "volatility_class_accuracy": volatility_class_accuracy,
                "stochasticity_class_accuracy": stochasticity_class_accuracy,
                "joint_class_accuracy": joint_class_accuracy,
                "episodes": len(final_rows),
                "steps": len(group),
            }
        )
    return summary_rows


def summarize_paired_comparisons(
    rows: list[dict[str, object]], seed: int, replicates: int = 2000
) -> list[dict[str, object]]:
    """Bootstrap matched episode contrasts, stratified by condition without treating time steps as replicates."""
    finals = {}
    episode_errors = defaultdict(list)
    for row in rows:
        key = (str(row["condition"]), str(row["agent"]), int(row["episode"]))
        finals[key] = row
        episode_errors[key].append(float(row["squared_error"]))
    conditions = sorted({key[0] for key in finals})
    agents = sorted({key[1] for key in finals} - {"active_temporal"})
    metrics = {
        "final_ratio_error": lambda key: float(finals[key]["ratio_abs_error"]),
        "time_tracking_mse": lambda key: float(
            finals[key]["cumulative_tracking_squared_error"]
        )
        / float(finals[key]["time"]),
        "observation_forecast_mse": lambda key: float(np.mean(episode_errors[key])),
        "total_sampling_cost": lambda key: float(
            finals[key]["cumulative_sampling_cost"]
        ),
        "final_volatility_log_error": lambda key: float(
            finals[key]["volatility_log_abs_error"]
        ),
        "final_stochasticity_log_error": lambda key: float(
            finals[key]["stochasticity_log_abs_error"]
        ),
    }
    comparisons = []
    for baseline in agents:
        for metric, getter in metrics.items():
            rng = np.random.default_rng(
                stable_seed(seed, "paired_bootstrap", baseline, metric)
            )
            draws, differences, active_means, baseline_means = [], [], [], []
            pair_count = 0
            for condition in conditions:
                episodes = sorted(
                    key[2]
                    for key in finals
                    if key[0] == condition and key[1] == "active_temporal"
                )
                active = np.array(
                    [
                        getter((condition, "active_temporal", episode))
                        for episode in episodes
                    ]
                )
                control = np.array(
                    [getter((condition, baseline, episode)) for episode in episodes]
                )
                difference = active - control
                pair_count += len(episodes)
                draws.append(
                    difference[
                        rng.integers(
                            0, len(difference), size=(replicates, len(difference))
                        )
                    ].mean(axis=1)
                )
                differences.append(float(np.mean(difference)))
                active_means.append(float(np.mean(active)))
                baseline_means.append(float(np.mean(control)))
            lower, upper = np.quantile(np.mean(draws, axis=0), [0.025, 0.975])
            comparisons.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "active_mean": float(np.mean(active_means)),
                    "baseline_mean": float(np.mean(baseline_means)),
                    "active_minus_baseline": float(np.mean(differences)),
                    "bootstrap_ci_low": float(lower),
                    "bootstrap_ci_high": float(upper),
                    "paired_episodes": pair_count,
                    "conditions": len(conditions),
                    "bootstrap_replicates": replicates,
                    "multiplicity_adjustment": "none",
                    "retrospective_control": baseline == "matched_histogram",
                }
            )
    return comparisons


def save_example_posterior(config: SimulationConfig, output_dir: Path) -> None:
    """Save the prespecified first active episode, without outcome-dependent selection."""
    condition = default_conditions()[0]
    seed = stable_seed(config.seed, condition.name, 0)
    rows = run_episode(condition, "active_temporal", config, 0, seed)
    volatility_grid, stochasticity_grid = config.grid.make_grids()
    belief = GridBelief(
        volatility_grid, stochasticity_grid, config.agent.initial_variance
    )
    env = TemporalSamplingEnv(
        condition,
        np.random.default_rng(stable_seed(seed, "env")),
        max_time=config.time_budget or config.steps * max(config.agent.action_space),
    )
    belief.reset(env.reset().observation)
    posterior_history = []
    for row in rows:
        belief.update(float(row["observation"]), int(row["dt"]))
        posterior_history.append(belief.posterior.copy())
    np.savez_compressed(
        output_dir / "example_posterior.npz",
        posterior=belief.posterior,
        posterior_history=np.asarray(posterior_history),
        ratio_history=np.asarray([row["posterior_log_ratio"] for row in rows]),
        dt_history=np.asarray([row["dt"] for row in rows]),
        volatility_grid=volatility_grid,
        stochasticity_grid=stochasticity_grid,
        true_volatility=condition.volatility,
        true_stochasticity=condition.stochasticity,
        selection_rule="prespecified first condition, episode zero",
        episode=0,
        observation_count=len(rows),
    )


@record_run_failures
def run_experiment(config: SimulationConfig) -> Path:
    output_dir = config.output_dir()
    prepare_run(output_dir, config, "ats_simulation")

    rows: list[dict[str, object]] = []
    for condition in default_conditions():
        for episode in range(config.episodes):
            seed = stable_seed(config.seed, condition.name, episode)
            active_rows = run_episode(
                condition, "active_temporal", config, episode, seed
            )
            for agent_name in default_agent_names():
                if agent_name == "active_temporal":
                    rows.extend(active_rows)
                else:
                    rows.extend(
                        run_episode(condition, agent_name, config, episode, seed)
                    )
            schedule_rng = np.random.default_rng(stable_seed(seed, "matched_histogram"))
            schedule = [int(row["dt"]) for row in active_rows]
            schedule_rng.shuffle(schedule)
            rows.extend(
                run_episode(
                    condition,
                    "matched_histogram",
                    config,
                    episode,
                    seed,
                    schedule=schedule,
                )
            )
            donor_seed = stable_seed(
                config.seed, "independent_yoked_donor", condition.name, episode
            )
            donor = run_episode(
                condition, "active_temporal", config, episode, donor_seed
            )
            rows.extend(
                run_episode(
                    condition,
                    "yoked_independent",
                    config,
                    episode,
                    seed,
                    schedule=[int(row["dt"]) for row in donor],
                )
            )

    fieldnames = list(rows[0].keys()) if rows else []
    write_csv(output_dir / "metrics.csv", rows, fieldnames)
    write_csv(output_dir / "trajectories.csv", rows, fieldnames)
    summary_rows = summarize(rows)
    write_csv(output_dir / "summary.csv", summary_rows, summary_rows[0].keys())
    comparisons = summarize_paired_comparisons(rows, config.seed)
    write_csv(output_dir / "paired_comparisons.csv", comparisons, comparisons[0].keys())
    save_example_posterior(config, output_dir)

    if config.include_ablation:
        ablation_rows = run_wait_cost_ablation(config)
        write_csv(
            output_dir / "wait_cost_ablation.csv",
            ablation_rows,
            ablation_rows[0].keys(),
        )
        ablation_summary = summarize_wait_cost_ablation(ablation_rows)
        write_csv(
            output_dir / "wait_cost_ablation_summary.csv",
            ablation_summary,
            ablation_summary[0].keys(),
        )

    if config.include_model_comparison:
        model_rows = run_model_comparison(config)
        write_csv(
            output_dir / "model_comparison_metrics.csv",
            model_rows,
            model_rows[0].keys(),
        )
        model_summary = summarize_model_comparison(model_rows)
        write_csv(
            output_dir / "model_comparison_summary.csv",
            model_summary,
            model_summary[0].keys(),
        )

    if config.include_competitive_comparison:
        competitive_rows = run_competitive_model_comparison(config)
        write_csv(
            output_dir / "competitive_model_metrics.csv",
            competitive_rows,
            competitive_rows[0].keys(),
        )
        competitive_summary = summarize_competitive_model_comparison(competitive_rows)
        write_csv(
            output_dir / "competitive_model_summary.csv",
            competitive_summary,
            competitive_summary[0].keys(),
        )

    complete_run(
        output_dir,
        budget_basis=(
            "physical_time" if config.time_budget is not None else "observation_count"
        ),
        likelihood="exact Gaussian mixture for grid models",
        posterior_example="prespecified condition zero, episode zero",
        control_notes={
            "matched_histogram": "retrospective diagnostic using the recipient active schedule; not a deployable policy",
            "yoked_independent": "prespecified active schedule from an independent donor trajectory in the same condition",
        },
    )
    return output_dir
