"""Training loop for the Deep Active Temporal Sampling planner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from deep_ats_planner.beliefs import GridBelief
from deep_ats_planner.config import (
    DeepPlannerConfig,
    ExperimentCondition,
    default_conditions,
)
from ats.artifacts import prepare_run, complete_run, json_value, record_run_failures
from ats.policies import sampling_cost
from deep_ats_planner.experiments import stable_seed, write_csv
from deep_ats_planner.planner import DeepATSPlanner, ReplayTransition
from deep_ats_planner.rl_envs import TemporalBanditEnv
from deep_ats_planner.state import belief_features, feature_names


def epsilon_at_step(config: DeepPlannerConfig, global_step: int) -> float:
    progress = min(
        1.0, float(global_step) / max(float(config.epsilon_decay_steps), 1.0)
    )
    return float(
        config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)
    )


def temporal_cost(config: DeepPlannerConfig, dt: int) -> float:
    return sampling_cost(
        dt,
        config.agent.wait_cost,
        config.agent.observation_rate_cost,
        config.agent.cost_mode,
    )


def transition_return(config: DeepPlannerConfig, task_reward: float, dt: int) -> float:
    """Discount an endpoint payoff and observation charge plus each unit of waiting cost."""
    if config.discount_basis == "decision":
        return task_reward - temporal_cost(config, dt)
    endpoint_cost = (
        config.agent.observation_rate_cost
        if config.agent.cost_mode == "per_observation"
        else config.agent.observation_rate_cost / dt
    )
    waiting = config.agent.wait_cost * sum(config.gamma**index for index in range(dt))
    return config.gamma ** (dt - 1) * (task_reward - endpoint_cost) - waiting


def make_planner(config: DeepPlannerConfig, input_dim: int) -> DeepATSPlanner:
    return DeepATSPlanner(
        input_dim=input_dim,
        task_action_count=config.task_action_count,
        action_space=config.agent.action_space,
        hidden_size=config.hidden_size,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        replay_capacity=config.replay_capacity,
        discount_basis=config.discount_basis,
    )


@record_run_failures
def run_deep_planner(config: DeepPlannerConfig) -> Path:
    output_dir = config.output_dir()
    prepare_run(output_dir, config, "belief_conditioned_dqn")
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    input_dim = len(feature_names(config.task_action_count, config.agent.action_space))
    planner = make_planner(config, input_dim)
    volatility_grid, stochasticity_grid = config.grid.make_grids()
    conditions = default_conditions()
    step_rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []
    global_step = 0

    for episode in range(config.episodes):
        condition = conditions[episode % len(conditions)]
        env = TemporalBanditEnv(
            condition=condition,
            rng=np.random.default_rng(
                stable_seed(config.seed, "deep_env", episode, condition.name)
            ),
            reward_scale=config.reward_scale,
            max_time=config.steps * max(config.agent.action_space),
        )
        belief = GridBelief(
            volatility_grid=volatility_grid,
            stochasticity_grid=stochasticity_grid,
            initial_variance=config.agent.initial_variance,
        )
        initial = env.reset()
        belief.reset(initial.observation)
        last_observation = initial.observation
        last_reward = 0.0
        last_action = -1
        last_dt = min(config.agent.action_space)
        cumulative_reward = 0.0
        cumulative_discounted_return = 0.0
        cumulative_task_reward = 0.0
        cumulative_cost = 0.0
        correct_actions = 0
        losses: list[float] = []
        dt_values: list[int] = []

        for step in range(config.steps):
            state = belief_features(
                belief=belief,
                action_space=config.agent.action_space,
                task_action_count=config.task_action_count,
                last_observation=last_observation,
                last_reward=last_reward,
                last_action=last_action,
                last_dt=last_dt,
                remaining_decisions_fraction=(config.steps - step) / config.steps,
                use_belief_features=config.use_belief_features,
            )
            epsilon = epsilon_at_step(config, global_step)
            task_action, dt, action_index, q_values = planner.select(
                state, epsilon, rng
            )
            result = env.step(task_action, dt)
            summary = belief.update(result.observation, dt)
            cost = temporal_cost(config, dt)
            reward = result.reward - cost
            cumulative_reward += reward
            elapsed_before = (
                result.time - dt if config.discount_basis == "physical_time" else step
            )
            cumulative_discounted_return += (
                config.gamma**elapsed_before
                * transition_return(config, result.reward, dt)
            )
            cumulative_task_reward += result.reward
            cumulative_cost += cost
            dt_values.append(dt)
            optimal_action = int(result.latent_state >= 0.0)
            correct_actions += int(task_action == optimal_action)

            next_state = belief_features(
                belief=belief,
                action_space=config.agent.action_space,
                task_action_count=config.task_action_count,
                last_observation=result.observation,
                last_reward=result.reward,
                last_action=task_action,
                last_dt=dt,
                remaining_decisions_fraction=(config.steps - step - 1) / config.steps,
                use_belief_features=config.use_belief_features,
            )
            done = step == config.steps - 1
            planner.replay.add(
                ReplayTransition(
                    state=state,
                    action_index=action_index,
                    reward=transition_return(config, result.reward, dt),
                    next_state=next_state,
                    done=done,
                    duration=dt,
                )
            )
            loss = None
            if global_step >= config.warmup_steps:
                loss = planner.train_step(config.batch_size, rng)
                if loss is not None:
                    losses.append(loss)
            if global_step % config.target_update_interval == 0:
                planner.update_target()

            step_rows.append(
                {
                    "episode": episode,
                    "step": step,
                    "global_step": global_step,
                    "condition": condition.name,
                    "true_volatility": condition.volatility,
                    "true_stochasticity": condition.stochasticity,
                    "time": result.time,
                    "task_action": task_action,
                    "optimal_action": optimal_action,
                    "dt": dt,
                    "latent_state": result.latent_state,
                    "observation": result.observation,
                    "reward": result.reward,
                    "temporal_cost": cost,
                    "net_reward": reward,
                    "discounted_transition_return": transition_return(
                        config, result.reward, dt
                    ),
                    "reward_probability": result.reward_probability,
                    "posterior_volatility": summary.volatility_mean,
                    "posterior_stochasticity": summary.stochasticity_mean,
                    "posterior_log_ratio": summary.expected_log_ratio,
                    "posterior_entropy": summary.entropy,
                    "learning_rate": summary.learning_rate,
                    "epsilon": epsilon,
                    "loss": "" if loss is None else loss,
                    "max_q": float(np.max(q_values)),
                }
            )
            last_observation = result.observation
            last_reward = result.reward
            last_action = task_action
            last_dt = dt
            global_step += 1

        episode_rows.append(
            {
                "episode": episode,
                "condition": condition.name,
                "cumulative_net_reward": cumulative_reward,
                "discounted_return": cumulative_discounted_return,
                "cumulative_task_reward": cumulative_task_reward,
                "cumulative_temporal_cost": cumulative_cost,
                "mean_dt": float(np.mean(dt_values)),
                "action_accuracy": correct_actions / max(config.steps, 1),
                "mean_loss": "" if not losses else float(np.mean(losses)),
                "final_epsilon": epsilon_at_step(config, global_step),
            }
        )

    summary_rows = summarize_deep_planner(episode_rows)
    write_csv(
        output_dir / "deep_steps.csv",
        step_rows,
        step_rows[0].keys() if step_rows else [],
    )
    write_csv(
        output_dir / "deep_episodes.csv",
        episode_rows,
        episode_rows[0].keys() if episode_rows else [],
    )
    write_csv(
        output_dir / "deep_summary.csv",
        summary_rows,
        summary_rows[0].keys() if summary_rows else [],
    )
    torch.save(
        {
            "state_dict": planner.network.state_dict(),
            "target_state_dict": planner.target_network.state_dict(),
            "joint_actions": [
                (item.task_action, item.dt) for item in planner.joint_actions
            ],
            "feature_names": feature_names(
                config.task_action_count, config.agent.action_space
            ),
            "config": json_value(config),
        },
        output_dir / "deep_ats_planner.pt",
    )
    evaluation = evaluate_deep_planner(planner, config)
    write_csv(output_dir / "deep_evaluation.csv", evaluation, evaluation[0].keys())
    complete_run(
        output_dir,
        evaluation_seed_namespace="deep_evaluation",
        discount_basis=config.discount_basis,
        inference="shared exact grid posterior; Q function is learned",
        held_out_condition="held_out_mid",
    )
    return output_dir


def evaluate_deep_planner(
    planner: DeepATSPlanner, config: DeepPlannerConfig
) -> list[dict[str, object]]:
    """Compare a frozen greedy policy with prespecified reactive schedules on new seeds."""
    rows = []
    conditions = default_conditions() + (ExperimentCondition("held_out_mid", 1.0, 3.0),)
    schedules = {
        "learned_greedy": None,
        **{f"reactive_dt_{dt}": dt for dt in config.agent.action_space},
    }
    for condition in conditions:
        for episode in range(config.evaluation_episodes):
            seed = stable_seed(config.seed, "deep_evaluation", condition.name, episode)
            for name, fixed_dt in schedules.items():
                env = TemporalBanditEnv(
                    condition,
                    np.random.default_rng(seed),
                    config.reward_scale,
                    config.steps * max(config.agent.action_space),
                )
                v, s = config.grid.make_grids()
                belief = GridBelief(v, s, config.agent.initial_variance)
                initial = env.reset()
                belief.reset(initial.observation)
                last_observation, last_reward, last_action, last_dt = (
                    initial.observation,
                    0.0,
                    -1,
                    min(config.agent.action_space),
                )
                rewards, costs, accuracy = 0.0, 0.0, 0
                discounted_return = 0.0
                durations = []
                for step in range(config.steps):
                    state = belief_features(
                        belief,
                        config.agent.action_space,
                        config.task_action_count,
                        last_observation,
                        last_reward,
                        last_action,
                        last_dt,
                        (config.steps - step) / config.steps,
                        config.use_belief_features,
                    )
                    if fixed_dt is None:
                        action, dt, _, _ = planner.select(
                            state,
                            0.0,
                            np.random.default_rng(stable_seed(seed, name, step)),
                        )
                    else:
                        action, dt = int(belief.predictive_mean() >= 0), fixed_dt
                    result = env.step(action, dt)
                    belief.update(result.observation, dt)
                    rewards += result.reward
                    elapsed_before = (
                        result.time - dt
                        if config.discount_basis == "physical_time"
                        else step
                    )
                    discounted_return += (
                        config.gamma**elapsed_before
                        * transition_return(config, result.reward, dt)
                    )
                    costs += temporal_cost(config, dt)
                    accuracy += int(action == int(result.latent_state >= 0))
                    durations.append(dt)
                    last_observation, last_reward, last_action, last_dt = (
                        result.observation,
                        result.reward,
                        action,
                        dt,
                    )
                rows.append(
                    {
                        "condition": condition.name,
                        "held_out_parameters": condition.name == "held_out_mid",
                        "episode": episode,
                        "seed": seed,
                        "policy": name,
                        "task_reward": rewards,
                        "sampling_cost": costs,
                        "net_reward": rewards - costs,
                        "discounted_return": discounted_return,
                        "action_accuracy": accuracy / config.steps,
                        "elapsed_time": sum(durations),
                        "observations": config.steps,
                        "mean_dt": float(np.mean(durations)),
                        "evaluation": "frozen_policy_independent_seeds",
                    }
                )
    return rows


def summarize_deep_planner(
    episode_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in episode_rows:
        groups.setdefault(str(row["condition"]), []).append(row)

    rows: list[dict[str, object]] = []
    for condition, group in sorted(groups.items()):
        rows.append(
            {
                "condition": condition,
                "episodes": len(group),
                "mean_cumulative_net_reward": float(
                    np.mean([float(row["cumulative_net_reward"]) for row in group])
                ),
                "mean_discounted_return": float(
                    np.mean([float(row["discounted_return"]) for row in group])
                ),
                "mean_cumulative_task_reward": float(
                    np.mean([float(row["cumulative_task_reward"]) for row in group])
                ),
                "mean_temporal_cost": float(
                    np.mean([float(row["cumulative_temporal_cost"]) for row in group])
                ),
                "mean_dt": float(np.mean([float(row["mean_dt"]) for row in group])),
                "mean_action_accuracy": float(
                    np.mean([float(row["action_accuracy"]) for row in group])
                ),
            }
        )
    return rows
