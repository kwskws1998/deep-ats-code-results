"""PPO training with optional Deep ATS observation features."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, replace
from pathlib import Path
from contextlib import ExitStack

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal

from deep_ats_planner.ats_encoder import (
    ATSFeatureSpec,
    NullATSEncoder,
    RandomProjectionATSEncoder,
    HistoryEncoder,
    EMAEncoder,
    PaddedEncoder,
)
from ats.artifacts import prepare_run, complete_run, json_value, record_run_failures
from ats.experiments import stable_seed
from deep_ats_planner.config import AgentConfig, GridConfig
from deep_ats_planner.device import select_device
from deep_ats_planner.rwrl_wrapper import RWRLSpec, make_continuous_env


@dataclass(frozen=True)
class PPOConfig:
    run_name: str = "deep_ats_ppo"
    output_root: Path = Path("outputs")
    env_id: str = "Pendulum-v1"
    seed: int = 20260521
    total_timesteps: int = 20_000
    num_envs: int = 4
    num_steps: int = 128
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    normalize_advantages: bool = True
    use_ats: bool = True
    device: str = "auto"
    hidden_size: int = 128
    ats_projection_count: int = 8
    rwrl_domain: str = "walker"
    rwrl_task: str = "realworld_walk"
    rwrl_combined_challenge: str = "easy"
    feature_mode: str | None = None
    history_length: int = 5
    evaluation_episodes: int = 5
    evaluation_max_steps: int = 10000

    def __post_init__(self) -> None:
        """Reject invalid rollout shapes and budgets before allocating buffers."""
        for name in (
            "num_envs",
            "num_steps",
            "num_minibatches",
            "update_epochs",
            "hidden_size",
            "ats_projection_count",
            "history_length",
            "evaluation_episodes",
            "evaluation_max_steps",
        ):
            if (
                int(getattr(self, name)) != getattr(self, name)
                or getattr(self, name) < 1
            ):
                raise ValueError(f"{name} must be a positive integer.")
        if self.total_timesteps < self.batch_size:
            raise ValueError(
                "total_timesteps must cover at least one full rollout batch."
            )
        if self.batch_size % self.num_minibatches:
            raise ValueError("Rollout batch size must be divisible by num_minibatches.")
        if self.resolved_feature_mode not in {"ats", "none", "history", "ema"}:
            raise ValueError("feature_mode must be ats, none, history, or ema.")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must lie in [0, 1].")

    @property
    def resolved_feature_mode(self) -> str:
        return (
            self.feature_mode
            if self.feature_mode is not None
            else ("ats" if self.use_ats else "none")
        )

    @property
    def batch_size(self) -> int:
        return int(self.num_envs * self.num_steps)

    @property
    def minibatch_size(self) -> int:
        return int(self.batch_size // self.num_minibatches)

    def output_dir(self) -> Path:
        return self.output_root / self.run_name


class RunningMeanStd:
    """Online observation normalizer."""

    def __init__(self, shape: tuple[int, ...]):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        batch_mean = np.mean(values, axis=0)
        batch_var = np.var(values, axis=0)
        batch_count = values.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m_2 / total_count
        self.count = total_count

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return np.clip((values - self.mean) / np.sqrt(self.var + 1e-8), -10, 10).astype(
            np.float32
        )


class ActorCritic(nn.Module):
    """Gaussian actor-critic for bounded continuous-action PPO."""

    def __init__(self, observation_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(observation_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(observation_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, observation: torch.Tensor) -> torch.Tensor:
        return self.critic(observation)

    def get_action_and_value(
        self,
        observation: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean = self.actor_mean(observation)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        distribution = Normal(action_mean, action_std)
        if action is None:
            action = distribution.sample()
        logprob = distribution.log_prob(action).sum(1)
        entropy = distribution.entropy().sum(1)
        value = self.critic(observation)
        return action, logprob, entropy, value


def layer_init(
    layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def make_encoder(
    observation_dim: int, config: PPOConfig, env_index: int = 0, observation_scale=None
):
    """Use a shared projection basis and matched input widths in every vector environment."""
    width = max(15, observation_dim * config.history_length, 3 * observation_dim)
    mode = config.resolved_feature_mode
    if mode == "none":
        encoder = NullATSEncoder()
    elif mode == "history":
        encoder = HistoryEncoder(observation_dim, config.history_length)
    elif mode == "ema":
        encoder = EMAEncoder(observation_dim)
    else:
        encoder = RandomProjectionATSEncoder(
            observation_dim=observation_dim,
            grid=GridConfig(size=17, volatility_min=1e-5, stochasticity_min=1e-5),
            agent=AgentConfig(action_space=(1, 2, 4, 8)),
            spec=ATSFeatureSpec(
                projection_count=config.ats_projection_count,
                projection_seed=config.seed,
                update_dt=1,
                observation_scale=(
                    None if observation_scale is None else tuple(observation_scale)
                ),
            ),
        )
    return PaddedEncoder(encoder, width)


def observation_scales(space) -> np.ndarray:
    """Use fixed public observation bounds rather than changing online scales inside the filters."""
    low = np.asarray(space.low, dtype=float).reshape(-1)
    high = np.asarray(space.high, dtype=float).reshape(-1)
    width = (high - low) / 2
    return np.where(np.isfinite(width) & (width > 0), width, 1.0)


def generalized_advantage_estimate(
    rewards, values, next_values, terminated, episode_ends, gamma, gae_lambda
):
    """Bootstrap timeouts from their final state and stop traces before reset episodes."""
    advantages = torch.zeros_like(rewards)
    following = torch.zeros_like(rewards[0])
    for step in reversed(range(rewards.shape[0])):
        delta = (
            rewards[step]
            + gamma * next_values[step] * (1 - terminated[step])
            - values[step]
        )
        following = delta + gamma * gae_lambda * (1 - episode_ends[step]) * following
        advantages[step] = following
    return advantages, advantages + values


def augment_observation(
    raw_observation: np.ndarray, ats_features: np.ndarray
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(raw_observation, dtype=np.float32).reshape(-1),
            np.asarray(ats_features, dtype=np.float32),
        ],
        dtype=np.float32,
    )


@record_run_failures
def train_ppo(config: PPOConfig) -> Path:
    """Close all training environments even if model construction or rollout fails."""
    with ExitStack() as resources:
        return _train_ppo(config, resources)


def _train_ppo(config: PPOConfig, resources: ExitStack) -> Path:
    output_dir = config.output_dir()
    prepare_run(output_dir, config, "ppo_feature_ablation")
    device = select_device(config.device)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    rwrl_spec = RWRLSpec(
        domain_name=config.rwrl_domain,
        task_name=config.rwrl_task,
        combined_challenge=config.rwrl_combined_challenge,
        log_output=(
            str(output_dir / "rwrl_log.npz") if config.env_id == "rwrl" else None
        ),
    )
    envs = []
    for env_index in range(config.num_envs):
        spec = replace(
            rwrl_spec,
            log_output=(
                str(output_dir / f"rwrl_env_{env_index}.npz")
                if config.env_id == "rwrl"
                else None
            ),
        )
        env = make_continuous_env(config.env_id, config.seed + env_index, spec)
        resources.callback(env.close)
        envs.append(env)
    if not isinstance(envs[0].action_space, gym.spaces.Box):
        raise TypeError("Deep ATS-PPO currently expects a continuous Box action space.")
    action_space = envs[0].action_space
    action_dim = int(np.prod(action_space.shape))

    raw_observations = []
    infos = []
    for env_index, env in enumerate(envs):
        observation, info = env.reset(seed=config.seed + env_index)
        raw_observations.append(np.asarray(observation, dtype=np.float32).reshape(-1))
        infos.append(info)
    raw_obs = np.stack(raw_observations)
    raw_dim = raw_obs.shape[1]
    scales = observation_scales(envs[0].observation_space)
    encoders = [
        make_encoder(raw_dim, config, env_index, scales)
        for env_index in range(config.num_envs)
    ]
    ats_features = np.stack(
        [encoder.reset(raw_obs[index]) for index, encoder in enumerate(encoders)]
    )
    obs = np.stack(
        [
            augment_observation(raw_obs[index], ats_features[index])
            for index in range(config.num_envs)
        ]
    )
    obs_rms = RunningMeanStd(shape=(obs.shape[1],))
    obs_rms.update(obs)
    normalized_obs = obs_rms.normalize(obs)

    agent = ActorCritic(obs.shape[1], action_dim, config.hidden_size).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    obs_buffer = torch.zeros(
        (config.num_steps, config.num_envs, obs.shape[1]), device=device
    )
    actions_buffer = torch.zeros(
        (config.num_steps, config.num_envs, action_dim), device=device
    )
    logprobs_buffer = torch.zeros((config.num_steps, config.num_envs), device=device)
    rewards_buffer = torch.zeros((config.num_steps, config.num_envs), device=device)
    ends_buffer = torch.zeros((config.num_steps, config.num_envs), device=device)
    terminated_buffer = torch.zeros_like(ends_buffer)
    next_values_buffer = torch.zeros_like(ends_buffer)
    values_buffer = torch.zeros((config.num_steps, config.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    episode_returns = np.zeros(config.num_envs, dtype=float)
    episode_lengths = np.zeros(config.num_envs, dtype=int)
    metric_rows: list[dict[str, object]] = []
    update_count = config.total_timesteps // config.batch_size

    for update in range(1, update_count + 1):
        frac = 1.0 - (update - 1.0) / update_count
        optimizer.param_groups[0]["lr"] = frac * config.learning_rate

        normalization_rows = []
        for step in range(config.num_steps):
            global_step += config.num_envs
            obs_buffer[step] = torch.as_tensor(
                normalized_obs, dtype=torch.float32, device=device
            )

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_buffer[step])
                values_buffer[step] = value.flatten()
            actions_buffer[step] = action
            logprobs_buffer[step] = logprob

            clipped_actions = np.clip(
                action.detach()
                .cpu()
                .numpy()
                .reshape(config.num_envs, *action_space.shape),
                action_space.low,
                action_space.high,
            )
            next_raw_obs = []
            next_rewards = []
            final_augmented_obs = []
            step_terminated = []
            step_ends = []

            for env_index, env in enumerate(envs):
                observation, reward, terminated, truncated, info = env.step(
                    clipped_actions[env_index]
                )
                done = bool(terminated or truncated)
                episode_returns[env_index] += float(reward)
                episode_lengths[env_index] += 1
                final_raw = np.asarray(observation, dtype=np.float32).reshape(-1)
                ats_feature = encoders[env_index].update(final_raw)
                final_augmented_obs.append(augment_observation(final_raw, ats_feature))
                step_terminated.append(float(terminated))
                step_ends.append(float(done))
                if done:
                    metric_rows.append(
                        {
                            "global_step": global_step,
                            "update": update,
                            "env_index": env_index,
                            "episode_return": episode_returns[env_index],
                            "episode_length": episode_lengths[env_index],
                            "use_ats": config.resolved_feature_mode == "ats",
                            "feature_mode": config.resolved_feature_mode,
                        }
                    )
                    observation, _ = env.reset(
                        seed=config.seed + env_index + global_step
                    )
                    ats_feature = encoders[env_index].reset(
                        np.asarray(observation, dtype=np.float32).reshape(-1)
                    )
                    episode_returns[env_index] = 0.0
                    episode_lengths[env_index] = 0
                next_raw_obs.append(
                    np.asarray(observation, dtype=np.float32).reshape(-1)
                )
                next_rewards.append(float(reward))
                ats_features[env_index] = ats_feature

            with torch.no_grad():
                final_inputs = torch.as_tensor(
                    obs_rms.normalize(np.stack(final_augmented_obs)), device=device
                )
                next_values_buffer[step] = agent.get_value(final_inputs).flatten()
            rewards_buffer[step] = torch.as_tensor(
                next_rewards, dtype=torch.float32, device=device
            )
            terminated_buffer[step] = torch.as_tensor(
                step_terminated, dtype=torch.float32, device=device
            )
            ends_buffer[step] = torch.as_tensor(
                step_ends, dtype=torch.float32, device=device
            )
            raw_obs = np.stack(next_raw_obs)
            obs = np.stack(
                [
                    augment_observation(raw_obs[index], ats_features[index])
                    for index in range(config.num_envs)
                ]
            )
            normalization_rows.append(obs.copy())
            normalized_obs = obs_rms.normalize(obs)

        with torch.no_grad():
            advantages, returns = generalized_advantage_estimate(
                rewards_buffer,
                values_buffer,
                next_values_buffer,
                terminated_buffer,
                ends_buffer,
                config.gamma,
                config.gae_lambda,
            )

        b_obs = obs_buffer.reshape((-1, obs.shape[1]))
        b_logprobs = logprobs_buffer.reshape(-1)
        b_actions = actions_buffer.reshape((-1, action_dim))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buffer.reshape(-1)

        batch_indices = np.arange(config.batch_size)
        clipfracs = []
        for _ in range(config.update_epochs):
            np.random.shuffle(batch_indices)
            for start in range(0, config.batch_size, config.minibatch_size):
                end = start + config.minibatch_size
                mb_indices = batch_indices[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_indices], b_actions[mb_indices]
                )
                logratio = newlogprob - b_logprobs[mb_indices]
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > config.clip_coef).float().mean().item()
                    )
                mb_advantages = b_advantages[mb_indices]
                if config.normalize_advantages:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std(unbiased=False) + 1e-8
                    )
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - config.clip_coef, 1 + config.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_indices]) ** 2
                v_clipped = b_values[mb_indices] + torch.clamp(
                    newvalue - b_values[mb_indices],
                    -config.clip_coef,
                    config.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_indices]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                entropy_loss = entropy.mean()
                loss = (
                    pg_loss - config.ent_coef * entropy_loss + config.vf_coef * v_loss
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

            if (
                config.target_kl is not None
                and float(approx_kl.item()) > config.target_kl
            ):
                break

        obs_rms.update(np.concatenate(normalization_rows, axis=0))
        normalized_obs = obs_rms.normalize(obs)
        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        metric_rows.append(
            {
                "global_step": global_step,
                "update": update,
                "env_index": "all",
                "episode_return": "",
                "episode_length": "",
                "use_ats": config.resolved_feature_mode == "ats",
                "feature_mode": config.resolved_feature_mode,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "value_loss": float(v_loss.item()),
                "policy_loss": float(pg_loss.item()),
                "entropy": float(entropy_loss.item()),
                "old_approx_kl": float(old_approx_kl.item()),
                "approx_kl": float(approx_kl.item()),
                "clipfrac": float(np.mean(clipfracs)),
                "explained_variance": float(explained_var),
                "sps": int(global_step / max(time.time() - start_time, 1e-9)),
            }
        )

    resources.close()
    evaluation_rows = evaluate_ppo(agent, obs_rms, config, scales, device)
    write_metrics(output_dir / "ppo_evaluation.csv", evaluation_rows)
    write_metrics(output_dir / "ppo_metrics.csv", metric_rows)
    torch.save(
        {
            "model_state_dict": agent.state_dict(),
            "config": json_value(config),
            "obs_mean": torch.as_tensor(obs_rms.mean),
            "obs_var": torch.as_tensor(obs_rms.var),
            "obs_count": obs_rms.count,
            "observation_scales": torch.as_tensor(scales),
            "feature_names": list(encoders[0].feature_names),
            "raw_observation_dim": raw_dim,
        },
        output_dir / "deep_ats_ppo.pt",
    )
    complete_run(
        output_dir,
        actual_timesteps=global_step,
        parameter_count=sum(parameter.numel() for parameter in agent.parameters()),
        feature_mode=config.resolved_feature_mode,
        evaluation_seed_namespace="ppo_evaluation",
        active_observation_timing=False,
    )
    return output_dir


def evaluate_ppo(
    agent, normalizer, config: PPOConfig, scales, device
) -> list[dict[str, object]]:
    """Evaluate a frozen deterministic policy on independent seeds without updating normalization."""
    rows = []
    agent.eval()
    for episode in range(config.evaluation_episodes):
        seed = stable_seed(config.seed, "ppo_evaluation", episode)
        spec = RWRLSpec(
            config.rwrl_domain, config.rwrl_task, config.rwrl_combined_challenge
        )
        env = make_continuous_env(config.env_id, seed, spec)
        try:
            observation, _ = env.reset(seed=seed)
            raw = np.asarray(observation, dtype=np.float32).reshape(-1)
            encoder = make_encoder(raw.size, config, observation_scale=scales)
            features = encoder.reset(raw)
            total_return = 0.0
            length = 0
            while True:
                augmented = augment_observation(raw, features)
                with torch.no_grad():
                    inputs = torch.as_tensor(
                        normalizer.normalize(augmented)[None], device=device
                    )
                    action = (
                        agent.actor_mean(inputs)
                        .cpu()
                        .numpy()[0]
                        .reshape(env.action_space.shape)
                    )
                observation, reward, terminated, truncated, _ = env.step(
                    np.clip(action, env.action_space.low, env.action_space.high)
                )
                raw = np.asarray(observation, dtype=np.float32).reshape(-1)
                features = encoder.update(raw)
                total_return += float(reward)
                length += 1
                if terminated or truncated:
                    break
                if length >= config.evaluation_max_steps:
                    raise RuntimeError(
                        "Evaluation exceeded evaluation_max_steps without an episode boundary."
                    )
            rows.append(
                {
                    "episode": episode,
                    "seed": seed,
                    "episode_return": total_return,
                    "episode_length": length,
                    "feature_mode": config.resolved_feature_mode,
                    "policy": "frozen_deterministic",
                }
            )
        finally:
            env.close()
    agent.train()
    return rows


def write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
