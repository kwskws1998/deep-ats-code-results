"""Command-line interface for Deep Active Temporal Sampling Planner."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from deep_ats_planner.config import GridConfig, deep_smoke_config, deep_train_config
from deep_ats_planner.deep_ppo import PPOConfig, train_ppo
from deep_ats_planner.rwrl_reference import (
    best_reference_rows,
    write_rwrl_reference_csv,
)
from deep_ats_planner.training import run_deep_planner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deep Active Temporal Sampling Planner."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser(
        "smoke", help="Run a tiny Deep ATS Planner validation run."
    )
    add_train_arguments(
        smoke, default_run_name="deep_ats_smoke", default_episodes=4, default_steps=24
    )

    train = subparsers.add_parser("train", help="Train the Deep ATS Planner.")
    add_train_arguments(
        train,
        default_run_name="deep_ats_planner",
        default_episodes=64,
        default_steps=80,
    )

    ppo_smoke = subparsers.add_parser(
        "ppo-smoke", help="Run a tiny Deep ATS-PPO smoke test."
    )
    add_ppo_arguments(
        ppo_smoke, default_run_name="deep_ats_ppo_smoke", default_timesteps=512
    )

    ppo_train = subparsers.add_parser("ppo-train", help="Train PPO or Deep ATS-PPO.")
    add_ppo_arguments(
        ppo_train, default_run_name="deep_ats_ppo", default_timesteps=20000
    )

    baselines = subparsers.add_parser(
        "rwrl-baselines", help="Write official RWRL reference baselines."
    )
    baselines.add_argument("--output", default="outputs/rwrl_reference_baselines.csv")
    baselines.add_argument("--best-only", action="store_true")

    return parser


def add_train_arguments(
    parser: argparse.ArgumentParser,
    default_run_name: str,
    default_episodes: int,
    default_steps: int,
) -> None:
    parser.add_argument("--run-name", default=default_run_name)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--episodes", type=int, default=default_episodes)
    parser.add_argument("--steps", type=int, default=default_steps)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument(
        "--discount-basis",
        choices=("physical_time", "decision"),
        default="physical_time",
    )
    parser.add_argument("--no-belief-features", action="store_true")
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--wait-cost", type=float, default=None)
    parser.add_argument(
        "--observation-cost",
        "--observation-rate-cost",
        dest="observation_rate_cost",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--cost-mode",
        choices=("per_observation", "legacy_rate"),
        default="per_observation",
    )


def add_ppo_arguments(
    parser: argparse.ArgumentParser,
    default_run_name: str,
    default_timesteps: int,
) -> None:
    parser.add_argument("--run-name", default=default_run_name)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--env-id", default="Pendulum-v1")
    parser.add_argument("--total-timesteps", type=int, default=default_timesteps)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument(
        "--feature-mode", choices=("ats", "none", "history", "ema"), default=None
    )
    parser.add_argument("--history-length", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--ats-projection-count", type=int, default=8)
    parser.add_argument("--no-ats", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rwrl-domain", default="walker")
    parser.add_argument("--rwrl-task", default="realworld_walk")
    parser.add_argument(
        "--rwrl-combined-challenge", default="easy", choices=("easy", "medium", "hard")
    )


def config_from_args(args: argparse.Namespace, smoke: bool):
    config = (
        deep_smoke_config(args.run_name) if smoke else deep_train_config(args.run_name)
    )
    agent = replace(config.agent, cost_mode=args.cost_mode)
    if args.wait_cost is not None or args.observation_rate_cost is not None:
        agent = replace(
            agent,
            wait_cost=(
                config.agent.wait_cost if args.wait_cost is None else args.wait_cost
            ),
            observation_rate_cost=(
                config.agent.observation_rate_cost
                if args.observation_rate_cost is None
                else args.observation_rate_cost
            ),
        )
    return replace(
        config,
        output_root=Path(args.output_root),
        episodes=args.episodes,
        seed=args.seed,
        evaluation_episodes=args.eval_episodes,
        discount_basis=args.discount_basis,
        use_belief_features=not args.no_belief_features,
        steps=args.steps,
        grid=replace(
            config.grid,
            size=config.grid.size if args.grid_size is None else args.grid_size,
        ),
        agent=agent,
        hidden_size=(
            config.hidden_size if args.hidden_size is None else args.hidden_size
        ),
        batch_size=config.batch_size if args.batch_size is None else args.batch_size,
        learning_rate=(
            config.learning_rate if args.learning_rate is None else args.learning_rate
        ),
    )


def ppo_config_from_args(args: argparse.Namespace) -> PPOConfig:
    return PPOConfig(
        run_name=args.run_name,
        output_root=Path(args.output_root),
        env_id=args.env_id,
        seed=args.seed,
        evaluation_episodes=args.eval_episodes,
        feature_mode=args.feature_mode,
        history_length=args.history_length,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        ats_projection_count=args.ats_projection_count,
        use_ats=not args.no_ats,
        device=args.device,
        rwrl_domain=args.rwrl_domain,
        rwrl_task=args.rwrl_task,
        rwrl_combined_challenge=args.rwrl_combined_challenge,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        output_dir = run_deep_planner(config_from_args(args, smoke=True))
        print(f"wrote {output_dir}")
        return
    if args.command == "train":
        output_dir = run_deep_planner(config_from_args(args, smoke=False))
        print(f"wrote {output_dir}")
        return
    if args.command == "ppo-smoke":
        output_dir = train_ppo(ppo_config_from_args(args))
        print(f"wrote {output_dir}")
        return
    if args.command == "ppo-train":
        output_dir = train_ppo(ppo_config_from_args(args))
        print(f"wrote {output_dir}")
        return
    if args.command == "rwrl-baselines":
        rows = best_reference_rows() if args.best_only else None
        output = Path(args.output)
        if rows is None:
            write_rwrl_reference_csv(output)
        else:
            write_rwrl_reference_csv(output, rows)
        print(f"wrote {output}")
        return


if __name__ == "__main__":
    main()
