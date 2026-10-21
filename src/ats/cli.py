"""Command-line interface for running simulations and figures."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ats.config import AgentConfig, GridConfig, main_config, smoke_config
from ats.experiments import run_experiment
from ats.figures import generate_all_figures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Active temporal sampling experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="Run the small validation experiment.")
    smoke.add_argument("--run-name", default="smoke")
    smoke.add_argument("--output-root", default="outputs")
    smoke.add_argument("--grid-size", type=int, default=25)
    smoke.add_argument("--seed", type=int, default=20260521)
    smoke.add_argument("--time-budget", type=int, default=None)
    add_agent_arguments(smoke)

    run = subparsers.add_parser("run", help="Run the main experiment.")
    run.add_argument("--run-name", default="main")
    run.add_argument("--output-root", default="outputs")
    run.add_argument("--episodes", type=int, default=96)
    run.add_argument("--steps", type=int, default=80)
    run.add_argument("--grid-size", type=int, default=25)
    run.add_argument("--seed", type=int, default=20260521)
    run.add_argument("--time-budget", type=int, default=None)
    run.add_argument("--skip-ablation", action="store_true")
    run.add_argument("--model-comparison-episodes", type=int, default=8)
    run.add_argument("--competitive-episodes", type=int, default=8)
    run.add_argument("--skip-model-comparison", action="store_true")
    run.add_argument("--skip-competitive-comparison", action="store_true")
    add_agent_arguments(run)

    figures = subparsers.add_parser(
        "figures", help="Regenerate figures for an existing run."
    )
    figures.add_argument("--run-name", default="main")
    figures.add_argument("--output-root", default="outputs")
    return parser


def add_agent_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wait-cost", type=float, default=0.05)
    parser.add_argument(
        "--observation-cost",
        "--observation-rate-cost",
        dest="observation_rate_cost",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--cost-mode",
        choices=("per_observation", "legacy_rate"),
        default="per_observation",
    )
    parser.add_argument("--eig-samples", type=int, default=128)
    parser.add_argument("--calibration-cycles", type=int, default=0)
    parser.add_argument(
        "--information-objective", choices=("entropy", "log_ratio"), default="entropy"
    )


def agent_config_from_args(args: argparse.Namespace) -> AgentConfig:
    return AgentConfig(
        wait_cost=args.wait_cost,
        observation_rate_cost=args.observation_rate_cost,
        eig_samples=args.eig_samples,
        calibration_cycles=args.calibration_cycles,
        information_objective=args.information_objective,
        cost_mode=args.cost_mode,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        config = smoke_config(args.run_name)
        config = replace(
            config,
            output_root=Path(args.output_root),
            seed=args.seed,
            time_budget=args.time_budget,
            grid=replace(config.grid, size=args.grid_size),
            agent=agent_config_from_args(args),
        )
        output_dir = run_experiment(config)
        generate_all_figures(output_dir)
        print(f"wrote {output_dir}")
        return

    if args.command == "run":
        config = main_config(args.run_name)
        config = replace(
            config,
            output_root=Path(args.output_root),
            seed=args.seed,
            time_budget=args.time_budget,
            episodes=args.episodes,
            steps=args.steps,
            grid=GridConfig(size=args.grid_size),
            agent=agent_config_from_args(args),
            include_ablation=not args.skip_ablation,
            include_model_comparison=not args.skip_model_comparison,
            model_comparison_episodes=args.model_comparison_episodes,
            include_competitive_comparison=not args.skip_competitive_comparison,
            competitive_episodes=args.competitive_episodes,
        )
        output_dir = run_experiment(config)
        generate_all_figures(output_dir)
        print(f"wrote {output_dir}")
        return

    if args.command == "figures":
        output_dir = Path(args.output_root) / args.run_name
        fig_dir = generate_all_figures(output_dir)
        print(f"wrote {fig_dir}")


if __name__ == "__main__":
    main()
