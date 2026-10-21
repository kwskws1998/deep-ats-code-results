"""Run independent-seed validation of the revised ATS, DQN, and PPO code paths.

These short runs validate implementation and comparison machinery. Their budgets
do not establish converged control performance or priority over published work.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from ats.artifacts import complete_run, prepare_run
from ats.beliefs import GridBelief
from ats.config import AgentConfig, GridConfig, SimulationConfig
from ats.experiments import run_experiment, write_csv
from ats.figures import generate_all_figures
from ats.policies import active_temporal_sampling_scores
from deep_ats_planner.config import DeepPlannerConfig
from deep_ats_planner.deep_ppo import PPOConfig, train_ppo
from deep_ats_planner.training import run_deep_planner


def csv_rows(path: Path):
    """Read only newly produced validation artifacts."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def acquisition_diagnostic(root: Path) -> None:
    """Report action stability over estimator seeds at a fixed belief, without selecting favorable states."""
    v, s = GridConfig(size=17).make_grids()
    belief = GridBelief(v, s)
    belief.reset(0.0)
    actions = (1, 2, 4, 8)
    rows = []
    for samples in (8, 128, 4096):
        repetitions = 128 if samples < 4096 else 8
        for seed in range(repetitions):
            scores = active_temporal_sampling_scores(
                belief,
                actions,
                0.05,
                0.08,
                samples,
                "entropy",
                np.random.default_rng(seed),
            )
            rows.append(
                {
                    "samples": samples,
                    "estimator_seed": seed,
                    "chosen_dt": max(scores, key=scores.get),
                    **{f"score_dt_{dt}": scores[dt] for dt in actions},
                }
            )
    write_csv(root / "acquisition_stability.csv", rows, rows[0].keys())


def main() -> None:
    """Persist a fresh suite with equal budgets, disjoint evaluation seeds, and source hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/revision_validation_20261021")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    parser.add_argument(
        "--section", choices=("all", "core", "ppo", "dqn"), default="all"
    )
    parser.add_argument("--core-episodes", type=int, default=8)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--ppo-timesteps", type=int, default=2048)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Training seeds must be distinct.")
    torch.set_num_threads(1)
    prepare_run(args.output_root, vars(args), "revision_validation_suite")
    summary = []
    if args.section in {"all", "core"}:
        acquisition_diagnostic(args.output_root)
        for seed in args.seeds:
            for budget in ("observations", "time"):
                config = SimulationConfig(
                    output_root=args.output_root,
                    run_name=f"core_{budget}_seed_{seed}",
                    seed=seed,
                    episodes=args.core_episodes,
                    steps=args.steps,
                    time_budget=4 * args.steps if budget == "time" else None,
                    grid=GridConfig(size=17),
                    agent=AgentConfig(eig_samples=128),
                    include_ablation=False,
                    model_comparison_episodes=2,
                    competitive_episodes=2,
                )
                directory = run_experiment(config)
                if args.figures:
                    generate_all_figures(directory)
                for row in csv_rows(directory / "summary.csv"):
                    summary.append(
                        {"family": "core", "seed": seed, "budget": budget, **row}
                    )
                print(f"completed {directory}", flush=True)
    if args.section in {"all", "dqn"}:
        for seed in args.seeds:
            for use_belief in (True, False):
                config = DeepPlannerConfig(
                    output_root=args.output_root,
                    run_name=f"dqn_{'belief' if use_belief else 'raw'}_seed_{seed}",
                    seed=seed,
                    episodes=16,
                    steps=args.steps,
                    grid=GridConfig(size=9),
                    hidden_size=32,
                    batch_size=32,
                    warmup_steps=32,
                    evaluation_episodes=3,
                    use_belief_features=use_belief,
                )
                directory = run_deep_planner(config)
                for row in csv_rows(directory / "deep_evaluation.csv"):
                    summary.append(
                        {
                            "family": "dqn",
                            "training_seed": seed,
                            "use_belief_features": use_belief,
                            **row,
                        }
                    )
                print(f"completed {directory}", flush=True)
    if args.section in {"all", "ppo"}:
        for seed in args.seeds:
            for mode in ("ats", "none", "history", "ema"):
                config = PPOConfig(
                    output_root=args.output_root,
                    run_name=f"ppo_{mode}_seed_{seed}",
                    seed=seed,
                    total_timesteps=args.ppo_timesteps,
                    num_envs=2,
                    num_steps=64,
                    hidden_size=64,
                    ats_projection_count=4,
                    evaluation_episodes=3,
                    feature_mode=mode,
                    device="cpu",
                )
                directory = train_ppo(config)
                for row in csv_rows(directory / "ppo_evaluation.csv"):
                    summary.append({"family": "ppo", "training_seed": seed, **row})
                print(f"completed {directory}", flush=True)
    write_csv(
        args.output_root / "suite_metrics.csv",
        summary,
        sorted({key for row in summary for key in row}),
    )
    complete_run(
        args.output_root,
        interpretation="short implementation validation; not a convergence or superiority study",
    )
    print(args.output_root / "suite_metrics.csv", flush=True)


if __name__ == "__main__":
    main()
