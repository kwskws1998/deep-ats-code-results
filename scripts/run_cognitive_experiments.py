"""Run and audit a prespecified cognitive-only simulation suite without training neural policies."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ats.artifacts import complete_run, prepare_run, verify_run
from ats.config import AgentConfig, GridConfig, SimulationConfig
from ats.experiments import run_experiment, stable_seed, write_csv
from ats.figures import generate_all_figures


def execute_seed(root: Path, seed: int, protocol: dict) -> list[str]:
    """Run both budget regimes for one independent seed with frozen model settings."""
    directories = []
    for budget in ("observations", "time"):
        config = SimulationConfig(
            output_root=root,
            run_name=f"{budget}_seed_{seed}",
            seed=seed,
            episodes=protocol["episodes_per_condition_per_seed"],
            steps=protocol["observation_budget"],
            time_budget=protocol["physical_time_budget"] if budget == "time" else None,
            grid=GridConfig(size=protocol["grid_size"]),
            agent=AgentConfig(eig_samples=protocol["eig_samples"]),
            include_ablation=budget == "observations",
            ablation_episodes=protocol["sensitivity_episodes_per_condition_per_seed"],
            ablation_wait_costs=(0.0, 0.01, 0.05),
            model_comparison_episodes=protocol["model_episodes_per_condition_per_seed"],
            competitive_episodes=protocol[
                "competitive_episodes_per_condition_per_seed"
            ],
        )
        print(f"starting {config.run_name}", flush=True)
        directory = run_experiment(config)
        generate_all_figures(directory)
        failures = verify_run(directory, check_current_source=True)
        if failures:
            raise RuntimeError(f"Integrity failure for {directory}: {failures}")
        directories.append(str(directory))
        print(f"completed {config.run_name}", flush=True)
    return directories


def reduce_episodes(
    path: Path, family: str, budget: str, seed: int
) -> tuple[list, dict]:
    """Reduce raw forecasts to independent episode units and check shared paths and budgets."""
    name_field = "agent" if family == "sampling" else "model"
    groups = {}
    shared_observations = {}
    checked_rows = 0
    active_intervals = defaultdict(int)
    active_fixed_mismatches = 0
    for row in csv.DictReader(path.open(newline="")):
        condition, name, episode = (
            row["condition"],
            row[name_field],
            int(row["episode"]),
        )
        key = condition, name, episode
        if key not in groups:
            groups[key] = {"count": 0, "mse": 0.0, "nll": 0.0, "dt": 0.0}
        group = groups[key]
        for field in (
            "prediction_before",
            "predictive_variance",
            "squared_error",
            "negative_log_likelihood",
        ):
            if not np.isfinite(float(row[field])):
                raise FloatingPointError(f"Nonfinite {field} in {path}, {key}")
        if float(row["predictive_variance"]) <= 0:
            raise FloatingPointError(f"Nonpositive forecast variance in {path}, {key}")
        time = int(row["time"])
        dt = int(row["dt"])
        if dt not in (1, 2, 4, 8) or time != group.get("time", 0) + dt:
            raise ValueError(f"Invalid interval sequence in {path}, {key}")
        physical_key = condition, episode, time
        physical_values = float(row["latent_state"]), float(row["observation"])
        if (
            physical_key in shared_observations
            and shared_observations[physical_key] != physical_values
        ):
            raise ValueError(
                f"Underlying paths differ between agents: {path}, {physical_key}"
            )
        shared_observations[physical_key] = physical_values
        group["count"] += 1
        group["mse"] += float(row["squared_error"])
        group["nll"] += float(row["negative_log_likelihood"])
        group["dt"] += dt
        group["time"] = time
        group["last"] = row
        if family == "sampling" and name == "active_temporal":
            active_intervals[dt] += 1
        checked_rows += 1
    records = []
    for (condition, name, episode), group in sorted(groups.items()):
        row = group["last"]
        record = {
            "family": family,
            "budget": budget,
            "seed": seed,
            "condition": condition,
            "name": name,
            "episode": episode,
            "observations": group["count"],
            "elapsed_time": group["time"],
            "observation_forecast_mse": group["mse"] / group["count"],
            "predictive_nll": group["nll"] / group["count"],
            "mean_dt": group["dt"] / group["count"],
            "final_ratio_log_abs_error": float(row["ratio_abs_error"]),
        }
        if family in ("sampling", "competitive"):
            record.update(
                time_tracking_mse=float(row["cumulative_tracking_squared_error"])
                / group["time"],
                total_sampling_cost=float(row["cumulative_sampling_cost"]),
            )
        if family == "sampling":
            record.update(
                final_volatility_log_abs_error=float(row["volatility_log_abs_error"]),
                final_stochasticity_log_abs_error=float(
                    row["stochasticity_log_abs_error"]
                ),
                volatility_coverage_95=float(row["volatility_covered_95"]),
                stochasticity_coverage_95=float(row["stochasticity_covered_95"]),
            )
            if name == "active_temporal":
                reference = groups[(condition, "fixed_short", episode)]
                active_fixed_mismatches += any(
                    group[field] != reference[field]
                    for field in ("count", "time", "mse", "nll")
                )
        records.append(record)
    audit = {
        "forecast_rows": checked_rows,
        "episode_model_pairs": len(records),
        "shared_physical_points": len(shared_observations),
        "active_interval_counts": dict(active_intervals),
        "active_fixed_short_nonidentical_episodes": active_fixed_mismatches,
    }
    return records, audit


def aggregate(records: list[dict], by_condition: bool) -> list[dict]:
    """Average episode metrics with equal weight for the prespecified condition and seed blocks."""
    identifiers = ("family", "budget", "name") + (
        ("condition",) if by_condition else ()
    )
    groups = defaultdict(list)
    for record in records:
        groups[tuple(record[key] for key in identifiers)].append(record)
    excluded = {"family", "budget", "seed", "condition", "name", "episode"}
    output = []
    for key, group in sorted(groups.items()):
        row = dict(zip(identifiers, key))
        row.update(episodes=len(group), seeds=len({item["seed"] for item in group}))
        for metric in sorted(set().union(*(item.keys() for item in group)) - excluded):
            blocks = defaultdict(list)
            for item in group:
                value = item.get(metric, float("nan"))
                if np.isfinite(value):
                    blocks[(item["seed"], item["condition"])].append(value)
            means = [float(np.mean(values)) for values in blocks.values()]
            row[metric] = float(np.mean(means)) if means else float("nan")
        output.append(row)
    return output


def paired_contrasts(records: list[dict], replicates: int = 5000) -> list[dict]:
    """Bootstrap paired independent trajectories within fixed seed and condition strata; intervals are exploratory."""
    references = {
        "sampling": "active_temporal",
        "fixed_sequence": "joint_grid",
        "competitive": "ats_joint_active",
    }
    metrics = (
        "final_ratio_log_abs_error",
        "observation_forecast_mse",
        "predictive_nll",
        "time_tracking_mse",
        "total_sampling_cost",
    )
    indexed = {
        (
            r["family"],
            r["budget"],
            r["name"],
            r["seed"],
            r["condition"],
            r["episode"],
        ): r
        for r in records
    }
    output = []
    for family, budget in sorted({(r["family"], r["budget"]) for r in records}):
        reference = references[family]
        names = sorted(
            {
                r["name"]
                for r in records
                if r["family"] == family and r["budget"] == budget
            }
            - {reference}
        )
        for baseline in names:
            matched = []
            for key, row in indexed.items():
                if key[:3] == (family, budget, reference):
                    matched.append((row, indexed[(family, budget, baseline, *key[3:])]))
            for metric in metrics:
                blocks = defaultdict(list)
                for left, right in matched:
                    if (
                        metric in left
                        and metric in right
                        and np.isfinite([left[metric], right[metric]]).all()
                    ):
                        blocks[(left["seed"], left["condition"])].append(
                            left[metric] - right[metric]
                        )
                if not blocks:
                    continue
                rng = np.random.default_rng(
                    stable_seed(8731, family, budget, baseline, metric)
                )
                draws = []
                for values in blocks.values():
                    array = np.asarray(values)
                    draws.append(
                        array[
                            rng.integers(0, len(array), size=(replicates, len(array)))
                        ].mean(axis=1)
                    )
                low, high = np.quantile(np.mean(draws, axis=0), (0.025, 0.975))
                output.append(
                    {
                        "family": family,
                        "budget": budget,
                        "reference": reference,
                        "baseline": baseline,
                        "metric": metric,
                        "reference_minus_baseline": float(
                            np.mean([np.mean(v) for v in blocks.values()])
                        ),
                        "ci_low": float(low),
                        "ci_high": float(high),
                        "paired_episodes": sum(map(len, blocks.values())),
                        "bootstrap_replicates": replicates,
                        "strata": "fixed seed x condition",
                        "multiplicity_adjustment": "none",
                    }
                )
    return output


def save_table(path: Path, rows: list[dict]) -> None:
    """Persist heterogeneous results without requiring an additional dataframe dependency."""
    write_csv(path, rows, sorted(set().union(*(row.keys() for row in rows))))


def plot_overview(root: Path, summary: list[dict], contrasts: list[dict]) -> None:
    """Plot error, latent tracking, and observation cost for both distinct budget regimes."""
    import matplotlib.pyplot as plt

    names = (
        "active_temporal",
        "fixed_short",
        "fixed_dt_2",
        "fixed_dt_4",
        "fixed_long",
        "passive_grid",
        "random_interval",
        "yoked_independent",
    )
    labels = (
        "ATS",
        "Fixed 1",
        "Fixed 2",
        "Fixed 4",
        "Fixed 8",
        "Cycle",
        "Random",
        "Indep. yoke",
    )
    metrics = ("final_ratio_log_abs_error", "time_tracking_mse", "total_sampling_cost")
    titles = (
        "Final |log(q/r) error|",
        "Latent tracking MSE / time unit",
        "Total sampling cost",
    )
    lookup = {(r["budget"], r["name"]): r for r in summary if r["family"] == "sampling"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for i, budget in enumerate(("observations", "time")):
        for j, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i, j]
            values = [lookup[(budget, name)][metric] for name in names]
            ax.bar(
                np.arange(len(names)),
                values,
                color=["#b34731"] + ["#527d99"] * (len(names) - 1),
            )
            ax.set_xticks(np.arange(len(names)), labels, rotation=35, ha="right")
            ax.set_title(title + " (lower is better)")
            ax.set_ylabel(
                "80 observations" if budget == "observations" else "320 time units"
            )
            ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Cognitive models only: 3 seeds x 32 episodes x 4 conditions\nEpisode means; identical latent paths within each matched comparison"
    )
    fig.savefig(root / "cognitive_overview.png", dpi=180)
    plt.close(fig)
    contrasts = [
        r
        for r in contrasts
        if r["family"] == "sampling"
        and r["budget"] == "observations"
        and r["metric"] == "final_ratio_log_abs_error"
    ]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for i, row in enumerate(contrasts):
        value = row["reference_minus_baseline"]
        ax.errorbar(
            value,
            i,
            xerr=[[max(0, value - row["ci_low"])], [max(0, row["ci_high"] - value)]],
            fmt="o",
            color="#b34731",
        )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(range(len(contrasts)), [r["baseline"] for r in contrasts])
    ax.set_xlabel(
        "ATS minus baseline: final |log(q/r) error|\nNegative favors ATS; 95% paired bootstrap intervals, no multiplicity adjustment"
    )
    ax.set_title("Observation budget: exploratory paired contrasts")
    fig.savefig(root / "paired_ratio_contrasts.png", dpi=180)
    plt.close(fig)


def analyze(root: Path, protocol: dict) -> None:
    """Independently audit raw outputs, aggregate episode-level results, and retain null outcomes."""
    records, audits, sensitivity = [], {}, []
    files = {
        "sampling": "metrics.csv",
        "fixed_sequence": "model_comparison_metrics.csv",
        "competitive": "competitive_model_metrics.csv",
    }
    for seed in protocol["seeds"]:
        for budget in ("observations", "time"):
            directory = root / f"{budget}_seed_{seed}"
            for family, filename in files.items():
                reduced, audit = reduce_episodes(
                    directory / filename, family, budget, seed
                )
                expected_count = protocol["observation_budget"]
                for row in reduced:
                    if (
                        budget == "time"
                        and row["elapsed_time"] != protocol["physical_time_budget"]
                    ):
                        raise ValueError(
                            "An episode did not exhaust its exact physical-time budget."
                        )
                    if (
                        budget == "observations"
                        and row["observations"] != expected_count
                    ):
                        raise ValueError(
                            "An episode did not exhaust its observation budget."
                        )
                records.extend(reduced)
                audits[f"{directory.name}/{family}"] = audit
            if budget == "observations":
                groups = defaultdict(list)
                for row in csv.DictReader(
                    (directory / "wait_cost_ablation.csv").open(newline="")
                ):
                    groups[
                        (float(row["wait_cost"]), row["condition"], int(row["episode"]))
                    ].append(row)
                for (cost, condition, episode), rows in groups.items():
                    sensitivity.append(
                        {
                            "seed": seed,
                            "condition": condition,
                            "episode": episode,
                            "wait_cost": cost,
                            "mean_dt": float(np.mean([int(row["dt"]) for row in rows])),
                            "fraction_dt_1": float(
                                np.mean([int(row["dt"]) == 1 for row in rows])
                            ),
                            "final_ratio_log_abs_error": float(
                                rows[-1]["ratio_abs_error"]
                            ),
                            "observation_forecast_mse": float(
                                np.mean([float(row["squared_error"]) for row in rows])
                            ),
                        }
                    )
    summary = aggregate(records, by_condition=False)
    contrasts = paired_contrasts(records)
    save_table(root / "episode_metrics.csv", records)
    save_table(root / "aggregate_metrics.csv", summary)
    save_table(root / "condition_metrics.csv", aggregate(records, by_condition=True))
    save_table(root / "paired_contrasts.csv", contrasts)
    save_table(root / "wait_cost_sensitivity.csv", sensitivity)
    audit_report = {
        "passed": True,
        "deep_learning_runs": 0,
        "seeds": protocol["seeds"],
        "completed_runs": 2 * len(protocol["seeds"]),
        "forecast_rows_checked": sum(r["forecast_rows"] for r in audits.values()),
        "episode_model_pairs_checked": len(records),
        "all_overlapping_observations_and_latent_states_bit_identical": True,
        "all_forecast_fields_finite_and_variances_positive": True,
        "all_budgets_exact": True,
        "details": audits,
    }
    (root / "data_checks.json").write_text(json.dumps(audit_report, indent=2) + "\n")
    plot_overview(root, summary, contrasts)


def main() -> None:
    """Freeze the protocol before starting workers and fingerprint every result after successful audits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/cognitive_experiments_20261021"),
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    protocol = {
        "scope": "cognitive simulation only; no DQN/PPO or neural training",
        "seeds": [101, 202, 303],
        "episodes_per_condition_per_seed": 32,
        "conditions": {
            "low_v_low_s": [0.25, 1.0],
            "low_v_high_s": [0.25, 9.0],
            "high_v_low_s": [4.0, 1.0],
            "high_v_high_s": [4.0, 9.0],
        },
        "observation_budget": 80,
        "physical_time_budget": 320,
        "grid_size": 25,
        "eig_samples": 128,
        "model_episodes_per_condition_per_seed": 16,
        "competitive_episodes_per_condition_per_seed": 16,
        "sensitivity_episodes_per_condition_per_seed": 8,
        "wait_cost_sensitivity": [0.0, 0.01, 0.05],
        "model_settings": "fixed before evaluation; no parameter fitting or outcome-based selection",
        "primary_metric": "final absolute log(q/r) error; report latent tracking and cost alongside it",
        "inference_only_comparison": "all ten perceptual models receive identical cyclic (1,2,4,8) observations",
        "uncertainty": "5000 paired episode bootstrap replicates within fixed condition x seed strata; exploratory, no multiple-comparison correction",
        "limits": [
            "stationary scalar Gaussian simulation, no human fit",
            "finite-grid uncertainty intervals are diagnostics",
            "one-step acquisition is not an optimized long-horizon sensing policy",
            "matched_histogram is retrospective; independent donor calibration cost is excluded",
            "default HGF and heuristic competitor settings are not fitted to this task",
        ],
    }
    root = args.output_root
    prepare_run(root, protocol, "cognitive_experiment_suite")
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    try:
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(protocol["seeds"]))
        ) as pool:
            futures = [
                pool.submit(execute_seed, root, seed, protocol)
                for seed in protocol["seeds"]
            ]
            for future in as_completed(futures):
                future.result()
        analyze(root, protocol)
        for manifest in root.glob("*/manifest.json"):
            failures = verify_run(manifest.parent, check_current_source=True)
            if failures:
                raise RuntimeError(failures)
        complete_run(
            root,
            interpretation="fresh cognitive-only controlled simulations; no empirical human or novelty claim",
        )
        failures = verify_run(root, check_current_source=True)
        if failures:
            raise RuntimeError(failures)
    except Exception as error:
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise
    print(f"completed and verified {root}", flush=True)


if __name__ == "__main__":
    main()
