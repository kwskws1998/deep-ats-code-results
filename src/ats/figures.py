"""Figure generation for active temporal sampling experiments."""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ats.artifacts import register_derived_figures


AGENT_LABELS = {
    "fixed_short": "Fixed short",
    "fixed_long": "Fixed long",
    "fixed_dt_2": "Fixed dt 2",
    "fixed_dt_4": "Fixed dt 4",
    "matched_histogram": "Retrospective histogram match",
    "yoked_independent": "Independent yoked schedule",
    "random_interval": "Random",
    "passive_grid": "Passive grid",
    "active_temporal": "Active temporal",
}

MODEL_LABELS = {
    "oracle_kalman": "Oracle Kalman",
    "joint_grid": "Joint grid",
    "fixed_kalman": "Fixed Kalman",
    "pyhgf_full_3level": "PyHGF full 3-level",
    "pyhgf_full_3level_oracle_s": "PyHGF full 3-level oracle s",
    "hgf_style_volatility": "HGF-style volatility",
    "volatility_only_grid": "Volatility-only grid",
    "stochasticity_only_grid": "Stochasticity-only grid",
    "pearce_hall": "Pearce-Hall",
    "delta_rule": "Delta rule",
}

COMPETITIVE_MODEL_LABELS = {
    "ats_joint_active": "ATS joint active",
    "joint_fixed_short": "Joint fixed short",
    "joint_fixed_dt_2": "Joint fixed dt 2",
    "joint_fixed_dt_4": "Joint fixed dt 4",
    "joint_fixed_long": "Joint fixed long",
    "joint_passive_grid": "Joint passive grid",
    "kalman_fixed_active": "Kalman fixed active",
    "kalman_oracle_active": "Kalman oracle active",
    "pyhgf_full_active": "PyHGF full active",
    "pyhgf_full_oracle_s_active": "PyHGF full oracle-s active",
    "pearce_hall_active": "Pearce-Hall active",
    "delta_rule_active": "Delta rule active",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_fig_dir(output_dir: Path) -> Path:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def save_current(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_task_schematic(fig_dir: Path) -> None:
    rng = np.random.default_rng(11)
    dts = np.array([1, 1, 4, 1, 8, 2, 1, 4, 2, 8])
    times = np.cumsum(np.r_[0, dts])
    latent = [0.0]
    observations = [rng.normal(0.0, 1.0)]
    for dt in dts:
        latent.append(latent[-1] + rng.normal(0.0, np.sqrt(0.8 * dt)))
        observations.append(latent[-1] + rng.normal(0.0, 1.4))
    plt.figure(figsize=(8.0, 3.8))
    plt.plot(times, latent, color="#1f4e79", linewidth=2.4, label="latent state")
    plt.scatter(
        times, observations, color="#c43c39", s=34, label="noisy observation", zorder=3
    )
    for left, right, dt in zip(times[:-1], times[1:], dts):
        plt.axvspan(left, right, color="#d9e6f2", alpha=0.18 if dt <= 2 else 0.34)
        plt.text(
            (left + right) / 2,
            min(observations) - 2.0,
            f"dt={dt}",
            ha="center",
            fontsize=8,
        )
    plt.xlabel("time")
    plt.ylabel("state / observation")
    plt.title("Active temporal sampling task")
    plt.legend(frameon=False, ncols=2)
    save_current(fig_dir / "fig01_task_schematic.png")


def plot_example_posterior(output_dir: Path, fig_dir: Path) -> None:
    data = np.load(output_dir / "example_posterior.npz")
    posterior = data["posterior"]
    volatility_grid = data["volatility_grid"]
    stochasticity_grid = data["stochasticity_grid"]
    true_v = float(data["true_volatility"])
    true_s = float(data["true_stochasticity"])
    plt.figure(figsize=(5.3, 4.5))
    log_s, log_v = np.log10(stochasticity_grid), np.log10(volatility_grid)
    extent = [
        log_s[0] - (log_s[1] - log_s[0]) / 2,
        log_s[-1] + (log_s[-1] - log_s[-2]) / 2,
        log_v[0] - (log_v[1] - log_v[0]) / 2,
        log_v[-1] + (log_v[-1] - log_v[-2]) / 2,
    ]
    heatmap = plt.imshow(
        posterior,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
    )
    plt.scatter(
        np.log10(true_s),
        np.log10(true_v),
        color="white",
        edgecolor="black",
        s=70,
        label="true",
    )
    plt.xlabel("log10 stochasticity")
    plt.ylabel("log10 volatility")
    plt.title(
        "Prespecified episode posterior"
        if "selection_rule" in data
        else "Historical selected example"
    )
    plt.colorbar(heatmap, label="posterior mass")
    plt.legend(frameon=False)
    save_current(fig_dir / "fig02_example_posterior.png")


def plot_ratio_error(rows: list[dict[str, str]], fig_dir: Path) -> None:
    groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        groups[(row["agent"], int(row["step"]))].append(float(row["ratio_abs_error"]))
    plt.figure(figsize=(7.5, 4.4))
    for agent in sorted({row["agent"] for row in rows}):
        steps = sorted(step for (name, step) in groups if name == agent)
        means = [np.mean(groups[(agent, step)]) for step in steps]
        plt.plot(steps, means, linewidth=2.0, label=AGENT_LABELS[agent])
    plt.xlabel("step")
    plt.ylabel("|posterior log(v/s) - true log(v/s)|")
    plt.title("Noise-ratio inference error")
    plt.legend(frameon=False)
    save_current(fig_dir / "fig03_ratio_error.png")


def plot_prediction_mse(rows: list[dict[str, str]], fig_dir: Path) -> None:
    groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        groups[(row["agent"], int(row["step"]))].append(float(row["squared_error"]))
    plt.figure(figsize=(7.5, 4.4))
    for agent in sorted({row["agent"] for row in rows}):
        steps = sorted(step for (name, step) in groups if name == agent)
        mse = np.array([np.mean(groups[(agent, step)]) for step in steps])
        cumulative = np.cumsum(mse)
        plt.plot(steps, cumulative, linewidth=2.0, label=AGENT_LABELS[agent])
    plt.xlabel("step")
    plt.ylabel("cumulative mean squared prediction error")
    plt.title("Observation forecast loss at each agent’s chosen horizons")
    plt.legend(frameon=False)
    save_current(fig_dir / "fig04_prediction_mse.png")


def plot_dt_choices(rows: list[dict[str, str]], fig_dir: Path) -> None:
    active_rows = [row for row in rows if row["agent"] == "active_temporal"]
    conditions = sorted({row["condition"] for row in active_rows})
    action_space = [1, 2, 4, 8]
    counts = np.zeros((len(conditions), len(action_space)))
    for i, condition in enumerate(conditions):
        condition_rows = [row for row in active_rows if row["condition"] == condition]
        total = max(len(condition_rows), 1)
        for j, dt in enumerate(action_space):
            counts[i, j] = sum(int(row["dt"]) == dt for row in condition_rows) / total
    x = np.arange(len(conditions))
    width = 0.18
    plt.figure(figsize=(8.2, 4.4))
    for j, dt in enumerate(action_space):
        plt.bar(x + (j - 1.5) * width, counts[:, j], width=width, label=f"dt={dt}")
    plt.xticks(x, conditions, rotation=20, ha="right")
    plt.ylabel("choice proportion")
    plt.title("Active sampler interval choices")
    plt.legend(frameon=False, ncols=4)
    save_current(fig_dir / "fig05_dt_choices.png")


def plot_learning_rate(rows: list[dict[str, str]], fig_dir: Path) -> None:
    groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["agent"] == "active_temporal":
            groups[(row["condition"], int(row["step"]))].append(
                float(row["learning_rate"])
            )
    plt.figure(figsize=(7.5, 4.4))
    for condition in sorted({condition for condition, _ in groups}):
        steps = sorted(step for (name, step) in groups if name == condition)
        means = [np.mean(groups[(condition, step)]) for step in steps]
        plt.plot(steps, means, linewidth=2.0, label=condition)
    plt.xlabel("step")
    plt.ylabel("posterior expected Kalman gain")
    plt.title("Adaptive learning rate in the active sampler")
    plt.legend(frameon=False)
    save_current(fig_dir / "fig06_learning_rate.png")


def plot_wait_cost_ablation(output_dir: Path, fig_dir: Path) -> None:
    rows = read_csv(output_dir / "wait_cost_ablation_summary.csv")
    costs = np.array([float(row["wait_cost"]) for row in rows])
    ratio_error = np.array([float(row["mean_final_ratio_abs_error"]) for row in rows])
    mean_dt = np.array([float(row["mean_dt"]) for row in rows])
    fig, ax1 = plt.subplots(figsize=(7.0, 4.2))
    ax1.plot(costs, ratio_error, marker="o", color="#1f4e79", linewidth=2.0)
    ax1.set_xlabel("wait cost")
    ax1.set_ylabel("final noise-ratio error", color="#1f4e79")
    ax1.tick_params(axis="y", labelcolor="#1f4e79")
    ax2 = ax1.twinx()
    ax2.plot(costs, mean_dt, marker="s", color="#c43c39", linewidth=2.0)
    ax2.set_ylabel("mean chosen dt", color="#c43c39")
    ax2.tick_params(axis="y", labelcolor="#c43c39")
    plt.title("Wait-cost ablation")
    fig.tight_layout()
    plt.savefig(fig_dir / "fig07_wait_cost_ablation.png", dpi=180)
    plt.close()


def plot_pareto_tradeoff(output_dir: Path, fig_dir: Path) -> None:
    rows = read_csv(output_dir / "summary.csv")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["agent"]].append(row)

    plt.figure(figsize=(8.8, 5.0))
    for agent in sorted({row["agent"] for row in rows}):
        agent_rows = groups[agent]
        mse = np.mean([float(row["mean_mse"]) for row in agent_rows])
        ratio_error = np.mean(
            [float(row["mean_final_ratio_abs_error"]) for row in agent_rows]
        )
        mean_dt = np.mean([float(row["mean_dt"]) for row in agent_rows])
        size = 45 + 18 * mean_dt
        marker = "*" if agent == "active_temporal" else "o"
        plt.scatter(mse, ratio_error, s=size, marker=marker, label=AGENT_LABELS[agent])
    plt.xlabel("prediction MSE")
    plt.ylabel("final noise-ratio error")
    plt.title("Prediction-inference tradeoff")
    plt.grid(alpha=0.2)
    plt.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
    save_current(fig_dir / "fig08_pareto_tradeoff.png")


def finite_metric(rows: list[dict[str, str]], key: str) -> float:
    values = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
    if not values:
        return float("nan")
    return float(np.mean(values))


def plot_model_comparison_nll(output_dir: Path, fig_dir: Path) -> None:
    rows = read_csv(output_dir / "model_comparison_summary.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)

    models = [model for model in MODEL_LABELS if model in grouped]
    labels = [MODEL_LABELS[model] for model in models]
    nll = [
        finite_metric(grouped[model], "mean_negative_log_likelihood")
        for model in models
    ]
    mse = [finite_metric(grouped[model], "mean_mse") for model in models]
    x = np.arange(len(models))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    axes[0].bar(x, nll, color="#1f4e79")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].set_ylabel("mean negative log likelihood")
    axes[0].set_title("Predictive density")

    axes[1].bar(x, mse, color="#c43c39")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].set_ylabel("prediction MSE")
    axes[1].set_title("Point prediction")

    fig.suptitle("Perceptual model comparison")
    save_current(fig_dir / "fig09_model_comparison_nll.png")


def plot_model_comparison_ratio(output_dir: Path, fig_dir: Path) -> None:
    rows = read_csv(output_dir / "model_comparison_summary.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)

    models = [
        model
        for model in MODEL_LABELS
        if model in grouped
        and np.isfinite(finite_metric(grouped[model], "mean_final_ratio_abs_error"))
    ]
    labels = [MODEL_LABELS[model] for model in models]
    ratio_error = [
        finite_metric(grouped[model], "mean_final_ratio_abs_error") for model in models
    ]
    joint_accuracy = [
        finite_metric(grouped[model], "joint_class_accuracy") for model in models
    ]
    x = np.arange(len(models))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))
    axes[0].bar(x, ratio_error, color="#4c7f52")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].set_ylabel("final |log(v/s) error|")
    axes[0].set_title("Noise-ratio recovery")

    axes[1].bar(x, joint_accuracy, color="#7a4f9a")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("joint class accuracy")
    axes[1].set_title("Low/high noise classification")

    fig.suptitle("Volatility and stochasticity inference")
    save_current(fig_dir / "fig10_model_comparison_ratio.png")


def plot_competitive_model_loss(output_dir: Path, fig_dir: Path) -> None:
    rows = read_csv(output_dir / "competitive_model_summary.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)

    models = [model for model in COMPETITIVE_MODEL_LABELS if model in grouped]
    labels = [COMPETITIVE_MODEL_LABELS[model] for model in models]
    mse = [finite_metric(grouped[model], "mean_mse") for model in models]
    nll = [
        finite_metric(grouped[model], "mean_negative_log_likelihood")
        for model in models
    ]
    x = np.arange(len(models))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    colors = [
        "#c43c39" if model == "ats_joint_active" else "#1f4e79" for model in models
    ]
    axes[0].bar(x, mse, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].set_ylabel("prediction MSE")
    axes[0].set_title("Point prediction")

    axes[1].bar(x, nll, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].set_ylabel("mean negative log likelihood")
    axes[1].set_title("Predictive density")

    fig.suptitle("Active-sampling model competitors")
    save_current(fig_dir / "fig11_competitive_model_loss.png")


def plot_competitive_model_dt(output_dir: Path, fig_dir: Path) -> None:
    rows = read_csv(output_dir / "competitive_model_summary.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)

    models = [model for model in COMPETITIVE_MODEL_LABELS if model in grouped]
    labels = [COMPETITIVE_MODEL_LABELS[model] for model in models]
    mean_dt = [finite_metric(grouped[model], "mean_dt") for model in models]
    ratio_error = [
        finite_metric(grouped[model], "mean_final_ratio_abs_error") for model in models
    ]
    x = np.arange(len(models))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    colors = [
        "#c43c39" if model == "ats_joint_active" else "#4c7f52" for model in models
    ]
    axes[0].bar(x, mean_dt, color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].set_ylabel("mean chosen dt")
    axes[0].set_title("Sampling policy")

    axes[1].bar(x, ratio_error, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].set_ylabel("final |log(v/s) error|")
    axes[1].set_title("Noise-ratio inference")

    fig.suptitle("Model-as-agent comparison")
    save_current(fig_dir / "fig12_competitive_model_dt.png")


def generate_all_figures(output_dir: Path) -> Path:
    fig_dir = ensure_fig_dir(output_dir)
    rows = read_csv(output_dir / "metrics.csv")
    plot_task_schematic(fig_dir)
    plot_example_posterior(output_dir, fig_dir)
    plot_ratio_error(rows, fig_dir)
    plot_prediction_mse(rows, fig_dir)
    plot_dt_choices(rows, fig_dir)
    plot_learning_rate(rows, fig_dir)
    if (output_dir / "wait_cost_ablation_summary.csv").exists():
        plot_wait_cost_ablation(output_dir, fig_dir)
    plot_pareto_tradeoff(output_dir, fig_dir)
    if (output_dir / "model_comparison_summary.csv").exists():
        plot_model_comparison_nll(output_dir, fig_dir)
        plot_model_comparison_ratio(output_dir, fig_dir)
    if (output_dir / "competitive_model_summary.csv").exists():
        plot_competitive_model_loss(output_dir, fig_dir)
        plot_competitive_model_dt(output_dir, fig_dir)
    register_derived_figures(output_dir)
    return fig_dir
