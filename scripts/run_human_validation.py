"""Fit public human behavior with frozen evaluation splits and independent recovery simulations."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path

import numpy as np

from ats.artifacts import complete_run, prepare_run, verify_run
from ats.choice_models import (
    CHOICE_MODELS,
    ChoiceFit,
    block_features,
    fit_choice_model,
    predict_choice_model,
    simulate_observe_participant,
    choice_parameter_names,
)
from ats.experiments import stable_seed, write_csv
from ats.human_data import load_bucket_participants, load_observe_blocks, sha256_file
from ats.prediction_models import (
    PREDICTION_MODELS,
    FITTED_PREDICTION_MODELS,
    fit_prediction_model,
    perceptual_predictions,
    predict_bucket_responses,
    simulate_bucket_responses,
    prediction_parameter_names,
)


TEMPLATES = {}


def fixed_templates(outcomes: np.ndarray) -> dict[str, np.ndarray]:
    """Cache predictors of the shared public stimulus sequence, never participant response targets."""
    key = outcomes.tobytes()
    if key not in TEMPLATES:
        TEMPLATES[key] = {
            model: perceptual_predictions(outcomes, model)
            for model in PREDICTION_MODELS
            if model not in FITTED_PREDICTION_MODELS
        }
    return TEMPLATES[key]


def table(path: Path, rows: list[dict]) -> None:
    """Write a portable table while preserving missing metrics as empty cells."""
    write_csv(path, rows, sorted(set().union(*(row.keys() for row in rows))))


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Avoid interpreting a correlation when either series is constant."""
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    """Report individual predictive accuracy, agreement, and error in original response units."""
    covariance = float(
        np.mean((target - target.mean()) * (prediction - prediction.mean()))
    )
    denominator = float(
        np.var(target) + np.var(prediction) + (target.mean() - prediction.mean()) ** 2
    )
    return {
        "mse": float(np.mean((target - prediction) ** 2)),
        "mae": float(np.mean(np.abs(target - prediction))),
        "pearson": correlation(target, prediction),
        "ccc": 2 * covariance / denominator if denominator > 0 else float("nan"),
    }


def fit_bucket_participant(participant) -> tuple[list, list, list]:
    """Use the first 30 predictions per block for fitting and the final 20 for conditional evaluation."""
    templates = fixed_templates(participant.outcomes)
    training = np.zeros_like(participant.responses, dtype=bool)
    training[:30] = True
    test = ~training
    metrics, estimates, predictions = [], [], []
    for model in PREDICTION_MODELS:
        fit = fit_prediction_model(
            model,
            participant.outcomes,
            participant.responses,
            training,
            templates.get(model),
        )
        expected, log_probability = predict_bucket_responses(
            fit, participant.outcomes, participant.responses, templates.get(model)
        )
        metrics.append(
            {
                "dataset": f"bucket_experiment{participant.experiment}",
                "participant": participant.participant,
                "model": model,
                "n_train": int(training.sum()),
                "fit_converged": fit.success,
                "fit_on_boundary": bool(fit.boundary_parameters),
                "num_parameters": len(fit.parameters),
                "n_test": int(test.sum()),
                "test_nll_per_decision": float(-log_probability[test].mean()),
                **regression_metrics(participant.responses[test], expected[test]),
            }
        )
        estimates.append(
            {
                "dataset": f"bucket_experiment{participant.experiment}",
                "participant": participant.participant,
                "model": model,
                "training_nll": fit.training_nll,
                "parameter_names": json.dumps(prediction_parameter_names(model)),
                "success": fit.success,
                "boundary_indices": json.dumps(fit.boundary_parameters),
                **{
                    f"parameter_{i}": float(value)
                    for i, value in enumerate(fit.parameters)
                },
            }
        )
        for trial in range(30, 50):
            for condition in range(4):
                predictions.append(
                    {
                        "dataset": f"bucket_experiment{participant.experiment}",
                        "participant": participant.participant,
                        "model": model,
                        "condition": condition,
                        "trial": trial + 1,
                        "response": participant.responses[trial, condition],
                        "predicted_response": expected[trial, condition],
                        "negative_log_likelihood": -log_probability[trial, condition],
                    }
                )
    return metrics, estimates, predictions


def fit_observe_participant(blocks: list) -> tuple[list, list, list]:
    """Fit behavioral-session choices and score the participant's separate scanner-session choices."""
    train = [block for block in blocks if block.session == "train"]
    test = [block for block in blocks if block.session == "test"]
    train_features = [block_features(block) for block in train]
    test_features = [block_features(block) for block in test]
    choices = np.concatenate([block.choices for block in test])
    valid = choices != -2
    indices = choices[valid] + 1
    metrics, estimates, predictions = [], [], []
    for model in CHOICE_MODELS:
        fit = fit_choice_model(model, train, train_features)
        probabilities = predict_choice_model(fit, test, test_features)
        selected = probabilities[valid]
        nll = -np.log(selected[np.arange(len(indices)), indices])
        observed = choices[valid] == 0
        brier = np.sum((selected - np.eye(3)[indices]) ** 2, axis=1)
        metrics.append(
            {
                "dataset": "observe_bet",
                "participant": blocks[0].participant,
                "model": model,
                "n_train": int(sum(np.sum(block.choices != -2) for block in train)),
                "fit_converged": fit.success,
                "fit_on_boundary": bool(fit.boundary_parameters),
                "num_parameters": len(fit.parameters),
                "n_test": int(valid.sum()),
                "test_nll_per_decision": float(nll.mean()),
                "accuracy": float(np.mean(np.argmax(selected, axis=1) == indices)),
                "brier": float(brier.mean()),
                "observed_observe_rate": float(observed.mean()),
                "predicted_observe_rate": float(selected[:, 1].mean()),
            }
        )
        estimates.append(
            {
                "dataset": "observe_bet",
                "participant": blocks[0].participant,
                "model": model,
                "training_nll": fit.training_nll,
                "parameter_names": json.dumps(choice_parameter_names(model)),
                "success": fit.success,
                "boundary_indices": json.dumps(fit.boundary_parameters),
                **{
                    f"parameter_{i}": float(value)
                    for i, value in enumerate(fit.parameters)
                },
            }
        )
        offset = 0
        for block in test:
            for trial in range(50):
                if block.choices[trial] != -2:
                    p = probabilities[offset + trial]
                    predictions.append(
                        {
                            "dataset": "observe_bet",
                            "participant": block.participant,
                            "model": model,
                            "block": block.block,
                            "trial": trial + 1,
                            "choice": int(block.choices[trial]),
                            "p_bet_minus": p[0],
                            "p_observe": p[1],
                            "p_bet_plus": p[2],
                            "negative_log_likelihood": -np.log(
                                p[block.choices[trial] + 1]
                            ),
                        }
                    )
            offset += 50
    return metrics, estimates, predictions


def bucket_recovery_job(
    model: str, index: int, seed: int, outcomes: np.ndarray
) -> tuple[list, list]:
    """Recover shared response parameters and perceptual families from independently generated quantized behavior."""
    rng = np.random.default_rng(
        stable_seed(seed, "independent_bucket_parameters", model, index)
    )
    truth = np.array(
        [
            rng.uniform(0.05, 0.85),
            rng.uniform(-3, 3),
            rng.uniform(np.log(2), np.log(10)),
        ]
    )
    if model == "delta_rule":
        truth = np.r_[truth, rng.uniform(0.05, 0.9)]
    elif model == "fitted_kalman":
        truth = np.r_[truth, rng.uniform(np.log(0.03), np.log(3))]
    responses = simulate_bucket_responses(
        outcomes, model, truth, stable_seed(seed, "bucket_recovery", model, index)
    )
    from ats.human_data import BucketParticipant

    participant = BucketParticipant(
        f"synthetic_{model}_{seed}_{index}", 0, responses, outcomes, np.arange(4)
    )
    metrics, estimates, _ = fit_bucket_participant(participant)
    for row in metrics:
        row.update(
            dataset="bucket_synthetic", generator=model, seed=seed, simulation=index
        )
    own = next(row for row in estimates if row["model"] == model)
    parameters = []
    for i, value in enumerate(truth):
        true_value = (
            np.exp(value) if i == 2 or (i == 3 and model == "fitted_kalman") else value
        )
        fitted = own[f"parameter_{i}"]
        fitted_value = (
            np.exp(fitted)
            if i == 2 or (i == 3 and model == "fitted_kalman")
            else fitted
        )
        parameters.append(
            {
                "dataset": "bucket_synthetic",
                "generator": model,
                "seed": seed,
                "simulation": index,
                "parameter": i,
                "true": true_value,
                "parameter_name": (
                    "response_sigma"
                    if i == 2
                    else (
                        "process_to_observation_ratio"
                        if i == 3 and model == "fitted_kalman"
                        else prediction_parameter_names(model)[i]
                    )
                ),
                "fitted": fitted_value,
                "fit_success": own["success"],
                "boundary_indices": own["boundary_indices"],
            }
        )
    return metrics, parameters


def choice_recovery_job(model: str, index: int, seed: int) -> tuple[list, list]:
    """Use new latent outcomes and self-generated observation histories for choice/model recovery."""
    rng = np.random.default_rng(
        stable_seed(seed, "independent_choice_parameters", model, index)
    )
    if model == "constant":
        truth = np.array([rng.uniform(-1, 1), rng.uniform(-1, 2)])
    elif model == "leaky_threshold":
        truth = np.array(
            [
                rng.uniform(0.03, 0.5),
                rng.uniform(2, 5),
                rng.uniform(0.3, 1.2),
                rng.uniform(0.2, 0.8),
                rng.uniform(0.5, 2),
            ]
        )
    else:
        truth = np.array(
            [
                rng.uniform(3, 15),
                rng.uniform(-1, 2),
                rng.uniform(-0.5, 0.5),
            ]
        )
        if model != "bayes_value":
            extra = rng.uniform(-2, 4) if model == "bayes_time" else rng.uniform(2, 20)
            truth = np.r_[truth, extra]
    blocks = simulate_observe_participant(
        model, truth, stable_seed(seed, "choice_recovery", model, index)
    )
    metrics, estimates, _ = fit_observe_participant(blocks)
    for row in metrics:
        row.update(
            dataset="observe_synthetic", generator=model, seed=seed, simulation=index
        )
    own = next(row for row in estimates if row["model"] == model)
    parameters = [
        {
            "dataset": "observe_synthetic",
            "generator": model,
            "seed": seed,
            "simulation": index,
            "parameter": i,
            "true": value,
            "parameter_name": choice_parameter_names(model)[i],
            "fitted": own[f"parameter_{i}"],
            "fit_success": own["success"],
            "boundary_indices": own["boundary_indices"],
        }
        for i, value in enumerate(truth)
    ]
    return metrics, parameters


def parallel_jobs(pool, function, arguments: list[tuple], label: str) -> list:
    """Print completed job counts and return every result without dropping failed fits or participants."""
    futures = [pool.submit(function, *argument) for argument in arguments]
    results = []
    for i, future in enumerate(as_completed(futures), 1):
        results.append(future.result())
        if i % 25 == 0 or i == len(futures):
            print(f"{label}: {i}/{len(futures)} completed", flush=True)
    return results


def summary_metrics(metrics: list[dict]) -> list[dict]:
    """Average participant-level held-out metrics, not individual trials pooled across people."""
    groups = defaultdict(list)
    for row in metrics:
        groups[(row["dataset"], row["model"])].append(row)
    summary = []
    for (dataset, model), rows in sorted(groups.items()):
        result = {
            "dataset": dataset,
            "model": model,
            "participants": len(rows),
            "test_decisions": sum(row["n_test"] for row in rows),
        }
        for field in sorted(
            set().union(*(row.keys() for row in rows))
            - {"dataset", "model", "participant", "n_train", "n_test"}
        ):
            values = np.asarray([row.get(field, np.nan) for row in rows], dtype=float)
            finite = values[np.isfinite(values)]
            result[field] = float(np.mean(finite)) if len(finite) else float("nan")
            result[f"valid_participants_{field}"] = len(finite)
        summary.append(result)
    return summary


def paired_bootstraps(metrics: list[dict], replicates: int = 5000) -> list[dict]:
    """Resample people as the unit of inference, conditional on the shared fixed stimulus sequences."""
    indexed = {(r["dataset"], r["model"], r["participant"]): r for r in metrics}
    results = []
    for dataset in sorted({row["dataset"] for row in metrics}):
        reference = (
            "ats_binary_parameter_ig" if dataset == "observe_bet" else "joint_grid"
        )
        participants = sorted(
            {row["participant"] for row in metrics if row["dataset"] == dataset}
        )
        baselines = sorted(
            {row["model"] for row in metrics if row["dataset"] == dataset} - {reference}
        )
        for baseline in baselines:
            fields = (
                ("test_nll_per_decision", "accuracy", "brier")
                if dataset == "observe_bet"
                else ("test_nll_per_decision", "mse", "mae", "pearson", "ccc")
            )
            for field in fields:
                difference = np.array(
                    [
                        indexed[(dataset, reference, p)][field]
                        - indexed[(dataset, baseline, p)][field]
                        for p in participants
                    ]
                )
                difference = difference[np.isfinite(difference)]
                if not len(difference):
                    continue
                rng = np.random.default_rng(stable_seed(4819, dataset, baseline, field))
                bootstrap = difference[
                    rng.integers(0, len(difference), size=(replicates, len(difference)))
                ].mean(axis=1)
                low, high = np.quantile(bootstrap, (0.025, 0.975))
                results.append(
                    {
                        "dataset": dataset,
                        "reference": reference,
                        "baseline": baseline,
                        "metric": field,
                        "difference": float(difference.mean()),
                        "ci_low": float(low),
                        "ci_high": float(high),
                        "paired_participants": len(difference),
                        "bootstrap_replicates": replicates,
                        "multiplicity_adjustment": "none",
                    }
                )
    return results


def recovery_summaries(
    metrics: list[dict], parameters: list[dict]
) -> tuple[list, list]:
    """Retain all model confusions and parameter failures, including ties and boundary estimates."""
    groups = defaultdict(list)
    for row in metrics:
        groups[
            (row["dataset"], row["generator"], row["seed"], row["simulation"])
        ].append(row)
    counts = defaultdict(int)
    for (dataset, generator, _, _), rows in groups.items():
        winner = min(
            rows, key=lambda row: (row["test_nll_per_decision"], row["model"])
        )["model"]
        counts[(dataset, generator, winner)] += 1
    confusion = []
    for dataset, generator in sorted({(key[0], key[1]) for key in groups}):
        models = sorted({row["model"] for row in metrics if row["dataset"] == dataset})
        total = sum(
            value
            for (d, g, _), value in counts.items()
            if (d, g) == (dataset, generator)
        )
        for winner in models:
            count = counts[(dataset, generator, winner)]
            confusion.append(
                {
                    "dataset": dataset,
                    "generator": generator,
                    "selected_model": winner,
                    "count": count,
                    "simulations": total,
                    "fraction": count / total,
                }
            )
    grouped_parameters = defaultdict(list)
    for row in parameters:
        grouped_parameters[(row["dataset"], row["generator"], row["parameter"])].append(
            row
        )
    recovery = []
    for (dataset, generator, parameter), rows in sorted(grouped_parameters.items()):
        true = np.array([row["true"] for row in rows])
        fitted = np.array([row["fitted"] for row in rows])
        recovery.append(
            {
                "dataset": dataset,
                "generator": generator,
                "parameter": parameter,
                "parameter_name": rows[0]["parameter_name"],
                "simulations": len(rows),
                "pearson": correlation(true, fitted),
                "mae": float(np.mean(np.abs(true - fitted))),
                "bias": float(np.mean(fitted - true)),
                "failed_fits": sum(not row["fit_success"] for row in rows),
                "boundary_fits": sum(
                    bool(json.loads(row["boundary_indices"])) for row in rows
                ),
            }
        )
    return confusion, recovery


def plots(
    root: Path, summary: list[dict], confusion: list[dict], parameter_rows: list[dict]
) -> None:
    """Inspect held-out likelihoods, actual/predicted observation rates, and unfiltered recovery confusions."""
    import matplotlib.pyplot as plt

    datasets = ("bucket_experiment1", "bucket_experiment2", "observe_bet")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for ax, dataset in zip(axes, datasets):
        rows = sorted(
            [row for row in summary if row["dataset"] == dataset],
            key=lambda row: row["test_nll_per_decision"],
        )
        ax.barh(
            [row["model"] for row in rows],
            [row["test_nll_per_decision"] for row in rows],
            color="#527d99",
        )
        ax.invert_yaxis()
        ax.set_title(dataset)
        ax.set_xlabel("Held-out NLL / decision (lower is better)")
    fig.suptitle("Public human behavior: participant means, frozen evaluation splits")
    fig.savefig(root / "human_heldout_likelihood.png", dpi=180)
    plt.close(fig)
    for dataset in sorted({row["dataset"] for row in confusion}):
        rows = [row for row in confusion if row["dataset"] == dataset]
        generators = sorted({row["generator"] for row in rows})
        candidates = sorted({row["selected_model"] for row in rows})
        lookup = {(r["generator"], r["selected_model"]): r["fraction"] for r in rows}
        matrix = np.array([[lookup[(g, m)] for m in candidates] for g in generators])
        fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
        artist = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for i in range(len(generators)):
            for j in range(len(candidates)):
                ax.text(
                    j,
                    i,
                    f"{matrix[i,j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > 0.55 else "black",
                )
        ax.set_xticks(range(len(candidates)), candidates, rotation=35, ha="right")
        ax.set_yticks(range(len(generators)), generators)
        ax.set_xlabel("Selected by held-out likelihood")
        ax.set_ylabel("Generating model")
        ax.set_title(dataset + ": independent closed-loop model recovery")
        fig.colorbar(artist, ax=ax, label="Selection fraction")
        fig.savefig(root / f"{dataset}_model_recovery.png", dpi=180)
        plt.close(fig)


def main() -> None:
    """Freeze the protocol before participant fitting; preserve all outcomes and archive source identities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/external"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/human_validation_20261021")
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--recovery-per-seed", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or args.recovery_per_seed < 1:
        raise ValueError("Worker and recovery counts must be positive.")
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    participants, bucket_sources = load_bucket_participants(
        args.data_root / "stovol_learning_2024"
    )
    observe, observe_sources = load_observe_blocks(args.data_root / "observe_bet_2017")
    protocol = {
        "scope": "Public human behavior plus separately labeled synthetic recovery; cognitive models only",
        "bucket_models": PREDICTION_MODELS,
        "choice_models": CHOICE_MODELS,
        "bucket_split": "first 30 trials per condition train; final 20 test, all four blocks; one-step conditional predictions",
        "bucket_response_likelihood": "Gaussian probabilities integrated over 2-unit bins; endpoints 10 and 90 censored",
        "bucket_individual_parameters": "response inertia, bias, noise; delta and Kalman also fit a learning parameter",
        "bucket_sensory_prior": "previous ATS grid and variance, common to all conditions; fixed affine units (outcome-60)/4",
        "bucket_replication": "experiment2 repeats a locked specification on new participants; stimuli are shared, not new stimulus generalization",
        "observe_split": "fit all behav.behav*.csv; test only distinct behav.00*.csv scanner-session blocks",
        "observe_likelihood": "three-way softmax with fixed 0.02 uniform lapse; published leaky-threshold mechanism uses ML with same lapse",
        "binary_adaptation": "joint finite-grid state-switching/emission-reliability inference; parameter IG versus conditional state IG and deadline controls",
        "binary_utility": "bet logits = +/- (reward sensitivity * predicted signed outcome + outcome-color bias); observe logit = bias + info weight * remaining horizon * information",
        "choice_parameter_names": {
            model: choice_parameter_names(model) for model in CHOICE_MODELS
        },
        "prediction_parameter_names": {
            model: prediction_parameter_names(model) for model in PREDICTION_MODELS
        },
        "original_interval_policy": "API supplied separately; constant per-observation charge cannot be identified from mandatory interval choices",
        "recovery": {
            "parameter_design": "independent uniform draws per parameter; response noise and Kalman ratio uniform on log scale",
            "seeds": [17, 29, 43],
            "participants_per_generator_per_seed": args.recovery_per_seed,
            "bucket_generators": [
                "joint_grid",
                "volatility_only_grid",
                "delta_rule",
                "fitted_kalman",
            ],
            "choice_generators": CHOICE_MODELS,
            "synthetic_dynamics": "new closed-loop behavior, 4 bucket blocks or 6 training + 5 test binary blocks",
        },
        "uncertainty": "5000 participant bootstrap replicates; exploratory intervals without multiple-comparison correction",
        "excluded_inputs": [
            "demographics",
            "published fitted estimates",
            "current/future outcomes before choice",
            "hidden bet outcomes",
            "reaction times as sampling intervals",
        ],
        "no_automatic_claim": "human fit of an inference model or a binary-task adaptation does not validate the complete original Gaussian ATS timing policy",
    }
    root = args.output_root
    prepare_run(root, protocol, "public_human_validation")
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (root / "data_sources.json").write_text(
        json.dumps(
            {
                "bucket": bucket_sources,
                "observe": observe_sources,
                "archives": {
                    name: json.loads(
                        (args.data_root / name / "source_manifest.json").read_text()
                    )
                    for name in ("stovol_learning_2024", "observe_bet_2017")
                },
            },
            indent=2,
        )
        + "\n"
    )
    try:
        metrics, estimates, predictions, synthetic_metrics, synthetic_parameters = (
            [],
            [],
            [],
            [],
            [],
        )
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in parallel_jobs(
                pool,
                fit_bucket_participant,
                [(participant,) for participant in participants],
                "human bucket",
            ):
                for target, rows in zip((metrics, estimates, predictions), result):
                    target.extend(rows)
            grouped = defaultdict(list)
            for block in observe:
                grouped[block.participant].append(block)
            for result in parallel_jobs(
                pool,
                fit_observe_participant,
                [(blocks,) for _, blocks in sorted(grouped.items())],
                "human observe/bet",
            ):
                for target, rows in zip((metrics, estimates, predictions), result):
                    target.extend(rows)
            table(root / "participant_metrics.csv", metrics)
            table(root / "participant_parameters.csv", estimates)
            table(root / "heldout_predictions.csv", predictions)
            summaries = summary_metrics(metrics)
            table(root / "human_summary.csv", summaries)
            table(root / "human_paired_comparisons.csv", paired_bootstraps(metrics))
            print("human fits and held-out predictions saved", flush=True)
            recovery = protocol["recovery"]
            arguments = [
                (model, i, seed, participants[0].outcomes)
                for model in recovery["bucket_generators"]
                for seed in recovery["seeds"]
                for i in range(args.recovery_per_seed)
            ]
            for rows, parameters in parallel_jobs(
                pool, bucket_recovery_job, arguments, "bucket recovery"
            ):
                synthetic_metrics.extend(rows)
                synthetic_parameters.extend(parameters)
            arguments = [
                (model, i, seed)
                for model in CHOICE_MODELS
                for seed in recovery["seeds"]
                for i in range(args.recovery_per_seed)
            ]
            for rows, parameters in parallel_jobs(
                pool, choice_recovery_job, arguments, "choice recovery"
            ):
                synthetic_metrics.extend(rows)
                synthetic_parameters.extend(parameters)
        table(root / "synthetic_model_scores.csv", synthetic_metrics)
        table(root / "synthetic_parameter_estimates.csv", synthetic_parameters)
        confusion, recovered = recovery_summaries(
            synthetic_metrics, synthetic_parameters
        )
        table(root / "model_recovery.csv", confusion)
        table(root / "parameter_recovery.csv", recovered)
        plots(root, summaries, confusion, synthetic_parameters)
        sources = {
            item["raw_file"]: item["sha256"]
            for key, item in bucket_sources.items()
            if key.startswith("experiment")
        }
        sources.update(observe_sources["raw_sha256"])
        if any(sha256_file(Path(path)) != digest for path, digest in sources.items()):
            raise RuntimeError("Raw behavioral inputs changed during fitting.")
        for row in metrics:
            if not np.isfinite(row["test_nll_per_decision"]):
                raise FloatingPointError("A held-out likelihood is nonfinite.")
        checks = {
            "integrity_passed": True,
            "all_human_fits_converged": all(row["success"] for row in estimates),
            "all_synthetic_fits_converged": all(
                row["fit_converged"] for row in synthetic_metrics
            ),
            "human_bucket_participants": len(participants),
            "human_observe_participants": len(grouped),
            "human_fits": len(estimates),
            "human_test_predictions": len(predictions),
            "synthetic_model_fits": len(synthetic_metrics),
            "synthetic_parameter_estimates": len(synthetic_parameters),
            "numerically_nonconverged_human_fits": sum(
                not row["success"] for row in estimates
            ),
            "boundary_human_fits": sum(
                bool(json.loads(row["boundary_indices"])) for row in estimates
            ),
            "raw_inputs_unchanged": True,
            "synthetic_results_are_human_evidence": False,
        }
        (root / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
        complete_run(
            root,
            interpretation="held-out public human behavior and diagnostic synthetic recovery; report weaknesses and task adaptation explicitly",
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
