"""Run a causal paper-based cognitive comparison with a measured local compute gate.

Primary: held-out quantized response NLL. Secondary: full-data BIC under a common
unit Gaussian, as in the released fitting convention. Original archived fits
are audited separately and are never inputs to fresh fitting or test forecasts.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
from scipy.io import loadmat

from ats.artifacts import (
    complete_run,
    prepare_run,
    source_fingerprints,
    verify_run,
    validate_resume_identity,
)
from ats.experiments import stable_seed, write_csv
from ats.human_data import load_bucket_participants, sha256_file
from ats.piray_models import (
    DETERMINISTIC_MODELS,
    PARAMETER_COUNTS,
    deterministic_predictions,
    fit_deterministic,
    fit_particle,
    fit_response_sigma,
    particle_predictions,
    random_effects_bms,
    unit_gaussian_bic,
)
from ats.prediction_models import (
    bucket_log_probabilities,
    observed_response_expectation,
)


FIXED_TEMPLATES = {}


def atomic_json(path, value):
    """Commit each finished subject atomically so interrupted server runs can resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def table(path, rows):
    """Save heterogeneous metric and provenance tables without discarding rows."""
    write_csv(path, rows, sorted(set().union(*(r.keys() for r in rows))))


def regression_metrics(target, prediction):
    """Evaluate response errors and agreement in the original response units."""
    covariance = np.mean((target - target.mean()) * (prediction - prediction.mean()))
    denominator = (
        np.var(target) + np.var(prediction) + (target.mean() - prediction.mean()) ** 2
    )
    correlation = (
        np.corrcoef(target, prediction)[0, 1]
        if min(np.std(target), np.std(prediction)) > 1e-12
        else None
    )
    return dict(
        mae=float(np.abs(target - prediction).mean()),
        mse=float(np.square(target - prediction).mean()),
        pearson=None if correlation is None else float(correlation),
        ccc=None if denominator <= 0 else float(2 * covariance / denominator),
    )


def benchmark_task(arguments):
    """Measure sustained released-size particle inference in worker processes."""
    outcomes, particles, passes, seed = arguments
    begin = time.perf_counter()
    for i in range(passes):
        particle_predictions(outcomes, [0.5, 0.5, 0.5], particles, seed + i)
    return time.perf_counter() - begin


def fit_subject(arguments):
    """Fit independent training/full-data tracks with no current response entering a prediction."""
    participant, models, pf_config = arguments
    y, responses = participant.outcomes, participant.responses
    training = np.zeros_like(responses, dtype=bool)
    training[:30] = True
    all_trials = np.ones_like(training)
    rows, predictions, traces = [], [], []
    for model in models:
        begin = time.perf_counter()
        if model == "hierarchical_pf":
            trained, train_mean, history_train = fit_particle(
                y, responses, training, **pf_config
            )
            full, full_mean, history_full = fit_particle(
                y, responses, all_trials, **pf_config
            )
            traces += [
                dict(
                    track=track,
                    evaluation=i,
                    parameters=row[:3].tolist(),
                    regularized_sse=float(row[3]),
                )
                for track, history in [
                    ("training", history_train),
                    ("full_data", history_full),
                ]
                for i, row in enumerate(history)
            ]
            training_status = full_status = (
                "fixed_budget_completed_global_convergence_unproven"
            )
        else:
            if model in ("ats_grid_fixed", "last_outcome"):
                key = (y.tobytes(), model)
                if key not in FIXED_TEMPLATES:
                    FIXED_TEMPLATES[key] = deterministic_predictions(y, model)[0]
                train_mean = full_mean = FIXED_TEMPLATES[key]
                trained = full = None
                training_status = full_status = "fixed_perceptual_model"
            else:
                trained = fit_deterministic(model, y, responses, training)
                full = fit_deterministic(model, y, responses, all_trials)
                train_mean = deterministic_predictions(y, model, trained.parameters)[0]
                full_mean = deterministic_predictions(y, model, full.parameters)[0]
                training_status = "converged" if trained.success else "nonconverged"
                full_status = "converged" if full.success else "nonconverged"
        sigma = fit_response_sigma(responses, train_mean, training)
        logp = bucket_log_probabilities(responses, train_mean, sigma)
        expected = observed_response_expectation(train_mean, sigma)
        full_sse = (
            float(np.square(full_mean - responses).sum())
            if full is None
            else full.training_sse
        )
        rows.append(
            dict(
                dataset=f"bucket_experiment{participant.experiment}",
                participant=participant.participant,
                model=model,
                n_train=120,
                n_test=80,
                perceptual_parameters=PARAMETER_COUNTS[model],
                heldout_readout_parameters=1,
                readout_sigma=sigma,
                readout_sigma_on_boundary=bool(sigma < 0.25001 or sigma > 29.999),
                training_parameters=(
                    [] if trained is None else trained.parameters.tolist()
                ),
                full_data_parameters=[] if full is None else full.parameters.tolist(),
                training_status=training_status,
                full_data_status=full_status,
                training_on_boundary=False if trained is None else trained.boundary,
                full_data_on_boundary=False if full is None else full.boundary,
                full_data_sse=full_sse,
                full_data_bic_unit_sigma=unit_gaussian_bic(
                    full_sse, 200, PARAMETER_COUNTS[model]
                ),
                heldout_nll_per_decision=float(-logp[~training].mean()),
                seconds=time.perf_counter() - begin,
                **regression_metrics(responses[~training], expected[~training]),
            )
        )
        for trial in range(30, 50):
            for block in range(4):
                predictions.append(
                    dict(
                        dataset=f"bucket_experiment{participant.experiment}",
                        participant=participant.participant,
                        model=model,
                        trial=trial + 1,
                        condition=block,
                        response=float(responses[trial, block]),
                        perceptual_mean=float(train_mean[trial, block]),
                        expected_response=float(expected[trial, block]),
                        negative_log_likelihood=float(-logp[trial, block]),
                    )
                )
    return dict(metrics=rows, predictions=predictions, pf_search_traces=traces)


def run_subjects(pool, participants, models, pf_config, directory):
    """Resume only finished participant files belonging to the locked run configuration."""
    results, pending = [], []
    for participant in participants:
        path = directory / f"{participant.participant}.json"
        if path.exists():
            cached = json.loads(path.read_text())
            if {r["model"] for r in cached["metrics"]} != set(models) or {
                r["participant"] for r in cached["metrics"]
            } != {participant.participant}:
                raise ValueError(f"Checkpoint identity mismatch: {path}")
            results.append(cached)
        else:
            pending.append((participant, path))
    jobs = {
        pool.submit(fit_subject, (p, models, pf_config)): path for p, path in pending
    }
    for i, future in enumerate(as_completed(jobs), 1):
        result = future.result()
        atomic_json(jobs[future], result)
        results.append(result)
        if i % 25 == 0 or i == len(jobs) or "pf" in directory.name:
            print(f"{directory.name}: {i}/{len(jobs)} finished", flush=True)
    return results


def summarize(rows):
    """Use equal participant weights for predictive metrics and within-person BIC differences."""
    indexed = {(r["dataset"], r["participant"], r["model"]): r for r in rows}
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["model"])].append(row)
    output = []
    for (dataset, model), people in sorted(groups.items()):
        result = dict(
            dataset=dataset,
            model=model,
            participants=len(people),
            test_decisions=sum(r["n_test"] for r in people),
        )
        for metric in [
            "heldout_nll_per_decision",
            "full_data_bic_unit_sigma",
            "mae",
            "mse",
            "pearson",
            "ccc",
        ]:
            values = [r[metric] for r in people if r[metric] is not None]
            result[metric] = float(np.mean(values)) if values else ""
        deltas = []
        for person in people:
            family = [
                r
                for (d, p, _), r in indexed.items()
                if d == dataset and p == person["participant"]
            ]
            deltas.append(
                person["full_data_bic_unit_sigma"]
                - min(r["full_data_bic_unit_sigma"] for r in family)
            )
        result["mean_within_person_delta_bic"] = float(np.mean(deltas))
        result["nonconverged_fits"] = sum(
            r["training_status"] == "nonconverged"
            or r["full_data_status"] == "nonconverged"
            for r in people
        )
        output.append(result)
    return output


def paired_comparisons(rows):
    """Bootstrap paired people, conditional on the common fixed public stimulus sequences."""
    output = []
    for dataset in sorted({r["dataset"] for r in rows}):
        subset = [r for r in rows if r["dataset"] == dataset]
        index = {(r["participant"], r["model"]): r for r in subset}
        people = sorted({r["participant"] for r in subset})
        for model in sorted({r["model"] for r in subset} - {"ats_grid_fixed"}):
            for metric in ["heldout_nll_per_decision", "mae", "ccc"]:
                delta = np.array(
                    [
                        index[(p, "ats_grid_fixed")][metric] - index[(p, model)][metric]
                        for p in people
                        if index[(p, "ats_grid_fixed")][metric] is not None
                        and index[(p, model)][metric] is not None
                    ]
                )
                rng = np.random.default_rng(
                    stable_seed(4819, "piray_comparison", dataset, model, metric)
                )
                boot = delta[rng.integers(0, len(delta), (5000, len(delta)))].mean(
                    axis=1
                )
                low, high = np.quantile(boot, [0.025, 0.975])
                output.append(
                    dict(
                        dataset=dataset,
                        reference="ats_grid_fixed",
                        baseline=model,
                        metric=metric,
                        difference=float(delta.mean()),
                        ci_low=float(low),
                        ci_high=float(high),
                        participants=len(delta),
                        bootstrap_replicates=5000,
                        multiplicity_adjustment="none",
                    )
                )
    return output


def group_comparisons(rows, label):
    """Feed only full-data, consistently normalized BIC evidence into group BMS."""
    output = []
    for dataset in sorted({r["dataset"] for r in rows}):
        subset = [r for r in rows if r["dataset"] == dataset]
        people = sorted({r["participant"] for r in subset})
        models = sorted({r["model"] for r in subset})
        index = {(r["participant"], r["model"]): r for r in subset}
        evidence = np.array(
            [
                [-0.5 * index[(p, m)]["full_data_bic_unit_sigma"] for m in models]
                for p in people
            ]
        )
        bms = random_effects_bms(evidence, stable_seed(314, label, dataset))
        for i, model in enumerate(models):
            output.append(
                dict(
                    dataset=dataset,
                    model=model,
                    scope=label,
                    participants=len(people),
                    frequency=float(bms["frequency"][i]),
                    protected_exceedance=float(bms["protected_exceedance"][i]),
                    omnibus_risk=bms["omnibus_risk"],
                )
            )
    return output


def audit_archived_fits(data_root):
    """Recompute archived in-sample evidence constants without using fitted archives for test prediction."""
    rows, summary = [], []
    constant_gap = 199 * 0.5 * np.log(2 * np.pi)
    for experiment in (1, 2):
        directory = data_root / "unpacked" / f"experiment{experiment}"
        pf = loadmat(directory / "model_hpl.mat", simplify_cells=True)
        kf = loadmat(directory / "model_kf4.mat", simplify_cells=True)
        sse = np.asarray(pf["loglik"], dtype=float)
        released = -0.5 * sse - 0.5 * np.log(2 * np.pi) - 1.5 * np.log(200)
        corrected = -0.5 * np.array([unit_gaussian_bic(s, 200, 3) for s in sse])
        kf_lme = np.asarray(kf["lme"], dtype=float)
        raw = loadmat(directory / "data.mat", simplify_cells=True)
        kalman_recomputed = np.array(
            [
                -0.5
                * unit_gaussian_bic(
                    np.square(
                        deterministic_predictions(
                            raw["meta_data"]["bag"], "kalman4", parameters
                        )[0]
                        - person["bucket"]
                    ).sum(),
                    200,
                    4,
                )
                for parameters, person in zip(kf["kalman_parameter"], raw["data"])
            ]
        )
        kalman_discrepancy = float(np.max(np.abs(kalman_recomputed - kf_lme)))
        if kalman_discrepancy > 1e-7:
            raise AssertionError(
                "Archived Kalman likelihood convention did not reproduce."
            )
        stored = "lme" in pf
        matches = (
            bool(np.allclose(pf["lme"], released, rtol=0, atol=1e-10))
            if stored
            else None
        )
        corrected_bms = random_effects_bms(
            np.column_stack([corrected, kf_lme]), seed=experiment
        )
        raw_bms = random_effects_bms(
            np.column_stack([released, kf_lme]), seed=experiment
        )
        for i in range(len(sse)):
            rows.append(
                dict(
                    dataset=f"bucket_experiment{experiment}",
                    participant=f"bucket_e{experiment}_{i+1:04d}",
                    scope="archived_in_sample_not_fresh_heldout",
                    pf_saved_sse=float(sse[i]),
                    pf_lme_released_formula=float(released[i]),
                    pf_lme_matched_normalization=float(corrected[i]),
                    kf_saved_lme=float(kf_lme[i]),
                    pf_lme_actually_stored=stored,
                )
            )
        summary.append(
            dict(
                dataset=f"bucket_experiment{experiment}",
                participants=len(sse),
                scope="archived_in_sample_not_fresh_heldout",
                pf_lme_constant_advantage=constant_gap,
                pf_bic_shift=2 * constant_gap,
                pf_lme_actually_stored=stored,
                pf_stored_lme_matches_released_formula=matches,
                kalman_max_absolute_lme_reproduction_error=kalman_discrepancy,
                pf_win_fraction_released_formula=float(np.mean(released > kf_lme)),
                pf_win_fraction_matched_normalization=float(
                    np.mean(corrected > kf_lme)
                ),
                pf_frequency_released_formula=float(raw_bms["frequency"][0]),
                pf_frequency_matched_normalization=float(corrected_bms["frequency"][0]),
                pf_pxp_matched_normalization=float(
                    corrected_bms["protected_exceedance"][0]
                ),
                pf_archive_sha256=sha256_file(directory / "model_hpl.mat"),
                kf_archive_sha256=sha256_file(directory / "model_kf4.mat"),
            )
        )
    return rows, summary


def plot_results(root, summary):
    """Show matched held-out response likelihoods separately from in-sample BIC."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for experiment, ax in zip((1, 2), axes):
        rows = sorted(
            [r for r in summary if r["dataset"] == f"bucket_experiment{experiment}"],
            key=lambda r: r["heldout_nll_per_decision"],
        )
        ax.barh(
            [r["model"] for r in rows], [r["heldout_nll_per_decision"] for r in rows]
        )
        ax.invert_yaxis()
        ax.set_xlabel("Held-out response NLL (nats/decision; lower is better)")
        ax.set_title(f'Experiment {experiment}; n={rows[0]["participants"]}')
    fig.savefig(root / "heldout_comparison.png", dpi=180)
    plt.close(fig)


def main():
    """Benchmark first, run feasible comparisons, and defer a day-scale PF refit explicitly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/external/stovol_learning_2024")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/piray_comparison_20261021")
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-local-hours", type=float, default=24)
    parser.add_argument(
        "--pf-mode", choices=["auto", "full", "pilot", "none"], default="auto"
    )
    parser.add_argument("--pf-particles", type=int, default=10000)
    parser.add_argument("--pf-repeats", type=int, default=10)
    parser.add_argument("--pf-evaluations", type=int, default=200)
    parser.add_argument("--pf-sigma-min", type=float, default=0.05)
    parser.add_argument("--paper-likelihood-without-floor", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        args.workers < 1
        or args.pf_particles < 2
        or args.pf_repeats < 1
        or args.pf_evaluations < 5
        or args.max_local_hours <= 0
    ):
        raise ValueError("Invalid resource budget.")
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[key] = "1"
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))
    participants, sources = load_bucket_participants(args.data_root)
    config = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in ("resume", "output_root")
    }
    config.update(
        primary_metric="Held-out quantized/censored Gaussian response NLL, first 30 train and final 20 test in each condition.",
        secondary_metric="Full-data BIC: SSE + 200*log(2*pi) + k*log(200), fixed response sigma=1 for every model.",
        predictive_fit="Perceptual parameters minimize training SSE; response sigma is then fitted with proper bin likelihood on training data only. This is a declared two-stage estimator.",
        behavioral_mapping="Direct causal perceptual mean; no fitted response inertia or additive bias. Heldout predictions use outcomes, not previous human responses.",
        kalman_constraints="Released four effect constraints, initial normalized variance=1; direct ratio parametrization.",
        pf_budget="Released 10000 particles, 10 MC repeats, 200 objective evaluations; Python GP-EI with common seeds replaces MATLAB bayesopt.",
        pf_profile="Released code uses sigma (standard deviation) in [.05,1] and an epsilon likelihood floor. Paper states a variance lower bound .1; use --pf-sigma-min 0.316227766 --paper-likelihood-without-floor for that sensitivity.",
        compute_gate="Measured concurrent PF throughput, two fitting tracks with 0.6+1 sequence length, plus declared 25 percent planning allowance.",
        pilot="If full PF exceeds the compute gate, fit the first two participants of each experiment using 1000 particles, 3 repeats, 40 evaluations. Pilot results are separate from full-cohort results.",
        archive_audit="Historical in-sample values are reported separately; never use saved fitted estimates in fresh training or heldout predictions.",
        interpretation="Inference component only; fixed observation schedule cannot validate the original ATS interval-choice policy.",
        intervals="5000 paired participant bootstrap samples; exploratory, no multiple-comparison correction; fixed shared stimuli.",
    )
    root = args.output_root
    if args.resume:
        manifest = json.loads((root / "manifest.json").read_text())
        recorded = json.loads((root / "data_sources.json").read_text())
        validate_resume_identity(
            manifest, config, source_fingerprints(Path.cwd()), recorded["raw"], sources
        )
    else:
        prepare_run(root, config, "piray_2024_core_model_comparison")
    atomic_json(root / "protocol.json", config)
    reference_root = args.data_root / "unpacked" / "code"
    atomic_json(
        root / "data_sources.json",
        {
            "raw": sources,
            "download": json.loads(
                (args.data_root / "source_manifest.json").read_text()
            ),
            "reference_code_sha256": {
                str(path): sha256_file(path)
                for name in [
                    "get_data.m",
                    "kf_fit.m",
                    "other_rw.m",
                    "hpl_model.m",
                    "hpl_fit.m",
                ]
                for path in [reference_root / name]
            },
        },
    )
    pf_config = dict(
        particles=args.pf_particles,
        repeats=args.pf_repeats,
        evaluations=args.pf_evaluations,
        seed=0,
        sigma_min=args.pf_sigma_min,
        released_floor=not args.paper_likelihood_without_floor,
    )
    try:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            gate_path = root / "compute_gate.json"
            if gate_path.exists():
                gate = json.loads(gate_path.read_text())
            else:
                list(
                    pool.map(
                        benchmark_task,
                        [
                            (participants[0].outcomes, 100, 1, i)
                            for i in range(args.workers)
                        ],
                    )
                )
                start = time.perf_counter()
                timings = list(
                    pool.map(
                        benchmark_task,
                        [
                            (participants[0].outcomes, args.pf_particles, 10, i * 20)
                            for i in range(2 * args.workers)
                        ],
                    )
                )
                elapsed = time.perf_counter() - start
                passes = 20 * args.workers
                core_hours = (
                    elapsed
                    / passes
                    * len(participants)
                    * args.pf_evaluations
                    * args.pf_repeats
                    * 1.6
                    / 3600
                )
                gate = dict(
                    workers=args.workers,
                    particles=args.pf_particles,
                    benchmark_passes=passes,
                    batch_wall_seconds=elapsed,
                    worker_batch_seconds=timings,
                    projected_pf_core_hours=core_hours,
                    planning_allowance=1.25,
                    projected_pf_planning_hours=1.25 * core_hours,
                    max_local_hours=args.max_local_hours,
                )
                atomic_json(gate_path, gate)
            full_pf = args.pf_mode == "full" or (
                args.pf_mode == "auto"
                and gate["projected_pf_planning_hours"] <= args.max_local_hours
            )
            gate["full_pf_selected"] = full_pf
            atomic_json(gate_path, gate)
            print(json.dumps(gate, indent=2), flush=True)
            results = run_subjects(
                pool,
                participants,
                list(DETERMINISTIC_MODELS),
                pf_config,
                root / "deterministic_participants",
            )
            pilot_results = []
            if full_pf:
                results += run_subjects(
                    pool,
                    participants,
                    ["hierarchical_pf"],
                    pf_config,
                    root / "full_pf_participants",
                )
            elif args.pf_mode in ("auto", "pilot"):
                selected = [
                    p
                    for experiment in (1, 2)
                    for p in [q for q in participants if q.experiment == experiment][:2]
                ]
                pilot_config = dict(
                    pf_config, particles=1000, repeats=3, evaluations=40
                )
                pilot_results = run_subjects(
                    pool,
                    selected,
                    ["hierarchical_pf"],
                    pilot_config,
                    root / "pilot_pf_participants",
                )
        metrics = [r for result in results for r in result["metrics"]]
        predictions = [r for result in results for r in result["predictions"]]
        summary = summarize(metrics)
        table(root / "participant_metrics.csv", metrics)
        table(root / "heldout_predictions.csv", predictions)
        table(root / "summary.csv", summary)
        table(root / "paired_comparisons.csv", paired_comparisons(metrics))
        table(
            root / "group_bms.csv",
            group_comparisons(metrics, "fresh_full_data_bic_common_likelihood"),
        )
        if pilot_results:
            pilot = [r for result in pilot_results for r in result["metrics"]]
            selected = {r["participant"] for r in pilot}
            matched = [r for r in metrics if r["participant"] in selected] + pilot
            table(root / "pilot_participant_metrics.csv", matched)
            table(root / "pilot_summary.csv", summarize(matched))
        archived, archived_summary = audit_archived_fits(args.data_root)
        table(root / "archived_constant_audit.csv", archived)
        table(root / "archived_constant_audit_summary.csv", archived_summary)
        atomic_json(
            root / "released_code_notes.json",
            dict(
                unit_gaussian_constant="hpl_fit.m: logf=-.5*loglik-.5*log(2*pi); kf_fit.m/other_rw.m: sum a log(2*pi) term per response. Saved experiment1 HPL lme exactly matches its released single-constant formula.",
                unavailable_field="hpl_fit.m reads data{n}.bird; get_data.m adds only .bag, and raw participant records contain no .bird field. The Python port uses the observed bag, consistent with paper Eq.9.",
                deviations="Python constrained optimization/GP-EI differ from MATLAB fmincon/bayesopt. Original Hessian-failure fallback is not copied. Profile and causal readout are explicit in protocol.json.",
                excluded_supplementary_families="This run covers the central PF/Kalman/RW comparison and ATS, not the supplementary DBD/Weber/HGF/KFEM families.",
            ),
        )
        checks = dict(
            expected_deterministic_people=len(
                {r["participant"] for r in metrics if r["model"] == "rw4"}
            )
            == 643,
            all_predictive_nll_finite=bool(
                np.isfinite([r["heldout_nll_per_decision"] for r in metrics]).all()
            ),
            all_deterministic_fits_converged=all(
                r["training_status"] != "nonconverged"
                and r["full_data_status"] != "nonconverged"
                for r in metrics
            ),
            source_raw_data_unchanged=all(
                sha256_file(Path(v["raw_file"])) == v["sha256"]
                for k, v in sources.items()
                if k.startswith("experiment")
            ),
            full_pf_completed=bool(full_pf),
            full_pf_status=(
                "completed"
                if full_pf
                else (
                    "deferred_compute_budget"
                    if args.pf_mode == "auto"
                    else args.pf_mode
                )
            ),
            published_likelihood_reproduced_exactly=False,
            primary_and_pilot_separate=True,
        )
        atomic_json(root / "checks.json", checks)
        plot_results(root, summary)
        complete_run(
            root,
            full_pf_completed=bool(full_pf),
            participants=len(participants),
            deterministic_models=list(DETERMINISTIC_MODELS),
            elapsed_scope="Measured, gated core model comparison; see checks and protocol for unfinished full PF refit.",
        )
        failures = verify_run(root, check_current_source=True)
        if failures:
            raise RuntimeError(failures)
        print(json.dumps(dict(summary=summary, checks=checks), indent=2), flush=True)
    except Exception as error:
        path = root / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        atomic_json(path, manifest)
        raise


if __name__ == "__main__":
    main()
