# Active Temporal Sampling

This repository contains a scalar Bayesian sampling model (`ats`), a belief-conditioned DQN (`deep_ats_planner.training`), PPO with optional temporal features (`deep_ats_planner.deep_ppo`), and a coordinate-supervised video encoder/filter/decoder (`ats_vision`). They have different action spaces and evidence requirements. No general novelty or performance-superiority claim is established by the supplied historical results.

## Install and test

```bash
python -m pip install -e '.[dev,deep,paper]'
python -m pytest -q
```

The HGF adapter is validated against PyHGF 0.2.12. Run manifests record installed dependency versions and source hashes. The deep package re-exports the canonical ATS inference modules; fixes are shared rather than copied.

## Neural video encoder–decoder with temporal acquisition

For the real point-tracking pipeline, see [docs/tapvid_ats.md](docs/tapvid_ats.md): checksum-verified official TAP-Vid-DAVIS, source-disjoint published Kubric training/validation, query-conditioned spatial decoding, learned motion means, visibility prediction, resumed training, and official-metric parity tests. [docs/tapvid_research_assessment.md](docs/tapvid_research_assessment.md) separates implemented controls, optional pretrained competitors, and the evidence still needed for a CVPR submission. The moving-blob pilot below remains a separate implementation diagnostic.

The runnable implementation and protocol are described in [docs/vision_ats.md](docs/vision_ats.md). A CNN learns coordinate measurements and their effective error variance R. A differentiable Gaussian motion filter maintains position/velocity uncertainty, a causal GRU forecasts acceleration variance Q, and an MLP decoder predicts clean images. ATS chooses actual future frame acquisitions without inspecting unread frames.

```bash
python scripts/run_vision_ats.py --output-root outputs/vision_ats_new
```

The CPU defaults are a small synthetic moving-object pilot: 128 training clips, 24 validation clips, 32 test clips, 24 frames per clip, and 8 acquired frames including the first. The same frozen model is evaluated with uniform, random, total-predictive-uncertainty, and Q/R-aware sampling, plus two selection-only noise ablations. A full-frame reference uses 24 acquisitions and is explicitly unmatched. Every physical frame is scored. Checkpoint and threshold selection use validation clips only. The pilot requires coordinate/velocity and clean-image supervision and assumes a fixed known motion mean; it does not establish unsupervised Q/R identification or efficacy on real videos. Its integrated tracking-risk acquisition objective is an explicit adaptation of ATS, rather than the original scalar parameter-information objective.

## Public human behavior validation

```bash
python -m pip install -e '.[cognitive,dev]'
python scripts/download_human_data.py
PYTHONPATH=src python scripts/run_human_validation.py --workers 3
```

The downloader retains the Zenodo version and published MD5 checksums for Piray and Daw's 2024 prediction dataset, and a pinned commit for Blanchard and Gershman's observe-or-bet dataset. Downloaded MATLAB/R/Stan sources are inspected as reference material; they are not executed. Raw datasets stay in ignored `data/external/`. The run records raw-file SHA256 identities and excludes demographic fields and the authors' fitted estimates from model inputs.

The released bucket dataset contains 223 and 420 participant records, each with four condition-aligned blocks of 50 predictions. Each prediction precedes its corresponding outcome. The first 30 predictions in every block fit individual response inertia, bias, and response noise; the final 20 evaluate frozen estimates. The sensory prior is common across noise conditions and uses the original ATS grid in the declared units `(outcome-60)/4`. The likelihood integrates a Gaussian over the two-unit response bins, with endpoint censoring. Baselines include single-noise grids, fitted delta/Kalman rules, last outcome, and response history. One-step conditional predictions can use a person's preceding response, including preceding test responses, but not their current response or a future outcome. Experiment 2 uses new people but the same stimulus sequences, so this is not a test of new stimulus environments.

The second dataset has 18 participants with distinct behavioral and scanner sessions. Behavioral blocks fit parameters; scanner blocks are held out. Choices are bet minus, observe, and bet plus. Hidden outcomes during bets never enter inference, and missed responses advance the task clock without being scored as choices. The task-specific ATS variant uses an exact finite-grid binary hidden-state model with unknown switching probability and observation reliability. It tests parameter information gain against conditional state information gain, remaining-time, value-only, constant-choice, and a released leaky-threshold mechanism. The latter uses maximum likelihood and the same fixed lapse as other candidates, rather than reproducing the original Bayesian fitting procedure. These adaptations do not constitute validation of the original Gaussian interval policy.

Synthetic parameter and model recovery use independent seeds and newly generated closed-loop behavior. The generating agent receives feedback only under its own observation choices. Confusion matrices retain all incorrect selections and boundary estimates. Held-out human likelihoods, Pearson/CCC/MAE for bucket responses, choice accuracy/Brier scores, participant-level paired bootstrap contrasts, numerical convergence, and data provenance are saved separately. Synthetic recovery is an implementation and identifiability diagnostic, not human evidence. All intervals are exploratory without multiplicity correction.

The original interval model also has a reusable softmax likelihood in `ats.choice_models`. Its fitted coefficients multiply information gain and delay. A constant observation charge cancels between mandatory interval alternatives and cannot be recovered from those choices. Acquisition Monte Carlo variation is not a psychological response-noise model.

### Completed public-data run (2026-10-21)

The completed cognitive-only run is in `outputs/human_validation_20261021/`. It contains 5,895 human model fits and 1,728 fits to 240 synthetic participants. The human test data comprise 51,440 distinct bucket responses and 4,129 distinct observe-or-bet choices; the larger prediction-table row count includes repeated predictions by different models. Scores below are participant means. NLL is in nats per decision and lower is better; scores from the two tasks are not directly comparable.

| Bucket model | Experiment 1, 223 people | Experiment 2, 420 people |
| --- | ---: | ---: |
| Last outcome with fitted response inertia | 2.2095 | 2.2113 |
| Fitted delta rule | 2.2131 | 2.2128 |
| Fitted Kalman | 2.2164 | 2.2195 |
| Joint q/r grid | 2.3789 | 2.3612 |
| Volatility-only grid | 2.3992 | 2.3809 |
| Stochasticity-only grid | 2.4099 | 2.3896 |

The joint grid improves on either single-noise grid but loses to the simple fitted behavioral alternatives. Its NLL difference from the last-outcome baseline is +0.1695 in Experiment 1 (exploratory participant-bootstrap 95% interval [0.1470, 0.1945]) and +0.1498 in Experiment 2 ([0.1334, 0.1665]). This evaluates the present common sensory prior and response mapping; it does not reject every possible joint-inference cognitive model.

| Observe-or-bet model, 18 people | Held-out NLL |
| --- | ---: |
| Bayesian remaining-time control | 0.7693 |
| ATS binary parameter-information adaptation | 0.7707 |
| Bayesian conditional-state information | 0.7721 |
| Bayesian value only | 0.7885 |
| Leaky threshold with adapted ML fitting | 0.8150 |
| Constant choice probabilities | 1.0962 |

The ATS adaptation's difference from the remaining-time control is +0.0014 (exploratory participant-bootstrap 95% interval [-0.0095, 0.0119]). Its information-weight parameter has weak synthetic recovery (Pearson r = 0.3819; MAE = 9.1859 over 24 generated participants). Ordinary minimum-NLL model selection recovers ATS for 6/24 generated participants. Near ties matter: requiring a unique winner within 1e-6 nats per decision leaves only 5/24 uniquely correct selections. These diagnostics do not establish an individually interpretable information-seeking trait. Good recovery of the joint-grid response inertia, bias, and response noise does not establish recovery of individual latent q and r.

Five human ATS information weights reached the original upper bound of 50. A separate, preserved sensitivity run in `outputs/human_bound_sensitivity_20261021/` expands the bound to 200 and 1,000, using training choices only for fitting. No weights then reach the upper bound, but ATS held-out NLL is 0.7729 at either bound and its difference from the time control still has an interval spanning zero. Information-weight recovery remains r = 0.3819. The saved `driver.py` reproduces this supplementary analysis with a new `--output-root`; it preserves all caps rather than selecting one on test performance. Other candidates retain their primary fits for the supplementary synthetic selection comparison.

All human and synthetic fits numerically converged. Boundary estimates occurred in 2,846/5,895 human fits across the different candidate families; numerical convergence therefore must not be read as parameter identifiability. The original source/artifact verification and `outputs/human_validation_20261021_independent_checks.json` pass. The independent check matches exported targets to raw test trials and recomputes 35,370 individual metrics. The supplementary numerical-tie tables are `outputs/human_validation_20261021_model_recovery_tie_diagnostics.csv` and `outputs/human_validation_20261021_model_recovery_tie_summary.csv`. The present human results do not establish ATS-specific cognitive validity or novelty, and the original Gaussian interval policy remains untested in human interval-choice data.

## Paper-based inference comparison

`scripts/run_piray_comparison.py` compares the central released RW4, Kalman4, and hierarchical particle-filter families with the current fixed-prior ATS inference component and a last-outcome control. It does not reproduce the supplementary DBD, Weber, HGF, or KFEM comparisons. The Python implementation uses causal predictions made before the current outcome; the original Gaussian interval-selection policy cannot be tested with these fixed-schedule data.

The primary metric is held-out response NLL, in nats per decision, using the first 30 trials of each condition to fit and the final 20 to evaluate. All models use a direct perceptual mean with no response inertia or additive bias. Perceptual parameters minimize training SSE; one response-noise parameter is then estimated from training responses with the quantized/censored Gaussian likelihood. This is an explicit two-stage estimator. It does not reuse preceding test responses. Pearson, CCC, MAE, and MSE are secondary descriptive metrics.

The secondary comparison uses full-data BIC with the same unit-Gaussian response convention for every model: `BIC = SSE + 200*log(2*pi) + k*log(200)`. Here RW4 and Kalman4 have four parameters, the hierarchical PF has three, and the fixed-prior ATS and last-outcome controls have zero individual perceptual parameters. The response noise in this BIC track is fixed to one; it is separate from the predictive track's fitted response noise. Group model frequencies and protected exceedance probabilities use consistently normalized `-BIC/2` as approximate log evidence. A model's population prevalence and its average held-out NLL answer different questions.

Fresh full-cohort results are:

The verified run is saved in `outputs/piray_comparison_20261021_verified/`, including the locked protocol, raw-data identities, compute gate, participant checkpoints, likelihood/BIC summaries, bootstrap contrasts, and the separately labeled archived-fit audit. Earlier `piray_comparison*` directories retain intermediate execution checks.

| Model | Experiment 1 NLL, 223 people | Experiment 2 NLL, 420 people |
| --- | ---: | ---: |
| RW4 | 2.3013 | 2.2934 |
| Kalman4 | 2.3172 | 2.3202 |
| Fixed-prior ATS inference | 2.4078 | 2.3853 |
| Last outcome | 2.4237 | 2.4610 |

RW4 has the lowest mean predictive NLL, while Kalman4 has the largest BIC-based group model frequency, about 57% in each experiment. The fixed-prior ATS component is worse than both fitted alternatives in held-out NLL. This result concerns the current common ATS prior and a direct response mapping; it does not reject all possible individualized joint-inference models. The previous human-validation section uses a different response mapping and must not be treated as a matched comparison with this section.

The default particle-filter budget is the released code's 10,000 particles, 10 Monte Carlo repeats, and 200 optimization evaluations per person and fitting track. A measured compute gate defers the complete PF refit when it approaches a day. A four-person, reduced-budget PF pilot validates execution and remains separate from the full-cohort tables. Its fixed GP kernel and common random seeds differ from MATLAB's `bayesopt`; budget completion is not a global convergence guarantee. The source's standard-deviation range `[.05, 1]` and epsilon likelihood floor are explicit options. A variance-lower-bound sensitivity is available with `--pf-sigma-min 0.316227766 --paper-likelihood-without-floor`.

For a fresh local run, choose a new output directory:

```bash
python -m pip install -e '.[paper,dev]'
python scripts/run_piray_comparison.py --output-root outputs/piray_new_local
```

For the complete PF refit on a server:

```bash
python scripts/download_human_data.py
python scripts/run_piray_comparison.py --pf-mode full --workers 16 --output-root outputs/piray_full_server
```

After an interruption, repeat the same arguments and append `--resume`. Finished participant files are saved atomically. Source files, configuration, and raw data must remain unchanged. Explicit process spawning applies the worker thread limits on Linux as well as macOS. A new run must use a new output directory.

The archived-fit audit is separate from the fresh experiment. In released `hpl_fit.m`, `logf = -.5*loglik -0.5*log(2*pi)` includes a single Gaussian constant for 200 responses, whereas `kf_fit.m` includes one constant per response. Experiment 1's saved HPL evidence exactly matches that released formula, and the Kalman evidence reproduces from its saved parameters and raw data. Matching the normalization shifts PF evidence by -182.8688 nats per person, equivalent to +365.7375 BIC. This is a finding about the distributed code and cached in-sample artifacts, not a fresh PF fit or a substitute for the deferred held-out comparison. The audit also records the missing `data{n}.bird` lookup in the released fitter; the Python implementation uses observed bags.

## Revised scalar experiments

```bash
PYTHONPATH=src python -m ats.cli smoke --run-name revised_smoke --seed 11
PYTHONPATH=src python -m ats.cli run --run-name revised_observations_seed11 --seed 11
PYTHONPATH=src python -m ats.cli run --run-name revised_time_seed11 --seed 11 --time-budget 320
```

Use a fresh run name each time. Training and simulation commands refuse to overwrite nonempty run directories. Existing outputs predate the revised schema and are historical artifacts; they are not independent replications or measurements of the revised implementation.

The default observation budget fixes the number of samples. `--time-budget` instead fixes elapsed physical time and masks intervals that cannot fit before the boundary. Prediction error on observations still uses each policy's chosen horizon. `mean_time_tracking_mse` evaluates the held latent estimate at every physical time unit, including periods without an observation. These metrics answer different questions.

Within a condition and episode, all schedules share one presampled latent trajectory and one observation-noise sequence indexed by physical time. Policy randomness uses a separate seed. Controls include fixed intervals 1, 2, 4, and 8; random and cyclic schedules; an independently generated yoked schedule; and a shuffled interval-histogram diagnostic. The histogram diagnostic is retrospective because it uses the recipient active episode's complete schedule. It is not a deployable policy or a prospective baseline. The independent donor schedule does not use the recipient trajectory.

Core outputs include:

- `manifest.json`: full configuration, code hashes, dependency versions, budget definition, and artifact hashes.
- `metrics.csv`, `trajectories.csv`, `summary.csv`: observation forecasts, exact predictive-mixture likelihoods, elapsed time, costs, latent tracking error, separate log-variance estimation errors, and marginal credible-interval coverage.
- `paired_comparisons.csv`: condition-stratified paired episode bootstrap contrasts. These are exploratory intervals without multiplicity adjustment; individual time steps are not treated as independent experiments.
- `example_posterior.npz`: the prespecified first condition and episode, at the same budget as the main run. No search for a favorable realization is performed.
- Optional fixed-sequence and model-as-agent comparisons, plus figures.

The noise variances are stationary within an episode. The grid posterior is exact conditional on its finite hypothesis grid and chosen initialization, not a validated filter for changing noise hyperparameters. Marginal 95% intervals are not joint 95% regions. Initial-state assumptions and finite-grid approximation can affect coverage.

## Acquisition objective and costs

The default interval score is one-step expected information gain minus waiting and observation costs. Entropy acquisition estimates expected posterior-to-prior KL with 128 Monte Carlo samples and common random numbers across candidate intervals. Finite-sample acquisition remains approximate. `--information-objective log_ratio` instead minimizes posterior variance of the log ratio; that criterion is not mutual information.

`--observation-cost` charges a constant once per observation, while `--wait-cost` charges per physical time unit. A constant observation charge does not change interval rankings in a mandatory, one-step, per-decision objective; total charges nevertheless differ when policies take different numbers of observations under a time budget. This is not an optimal stopping policy.

`--observation-rate-cost` is retained as an alias for `--observation-cost`. `--cost-mode legacy_rate` explicitly restores the former inverse-interval penalty for historical sensitivity checks; it should not be interpreted as a literal constant measurement cost. Changing the cost mode or information estimator prevents direct comparison to historical output files.

The PyHGF adapter propagates unit transitions, marking intermediate observations missing. This preserves the observation equation instead of scaling its value coupling by a long interval. Its initial process variance matches the requested value, and invalid internal states raise an error instead of being converted into clipped forecasts. Grid models are scored using their full predictive mixture. Gaussian baselines retain their own Gaussian predictive approximation. HGF parameters are prespecified, not fitted to maximize test performance.

## Belief-conditioned DQN

```bash
PYTHONPATH=src python -m deep_ats_planner.cli smoke --run-name revised_dqn_smoke --seed 11 --eval-episodes 2
PYTHONPATH=src python -m deep_ats_planner.cli train --run-name revised_dqn_seed11 --seed 11
PYTHONPATH=src python -m deep_ats_planner.cli train --run-name revised_dqn_raw_seed11 --seed 11 --no-belief-features
```

This path retains Bayesian grid inference and learns an MLP value function over task-action and sampling-interval pairs. It does not amortize posterior inference. Its summary features are approximate policy inputs, not a sufficient representation of the complete Bayesian belief. The finite episode's remaining decision budget is included in the input.

The default `--discount-basis physical_time` discounts continuation by `gamma ** dt`, discounts the endpoint payoff and observation charge, and accumulates discounted waiting costs at each unit transition. `--discount-basis decision` explicitly selects a per-decision objective. One endpoint bandit reward is delivered per decision; this environment does not supply a reward at every skipped physical step.

`deep_evaluation.csv` evaluates the frozen greedy policy and reactive fixed-interval controls on independent seeds, with one additional held-out parameter condition. Training and evaluation remain separate. Fixed-interval reactive controls use the common grid filter. `discounted_return` reports the optimized return definition; undiscounted reward and cost totals are also retained. Raw-feature and belief-feature DQNs have identical network input dimensions. Repeated complete training seeds, longer training, and stronger temporal baselines remain necessary for performance conclusions.

## PPO feature ablations

```bash
PYTHONPATH=src python -m deep_ats_planner.cli ppo-smoke --run-name revised_ppo_smoke --seed 11 --device cpu
PYTHONPATH=src python -m deep_ats_planner.cli ppo-train --run-name revised_ppo_ats_seed11 --seed 11 --feature-mode ats
PYTHONPATH=src python -m deep_ats_planner.cli ppo-train --run-name revised_ppo_history_seed11 --seed 11 --feature-mode history
PYTHONPATH=src python -m deep_ats_planner.cli ppo-train --run-name revised_ppo_ema_seed11 --seed 11 --feature-mode ema
PYTHONPATH=src python -m deep_ats_planner.cli ppo-train --run-name revised_ppo_none_seed11 --seed 11 --feature-mode none
```

`--no-ats` remains a compatibility option for the no-feature control; an explicit `--feature-mode` takes precedence. All four modes use the same padded input width and actor-critic parameter count. ATS uses a shared projection basis across vector environments, with fixed coordinate scales derived from public observation bounds. Unbounded coordinates use scale 1. The history control exposes recent observations; the EMA control exposes means, residual variances, and increments.

These random-walk features are model-dependent proxies. The encoder does not condition its transition model on actions, and its inferred parameters are not established estimates of physical process noise and sensor noise in nonlinear control systems. Small positive grid floors approximate low noise; exactly zero variance is not represented.

PPO observes every step. It does not learn sampling intervals. Truncations bootstrap from the final observation, and advantage traces stop at resets. Observation normalization is held fixed within each rollout and updated between rollouts. All modes receive the same normalization procedure. `ppo_evaluation.csv` uses frozen normalization and deterministic actions on independent episode seeds. `manifest.json` reports actual consumed timesteps, which can be less than the request when a final incomplete rollout batch is omitted.

## Validation suite

```bash
PYTHONPATH=src python scripts/validate_revision.py
```

The default suite runs three distinct seeds; both scalar budget definitions; belief and raw-feature DQNs; and all four PPO feature modes. It saves a suite manifest, per-run artifacts, acquisition-stability diagnostics, and combined metrics. `--section core`, `--section dqn`, or `--section ppo` runs one family. Use `--output-root` to select a fresh destination. `--figures` also renders scalar figures.

These runs are deliberately short implementation checks, not a convergence, novelty, or superiority study. They do not constitute a reproduction of the published joint-estimation model or timing-action algorithms. Those comparisons remain research requirements.

## Optional RWRL

RWRL additionally requires MuJoCo, `dm-control`, and Google's `realworldrl_suite`. Each training environment receives its own log destination. The wrapper preserves the dm_env termination/truncation distinction and closes the underlying environment.

```bash
PYTHONPATH=src python -m deep_ats_planner.cli ppo-train --run-name revised_rwrl_walker_seed11 --seed 11 --env-id rwrl --rwrl-domain walker --rwrl-task realworld_walk --rwrl-combined-challenge hard
PYTHONPATH=src python -m deep_ats_planner.cli rwrl-baselines --output outputs/rwrl_reference.csv
```

The exported D4PG/DMPO numbers are historical reference values, not matched runs. A valid RWRL performance comparison requires matched environment versions, settings, training budgets, repeated seeds, and evaluation protocols.

Verify a completed suite and its source identity with:

```bash
PYTHONPATH=src python scripts/verify_artifacts.py outputs/revision_validation_20261021 --check-current-source
```

Manifests distinguish completed and failed runs and include implementation, entrypoint, test, and installed HGF source hashes. Figure regeneration records its own source identity and refuses to conceal changes to recorded data.

## RTX 4090 TAP-Vid experiment

For the real-video ATS-CNN experiment and one published Track-On-R reference, use [the 4090 setup/run guide](docs/tapvid_4090.md). The entrypoints are `scripts/setup_tapvid_4090.py` and `scripts/run_tapvid_4090.py`; `configs/tapvid_4090.json` freezes the training and evaluation plan. External competitor training is not required.

