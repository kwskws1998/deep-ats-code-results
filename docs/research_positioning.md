# Research positioning and evidence boundaries

## Implemented research question

The scalar model asks whether uncertainty-specific observation timing improves estimation efficiency or latent-state tracking under a declared sampling budget. It maintains a finite-grid posterior over stationary process variance q and observation variance r, with a scalar Kalman filter per hypothesis. Its one-step acquisition rule is a Bayesian experimental-design application.

The DQN extension learns task-action/interval values from belief summaries. The PPO extension receives temporal features and observes every environment step. Neither extension implements an amortized posterior network. A sequence encoder with learned posterior heads remains an unimplemented research direction.

## Closest primary sources

The following quotations are verbatim excerpts, not paraphrases.

- **Piray and Daw (2021), A model for learning based on the joint estimation of stochasticity and volatility.** The paper states: “we built a probabilistic model for learning in uncertain environments that tracks volatility and stochasticity simultaneously.” Joint noise-source inference is established prior work. [Publisher](https://www.nature.com/articles/s41467-021-26731-9)
- **Zhou et al. (2024), Timing as an Action: Learning When to Observe and Act.** The abstract states: “the choice of delay intervals could be naively folded in as part of a composite action.” Task-action/delay pairs therefore do not establish a new problem formulation. [Paper and software](https://proceedings.mlr.press/v238/zhou24c.html)
- **Bellinger et al. (2020), Active Measure Reinforcement Learning for Observation Cost Minimization.** Its costed return is “the discounted sum of rewards minus the sum of observation costs.” Measurement costs in RL are established prior work. [Paper](https://arxiv.org/abs/2005.12697)
- **Lee et al. (2018/2019), Bayesian Policy Optimization for Model Uncertainty.** Its policy architecture “encodes the belief distribution independently from the observable state.” Belief-conditioned neural policies are established. [Paper](https://arxiv.org/abs/1810.01014)
- **Foster et al. (2021), Deep Adaptive Design.** The abstract describes “learning an amortized design network upfront and then using this to rapidly run (multiple) adaptive experiments.” Learning an acquisition policy is established. [ICML paper](https://proceedings.mlr.press/v139/foster21a.html)
- **Huang et al. (2025), ALINE.** Its architecture is “trained via reinforcement learning with a reward based on self-estimated information gain.” Joint learned inference and acquisition also have close precedent. [Paper](https://arxiv.org/abs/2506.07259)

These sources establish component-level overlap and a direct precedent for joint action/timing decisions. They do not prove that the exact present combination has been published. Priority for that narrow combination remains unresolved. The repository does not contain faithful reproductions of these published algorithms, so it cannot claim superiority over them.

## A falsifiable contribution

A candidate contribution is that the joint noise-source posterior produces better timing decisions than equally costly nonadaptive schedules and generic uncertainty summaries. A useful test must isolate the acquisition rule from the inference model, observation rate, physical horizon, temporal memory, parameter count, and numerical acquisition noise.

The revised implementation supplies intermediate fixed intervals, shared trajectories, independent yoked schedules, a retrospective histogram diagnostic, separate parameter-recovery errors, marginal coverage, and episode-level paired contrasts. These support such an experiment but do not establish its outcome in advance. Select the primary utility and budgets before examining results. Preserve null and negative results.

At fixed sampling interval d, observation differences in the implemented stationary Gaussian model satisfy Var(y_n-y_(n-1)) = q*d + 2*r and lag-one covariance = -r. Both variances are therefore identifiable in principle under fixed sampling. A timing contribution must concern finite-sample efficiency or task utility, not the claim that variable intervals make an otherwise impossible inference possible.

## Limits on cognitive and neuroscience interpretation

The original scalar model is a normative simulation. Its simulation performance alone does not establish an explanation of human or neural behavior. Biological motivation does not establish a neural circuit implementation.

The new human-validation pipeline tests this limitation directly, with explicit scope boundaries. Piray and Daw's [2024 prediction task](https://doi.org/10.1038/s41467-024-53459-z) asks participants to “move their bucket to catch bags”; [versioned data](https://zenodo.org/records/13840905) contain continuous predictions suitable for testing the inference component. Blanchard and Gershman's [observe-or-bet task](https://doi.org/10.3758/s13415-017-0556-2) permits “either observe (pure exploration) or bet (pure exploitation)”; [author data and code](https://github.com/TommyBlanchard/ObserveBet) permit a distinct behavioral-training/scanner-testing split. A binary-state adaptation is required for the latter task. A fit of that adaptation is not evidence for the exact Gaussian ATS timing operator. The new pipeline does not use the existing published fitted parameters as held-out predictions, and it keeps synthetic recovery separate from human data. Results must be read from completed run artifacts before making any human-performance claim.

PPO's projection filters ignore action-conditioned dynamics and nonlinear observation structure. Their parameters must be described as model-dependent random-walk proxies, not validated physiological, physical, aleatoric, or epistemic uncertainty signals. Benefits over plain PPO could arise from temporal memory or feature engineering; history and EMA controls are necessary.

## Evidence required before stronger claims

- Faithful published joint-estimation and timing-action baselines, including adaptation details for irregular and noisy observations.
- A locked evaluation protocol, multiple complete training seeds, held-out parameter and dynamics conditions, and adequate training budgets.
- Separate continuous q and r recovery, posterior calibration, acquisition convergence, and error/cost frontiers.
- For changing noise conditions, a suitable hyperparameter transition model; the current stationary grid does not supply one.
- For cognitive claims, behavioral data, parameter and model recovery, and comparison of predictive behavioral signatures.
- For RWRL claims, matched live runs. Exported historical tables are contextual references only.
