# Competitors and CVPR assessment

Assessment date: 2026-10-21. This is research planning, not a prediction of acceptance. No full training result or direct pretrained-competitor win has been established by implementing the pipeline.

## Candidate contribution

The falsifiable hypothesis is that separating effective visual measurement uncertainty and temporal prediction uncertainty improves causal point-tracking accuracy per measured resource budget. The implemented operator is a neural coordinate filter plus a receding-horizon acquisition rule minimizing integrated forecast position variance. It changes actual future reads; it does not inspect a frame to decide whether that same frame should count as acquired.

The current small CNN/heatmap decoder, Gaussian update, learned variance heads, and covariance-based sensor scheduling are components, not individually established new contributions. Effective Q/R can absorb measurement bias, dynamics mismatch, and other errors. The present implementation does not prove physical noise separation. The original cognitive ATS parameter-information criterion has been replaced with an explicitly different task-risk criterion; cognitive-model evidence cannot be transferred to this neural version.

## Contemporary tracking competitors

| Competitor | Word-for-word primary-paper evidence | Comparison role |
| --- | --- | --- |
| TAPNext | “Our model is causal, tracks in a purely online fashion” | Strong low-latency causal baseline. [Paper](https://arxiv.org/abs/2504.05579) |
| TAPNext++ | “tracks points in sequences that are orders of magnitude longer” | Long-horizon and re-detection competitor. The 2026 publication is **CVPR Findings**. [Paper](https://arxiv.org/abs/2604.10582) |
| Track-On2 | “our model processes frames causally” | Causal memory-based competitor; its arXiv record identifies TPAMI 2026. [Paper](https://arxiv.org/abs/2509.19115) |
| CoTracker3 | “The model is available in online and offline variants” | Report the variant, window latency, support points, and accessible frames. [Paper](https://arxiv.org/abs/2410.11831) |

Do not compare a causal budget-limited method directly to an offline method as if information access were equal. A method labeled online may process a temporal chunk, so audit effective lookahead and output delay rather than relying on the name. All competitors need the same query convention, input resolution, declared pretraining data, actual acquisition budget, and explicit treatment of missing frames. Full-observation references answer a separate question. The optional TAPNext++ adapter makes one such sparse-cadence comparison executable; it is not a claim to reproduce the authors' reported rankings. Other competitors have not been run here.

## Closest functional precedents

Uncertainty-guided neural active perception already exists. The 2024 active-sensing paper states: “the perception model in each case is a variational autoencoder (VAE)”. It also optimizes information-driven sensing through a perception/action loop. This is component/functional overlap, although its fixation actions and objective differ from our time-selection tracking rule. [Paper, Discussion and Algorithm 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC11240181/)

This limited screen does not establish that the current covariance-risk acquisition formula is novel. A publication claim needs a dedicated comparison against stochastic sensor scheduling and active temporal acquisition at the equation/objective level. Performance alone cannot repair a novelty claim that reduces to an existing operator.

## What a strong result would establish

If ATS improves only over uniform/random schedules with this small CNN, that supports a useful scheduling effect for this implementation. It does not establish superiority to modern point trackers or explain whether Q/R separation caused the gain. If the advantage disappears when Q or R is fixed in the scheduler, that is informative about selection, but it still does not establish the need to learn both heads in the inference model. Separate retrained inference ablations are needed for that stronger claim.

My current judgment is that a high score on one DAVIS run is insufficient to call this CVPR-ready. A credible submission could arise if a distinct mechanism produces a reproducible accuracy/resource Pareto improvement over strong causal trackers, including when the acquisition rule is attached to a stronger visual backbone. The evidence should include multiple predeclared budgets and seeds, actual latency/compute including planning, diverse real evaluation sets, controlled source-level splits, and targeted ablations that distinguish dynamics error, visual ambiguity, occlusion, and calibration failure. A shared global frame-budget experiment would additionally be needed for a multi-point sensor-capture savings claim; the current per-query budget does not establish it.

These are project-specific evidentiary requirements, not formal CVPR rules or universal numerical acceptance thresholds. The published 2026 reviewer guidance asks reviewers to weigh “novelty and potential impact” alongside performance and says failing to exceed SOTA is not itself grounds for rejection. [CVPR 2026 reviewer guidance](https://cvpr.thecvf.com/Conferences/2026/ReviewerGuidelines)

The proposed go/no-go sequence is: first establish repeatable same-backbone gains beyond exploratory intervals; then run stronger causal competitors and model-level Q/R ablations; then test transfer across genuinely different real-video sources. If comparable-resource gains disappear against a strong backbone or simpler calibrated gate, revise the contribution rather than describing the prototype as a new general vision model. A final acceptance judgment must use those actual results and a more complete prior-art audit.
