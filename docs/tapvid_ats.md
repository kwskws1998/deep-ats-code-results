# TAP-Vid training, validation, and frozen testing

This pipeline learns a causal single-query point tracker from RGB videos, point coordinates, and occlusion labels. It requires no clean-image targets, velocities, known simulator motion law, or true Q/R. The earlier `run_vision_ats.py` moving-blob pipeline is separate.

## Dataset protocol

The TAP-Vid paper states: “we use Kubric for training the presented models”. Its real-video benchmarks are held out for evaluation. [TAP-Vid, section 3](https://proceedings.neurips.cc/paper_files/paper/2022/file/58168e8a92994655d6da3939e7cc0918-Paper-Datasets_and_Benchmarks.pdf)

Here training uses the publicly released **CoTracker3 Kubric variant**, not a claimed reproduction of the original TAP-Net MOVi-E training corpus. The release has approximately 6,000 scenes, 512x512 images, and 120 frames per scene. [Official dataset card](https://huggingface.co/datasets/facebook/CoTracker3_Kubric)

`prepare_tapvid_data.py` assigns whole Kubric source scenes to training or validation using a fixed seed before any temporal windowing. This is a custom validation split of the training release, not an official Kubric validation split. Training creates non-overlapping windows, default length 24; validation retains full 120-frame source videos. DAVIS retains complete clips exclusively in the test role. It is never used for gradient fitting, checkpoint selection, acquisition-threshold tuning, or visibility-threshold tuning.

Queries use each track's first visible point in `[t,y,x]` order; targets use `[x,y]`. Never-visible tracks are excluded as in the official reader. Targets and predictions are fractional raster coordinates, with metric distances measured after multiplying by 256. All frames after the query are scored, including frames that the model did not acquire. Query frames and preceding frames are excluded from metrics. An occluded point's coordinate is not used for coordinate supervision.

Legacy source pickles are decoded with a restricted NumPy unpickler. Prepared data use pickle-free NPZ. In the released Kubric archive, the field named `visibility` actually encodes invisibility; the official parser applies `logical_not` to obtain visibility. The adapter preserves that convention and tests it. [Official parser](https://github.com/facebookresearch/co-tracker/blob/main/cotracker/datasets/kubric_movif_dataset.py)

## Downloads and preparation

```bash
python -m pip install -e '.[vision,dev]'
python scripts/download_tapvid_data.py --davis --kubric-scenes 0 1
python scripts/prepare_tapvid_data.py --kubric-root data/external/tapvid/kubric --davis data/external/tapvid/tapvid_davis.zip --output-root data/external/tapvid/prepared256 --image-size 256
```

The downloader pins the Hugging Face dataset revision and verifies each selected archive against its published LFS SHA256. It supports verified byte-range resumption and refuses a mismatched existing file. DAVIS is about 1.67 GB and is checked against its published object MD5/ETag, then recorded with SHA256. Its verified MD5 must match before the evaluation can label a run as using the complete official DAVIS source. The official evaluator is downloaded at tapnet revision `730cda1c730877cfedbe01bf87fb1cadb78a565d`, with source hash recorded. Repository pagination supports higher scene indices.

Two Kubric scenes are sufficient only to check software execution: one source scene trains, another validates. They are not sufficient for a credible benchmark model. Select a materially larger, predeclared set of scenes for research training and use a new prepared directory. Archives are individually large; the complete release is not downloaded automatically. The preparation manifest freezes exactly which scenes and clips are used, their role, raw/prepared hashes, and resolution. Use source-scene separation for any new data; an exact-byte duplicate check does not detect all related or differently encoded videos.

For example, on a server with sufficient storage, this Python snippet requests 128 predeclared scenes without creating a shell script. These archives alone are roughly tens of gigabytes; prepared clips need additional space. This is a bounded research subset, not the complete training release.

```python
import subprocess
import sys

subprocess.run([
    sys.executable, "scripts/download_tapvid_data.py", "--kubric-scenes",
    *map(str, range(128)),
], check=True)
```

Then prepare a fresh directory such as `data/external/tapvid/prepared256_scenes128` and use its manifest for a fresh training run. Keep the split seed and test protocol fixed; do not select scenes or seeds by test performance.

## Architecture and fitting

A small CNN encodes an acquired image. The initial public query samples a fixed appearance descriptor. A query-conditioned spatial decoder predicts correspondence heatmaps; their moments and a learned calibration head produce a coordinate measurement and effective R. Another head predicts visibility. A Gaussian filter maintains `[x,y,vx,vy]` in fractional image units and advances once per physical frame. A causal GRU predicts mean acceleration and Q from previous observations/innovations. The filter downweights uncertain or likely invisible measurements using soft visibility. Unread images never reach the encoder.

The training loss combines visible-coordinate filtering NLL and Huber error, coordinate measurement NLL, heatmap cross-entropy, and visible/invisible BCE. Q receives gradients through filtering losses, not from simulator acceleration labels. Masks and observation budgets are randomized independently of image content. Q/R here mean effective model error; the learned heads do not by themselves identify distinct physical noise sources. Gaussian approximations, biased visual measurements, template drift, and uncertain future visibility remain limitations.

Checkpoint selection maximizes validation AJ at a 50% observation budget under a uniform schedule, with lower visible-coordinate MSE breaking ties. Validation subsets are deterministic and declared in the configuration. After choosing weights, training-only observed predictions supply median Q/R constants for the scheduling ablations. Validation then chooses a shared visibility threshold from `[0.3,0.5,0.7]` and a total-uncertainty acquisition threshold from `[0.001,0.01,0.1]`. Test evaluation refuses a checkpoint whose selection is unfinished or whose source/input identities changed.

The present reference-constant pass uses two deterministic queries from each of at most the first eight training windows. These are training-reference-subset medians, not estimates over the full training population; their representativeness should be checked before interpreting the fixed-Q/R ablations in a paper.

```bash
python scripts/run_tapvid_ats.py train --data-manifest data/external/tapvid/prepared256/manifest.json --output-root outputs/tapvid_train_seed17 --device cpu --steps 1000
```

For server training, choose explicit larger data/model/step budgets and `--device cuda`. The included small CNN is an experimental backbone, not a strong pretrained tracker. CUDA and optional pretrained weights need their own hardware validation; local CPU execution does not establish GPU throughput. No fixed number of training steps guarantees convergence. Inspect validation curves across multiple seeds.

Query descriptors use a gather-based bilinear sampler. Its CPU values and gradients are tested against `grid_sample` with border padding and `align_corners=False`; it avoids that operator's nondeterministic CUDA backward path. A CUDA-specific backward test is provided and skipped when no CUDA device is available. This removes a documented kernel incompatibility without claiming that a complete GPU training run was performed. [PyTorch grid_sample documentation](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.grid_sample.html), [deterministic operation support](https://docs.pytorch.org/docs/2.12/generated/torch.use_deterministic_algorithms.html)

The compute gate estimates remaining fitting time at validation checkpoints and defers if it exceeds `--max-hours` (default 12). This is an estimate, not a wall-clock guarantee; final calibration passes and I/O add work. Deferral saves an unfinished checkpoint without declaring final test readiness. To resume an interrupted/deferred run, repeat the original training command with `--resume`; only `--max-hours` may change. Completed runs refuse resumption. Atomic `last.pt` includes optimizer, step, random states, history, and best selection; exact interrupted-versus-uninterrupted replay is tested. A failure before the first durable checkpoint requires a fresh run.

## Observation budgets and test metrics

```bash
python scripts/run_tapvid_ats.py evaluate --data-manifest data/external/tapvid/prepared256/manifest.json --checkpoint outputs/tapvid_train_seed17/model.pt --output-root outputs/tapvid_test_seed17 --split test --fractions 0.25 0.5 1.0
```

This tests every DAVIS clip and every official first-visible query by default. `--max-queries` and `--max-clips` explicitly request a subset and prevent the run from being labeled complete official DAVIS. They are appropriate for execution probes, not a full-dataset claim. `--split validation` provides a separate saved validation report without changing fitted weights. The source manifest must be frozen before fitting and remain identical at evaluation.

Each independent query gets `ceil(fraction * remaining_frames)` acquisitions, including its query image. Other benchmark queries never enter that rollout. Counts are **image-query observations**, not the union of unique video frames across all queries. There is no cross-query image-feature cache. Thus the prototype does not establish a claimed percentage of global frame-capture savings for simultaneous multi-point tracking. Decode/resize of stored datasets is offline preparation; runtime measurements cover inference/planning after that preparation, not camera capture or end-to-end video decoding.

Uniform, random, total-predictive-uncertainty, ATS, fixed-planner-Q, and fixed-planner-R share weights and acquisition counts. The last two ablate only selection forecasts; they do not retrain/remove the noise heads in the filter. ATS approximately minimizes future integrated position variance with a gap-capped uniform continuation and frozen Q/R forecasts per planning call. It is a task-risk extension, not the scalar ATS parameter-information objective. Future visibility and image quality remain unknown. Visibility-gated effective R from the last real acquisition forecasts future reads. Matching read counts does not match total compute: planning cost is reported.

Metrics are the official AJ, average points within 1/2/4/8/16 pixel thresholds, and occlusion accuracy. Ratios are first aggregated over queries/time within a video, then averaged equally over videos. The implementation is checked numerically against the pinned official evaluator. Outputs use 0..1; multiply by 100 for paper percentages. Whole-video bootstrap comparisons are exploratory and not multiplicity-corrected. Budget-limited online scores are a modified protocol and cannot be directly substituted into published full-observation rankings. The 100% budget is useful as a quality reference; all internal scheduling arms must coincide there.

## External competitors

OpenCV LK is included as a classical control, not a competitive modern tracker:

```bash
python scripts/run_tapvid_baseline.py --data-manifest data/external/tapvid/prepared256/manifest.json --output outputs/tapvid_lk --backend lk
```

An optional official TAPNext++ API adapter accepts supplied local pretrained weights. Install its pinned implementation separately:

```bash
python -m pip install 'tapnet[torch] @ git+https://github.com/google-deepmind/tapnet.git@730cda1c730877cfedbe01bf87fb1cadb78a565d'
python scripts/run_tapvid_baseline.py --data-manifest data/external/tapvid/prepared256/manifest.json --output outputs/tapvid_tapnextpp --backend tapnextpp --checkpoint /absolute/path/tapnextpp_ckpt.pt --device cuda
```

The adapter initializes one independent query, uses no other annotated support points, and feeds only acquired images to the external online tracker. Missing times use causal hold or constant-velocity extrapolation based on past acquisitions, never retrospective interpolation. Sparse inputs change the tracker's usual temporal cadence, so report this adaptation explicitly and include a 100% acquisition reference. The pretrained checkpoint has a different training budget from the new CNN; report it separately rather than claiming a controlled architecture comparison. The adapter's interface is tested with a deterministic substitute; no claim is made that actual TAPNext++ weights were executed locally. Track-On2 and CoTracker3 are literature competitors, not implemented adapters in this repository.

## Verification and artifacts

```bash
python -m pytest -q tests/test_tapvid.py
PYTHONPATH=src python scripts/verify_artifacts.py outputs/tapvid_train_seed17 --check-current-source
PYTHONPATH=src python scripts/verify_artifacts.py outputs/tapvid_test_seed17 --check-current-source
```

Runs save configuration/source/dependency identities, selected and resumable checkpoints, training/validation histories, selected thresholds, per-video official metrics, independent query predictions, masks, candidate scores, acquisition counts, and wall time. Training and evaluation use separate output directories and refuse replacement of prior results. Metric parity, delayed queries, query-frame exclusion, exact budgets, no unread-image access, gradients, data isolation, archive parsing, resumed training, checkpoint replay, and completed artifact hashes are tested. Unit fixtures are explicitly generated test data, not research evidence.

## Completed local execution probe

The final local code passed 170 tests, with one CUDA-hardware test skipped. The real-data probe used one Kubric source scene for training (five 24-frame windows), one separate full source scene for validation, and the complete set of 30 DAVIS videos with two sampled query points each (60 of the 650 available trajectories). Training ran only 12 optimizer steps with feature dimension 16 and context dimension 24; validation selected step 1. This is execution evidence, not a trained research model or an official full-query benchmark result.

Final run folders have suffix `_20261021_verified`: `outputs/tapvid_real_pipeline_train_20261021_verified`, `outputs/tapvid_real_pipeline_validation_20261021_verified`, `outputs/tapvid_real_pipeline_test_20261021_verified`, and `outputs/tapvid_lk_pipeline_test_20261021_verified`. Earlier folders without the suffix preserve the initial sampler implementation. Sources and predictions remain separate across those runs.

The verified audit recomputes every saved metric using the pinned official evaluator, checks every per-query read budget, checks source isolation and artifact hashes, and requires the internal 100%-budget arms to produce identical predictions. The script is `outputs/tapvid_pipeline_verified_audit_20261021.py`; its report is `outputs/tapvid_pipeline_independent_checks_20261021_verified.json`. The short neural probe scores substantially below LK, so these execution results do not support a claim of model quality or CVPR readiness.
