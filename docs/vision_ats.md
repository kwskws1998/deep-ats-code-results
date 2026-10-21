# Neural encoder–decoder with active temporal sampling

This is a supervised 2D video-tracking prototype. Its purpose is to make the proposed perception/temporal-belief/acquisition loop executable and falsifiable. Synthetic execution is not evidence of general vision performance, cognitive validity, or algorithmic novelty.

## Architecture and units

`acquired image -> CNN measurement -> Gaussian state update -> clean-image decoder`

`current state covariance + Q forecast + last observed R -> next acquisition time`

The latent state is anchored to `[x, y, vx, vy]`, with image axes normalized to `[-1,1]` and velocity measured per physical frame. Coordinates are supervised. The encoder predicts a Gaussian coordinate measurement and diagonal R. R is an approximation to the encoder's coordinate error, not raw pixel variance or a VAE posterior variance. Bias and non-Gaussian errors can make this approximation miscalibrated.

The known mean motion uses damping 0.9 and spring strength 0.02. A causal GRU summarizes actual observations and innovations; its head forecasts diagonal acceleration variance Q. Gaussian acceleration affects both position and velocity through a shared acceleration map, so the full 4x4 state covariance includes their correlations. Every skipped physical frame executes a prediction step. Measurement updates use Joseph-form covariance propagation.

The MLP decoder reconstructs a clean image from estimated position. It is trained both with true training coordinates during warmup and with filtered states during joint training. All task predictions remain causal at evaluation. The prototype does not learn arbitrary dynamics or an unrestricted abstract latent space.

## Fitting and acquisition

Warmup fits the encoder's coordinate Gaussian NLL and a foreground-weighted clean-image reconstruction loss. Joint training uses independently randomized observation masks/budgets. Its loss is filtering coordinate NLL + 0.2 measurement coordinate NLL + 0.05 acceleration-residual NLL + 10 weighted reconstruction MSE. The Q loss uses realized acceleration residuals from training states; it does not consume generator variance labels. Q and R are positive bounded head outputs. These definitions make the meanings explicit but do not establish identifiability or calibration.

The ATS extension chooses a next observation time that minimizes predicted position variance summed over all remaining physical frames. For each candidate first acquisition, it evaluates a gap-capped uniform continuation using the remaining read budget. Q and R are held constant within a planning call. R is forecast from the last observed image; the policy cannot anticipate future image quality without observing it. The mean measurement update is integrated out through its expected covariance. The controller replans after every actual acquisition. This is approximate receding-horizon task-risk minimization, not the original scalar ATS mutual information about Q/R grid parameters, and not an optimal policy.

The first frame counts against the exact acquisition budget. Decisions receive only clock, remaining budget, current covariance, and causal Q/R forecasts. The complete frame array is retained by the rollout environment; only selected frames reach the encoder. A constant per-observation penalty is unnecessary when every primary arm has the same fixed read count.

## Controls and metrics

All primary controls share the same trained weights, motion filter, decoder, memory capacity, and read count:

- `uniform`: evenly spaced acquisitions starting at frame zero.
- `random`: a data-independent schedule, always including frame zero.
- `total_uncertainty`: read when forecast observation variance exceeds a validation-selected threshold, with a maximum gap and a remaining-budget feasibility constraint.
- `ats`: the Q/R-aware integrated-risk planner.
- `ats_fixed_q`, `ats_fixed_r`: replace only the planner's Q or R forecast with the respective median from frozen-model training observations. Inference still uses the original neural heads. These isolate scheduling, not the benefit of the noise heads in the filter.
- `all_frames`: the same model with an acquisition every frame. This uses a larger budget and is a reference only.

`max_gap` caps each adaptive decision and its approximate continuation. Uniform and random schedules are not capped. After the read budget is exhausted, the model predicts to the end of the clip.

The primary metric is mean clip position MSE, in squared normalized coordinate units, evaluated over every frame. Secondary outputs include RMSE, skipped-frame MSE, joint 2D position NLL, joint 95% ellipse coverage, clean-image MSE, encoder coordinate error, and Q/R forecasts. NLL includes its Gaussian normalization constant and numerical covariance jitter of `1e-7`. Continuous density NLL can be negative. True synthetic Q is recorded only for diagnostics; pixel noise labels are not ground truth for the encoder's R. Measurement error and R can be compared for calibration, but their aggregate agreement alone does not validate conditional calibration.

Acquired-image counts and decoded-image counts are saved. Policy and complete rollout wall time include planning overhead. Identical read counts do not imply identical total computation or latency. Timings are descriptive for the actual device, not a hardware-normalized efficiency benchmark. The decoder produces an output at every frame.

## Runs and data

```bash
python -m pip install -e '.[deep,dev]'
python scripts/run_vision_ats.py --output-root outputs/vision_ats_pilot_new
python -m pytest -q tests/test_ats_vision.py
PYTHONPATH=src python scripts/verify_artifacts.py outputs/vision_ats_pilot_new --check-current-source
```

CPU is the tested default. `--device cuda` is available for server use and enables deterministic PyTorch algorithms with the cuBLAS workspace setting. GPU execution has not been established by the CPU pilot; unsupported deterministic kernels raise rather than silently relaxing reproducibility. MPS is not offered because this Gaussian linear-algebra path has not been validated there.

Train, validation, and test have independent generator seeds and clip identities. The synthetic generator crosses two acceleration-variance levels with two pixel-noise levels. Test clips come from the same generator family as training clips; they do not establish distribution-shift robustness. Validation chooses a checkpoint by uniform-schedule position MSE and selects a threshold from `[0.002, 0.01, 0.05, 0.2, 1.0]`. Test labels never revise weights, thresholds, or ablation constants. Comparisons use paired whole-clip bootstrap intervals; they are exploratory and not corrected for multiple comparisons. One training seed is a pilot, not a replication study.

For an existing labeled dataset, provide `train.npz`, `validation.npz`, and `test.npz` in one directory and pass `--data-root PATH`, with matching `--sequence-length` and `--image-size`. Each NPZ must contain:

| Key | Shape / type | Meaning |
| --- | --- | --- |
| `frames` | N,T,C,H,W, float32 in [0,1] | Available noisy images |
| `clean_frames` | N,T,C,H,W, float32 in [0,1] | Clean decoder training/scoring targets |
| `states` | N,T,4, finite float32 | Normalized x,y,vx,vy labels |
| `conditions` | N, non-object strings | Reporting strata only |
| `clip_ids` | N, unique non-object strings | Globally distinct source clips |
| `process_variances` | Optional N,2 positive floats | Diagnostic true Q; never a model input |

Files are loaded with pickle disabled. Splits must have matching sequence/image shapes, globally disjoint IDs, and no exact reused frame sequence. This detects exact reuse, not related windows from the same source video; construct splits by source video/subject before conversion. Velocities and the known motion mean must use compatible units; adapting a real dataset may require changing the dynamics and its supervision. This interface does not turn arbitrary unlabeled video into a valid tracking benchmark.

Each run refuses to overwrite a nonempty directory and saves data identities, complete source/dependency fingerprints, `model.pt`, training/selection logs, per-clip metrics, aggregate/paired comparisons, all causal state predictions and acquisition masks, candidate scores, and two diagnostic figures. Interrupted runs are marked failed and require a new output directory. `load_checkpoint` restores the best model for inference. Resume training is not implemented.

The fixed first test clip supplies the qualitative example; it is not selected for favorable output. Metric recomputation and checkpoint replay are tested. No performance threshold is asserted by the smoke tests.
