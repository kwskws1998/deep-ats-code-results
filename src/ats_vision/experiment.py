"""Train on labeled clips, select on validation, and compare frozen schedulers.

No policy receives labels or unread images. Test targets are accessed only after
the complete causal rollout. The same checkpoint and read budget are used for
all primary methods; all_frames is an explicitly unmatched reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import os
import random
import time

import numpy as np
import torch
from torch import Tensor

from ats.artifacts import complete_run, prepare_run, record_run_failures
from ats_vision.data import VideoData, load_data, moving_blobs, validate_splits
from ats_vision.figures import plot_results
from ats_vision.model import Belief, VideoATSModel, gaussian_nll
from ats_vision.sampling import SamplingPolicy


@dataclass(frozen=True)
class VisionConfig:
    """Small CPU defaults; larger runs and labeled external splits are explicit."""

    output_root: Path = Path("outputs/vision_ats_pilot")
    data_root: Path | None = None
    seed: int = 17
    train_clips: int = 128
    validation_clips: int = 24
    test_clips: int = 32
    sequence_length: int = 24
    image_size: int = 24
    observation_budget: int = 8
    hidden_size: int = 32
    warmup_epochs: int = 4
    epochs: int = 12
    batch_size: int = 16
    learning_rate: float = 0.001
    max_gap: int = 8
    threads: int = 2
    device: str = "cpu"

    def output_dir(self) -> Path:
        """Follow the shared manifest interface without touching older run folders."""
        return Path(self.output_root)

    def validate(self) -> None:
        """Check compute settings before creating an output directory."""
        positive = (
            self.train_clips,
            self.validation_clips,
            self.test_clips,
            self.epochs,
            self.batch_size,
            self.hidden_size,
            self.max_gap,
            self.threads,
            self.sequence_length,
            self.image_size,
            self.observation_budget,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in positive
        ):
            raise ValueError(
                "Counts, epochs, and thread settings must be positive integers."
            )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (self.seed, self.warmup_epochs)
        ):
            raise ValueError("seed and warmup_epochs must be nonnegative integers.")
        if self.sequence_length < 3 or self.image_size < 8 or self.hidden_size < 4:
            raise ValueError("Invalid seed, warmup, image size, or sequence length.")
        if not 1 <= self.observation_budget <= self.sequence_length:
            raise ValueError(
                "observation_budget must be between 1 and sequence_length."
            )
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError(
                "device must be cpu or cuda; MPS is not validated for this filter."
            )


def write_json(path: Path, value) -> None:
    """Write strict JSON so nonfinite results cannot silently enter a report."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a nonempty table with a fixed field order."""
    if not rows:
        raise ValueError("Refuse an empty results table.")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def training_mask(
    batch: int, length: int, budget: int, rng: np.random.Generator
) -> np.ndarray:
    """Randomize observed timestamps and budgets independently of all clip values."""
    mask = np.zeros((batch, length), dtype=bool)
    mask[:, 0] = True
    for index in range(batch):
        count = int(rng.integers(max(1, budget // 2), min(length, 2 * budget) + 1))
        mask[index, rng.choice(np.arange(1, length), count - 1, replace=False)] = True
    return mask


def sequence_loss(
    model: VideoATSModel, frames: Tensor, clean: Tensor, states: Tensor, mask: Tensor
) -> dict[str, Tensor]:
    """Supervise physical filtering, visual measurement R, transition Q, and decoding.

    Q targets are realized acceleration residuals, not the generator's variance
    labels. R is trained against realized encoder-coordinate error. Their NLLs
    establish explicit meanings but do not guarantee out-of-domain calibration.
    """
    batch, length = frames.shape[:2]
    belief = model.initial_belief(batch)
    last_read = frames.new_zeros(batch)
    means, covariances, measurements, processes = [], [], [], []
    for clock in range(length):
        if clock:
            belief, q = model.predict(belief)
            residual = (
                states[:, clock, 2:]
                - (states[:, clock - 1] @ model.transition.T)[:, 2:]
            )
            processes.append(
                0.5
                * (np.log(2 * np.pi) + q.log() + residual.square() / q).sum(-1).mean()
            )
        indices = torch.nonzero(mask[:, clock], as_tuple=False).flatten()
        if indices.numel():
            elapsed = clock - last_read[indices]
            belief, measurement, variance = model.observed_subset(
                belief,
                indices,
                frames[indices, clock],
                elapsed,
            )
            measurements.append(
                gaussian_nll(
                    states[indices, clock, :2], measurement, torch.diag_embed(variance)
                ).sum()
            )
            last_read = last_read.index_fill(0, indices, float(clock))
        means.append(belief.mean)
        covariances.append(belief.covariance)
    means = torch.stack(means, 1)
    covariances = torch.stack(covariances, 1)
    filtering = gaussian_nll(
        states[..., :2], means[..., :2], covariances[..., :2, :2]
    ).mean()
    measurement_loss = torch.stack(measurements).sum() / mask.sum()
    process_loss = torch.stack(processes).mean()
    reconstructed = model.decode(means)
    reconstruction = ((reconstructed - clean).square() * (1 + 20 * clean)).mean()
    total = (
        filtering + 0.2 * measurement_loss + 0.05 * process_loss + 10 * reconstruction
    )
    return dict(
        total=total,
        filtering=filtering,
        measurement=measurement_loss,
        process=process_loss,
        reconstruction=reconstruction,
    )


@torch.no_grad()
def rollout(
    model: VideoATSModel,
    frames: np.ndarray,
    policy_name: str,
    budget: int,
    seed: int,
    max_gap: int,
    threshold: float = 0.01,
    fixed_q: np.ndarray | None = None,
    fixed_r: np.ndarray | None = None,
) -> dict:
    """Read selected frames only; targets and condition labels are absent by design."""
    length = len(frames)
    model.eval()
    policy = SamplingPolicy(
        "uniform" if policy_name == "all_frames" else policy_name,
        length,
        budget,
        model.transition.cpu().numpy().astype(float),
        model.acceleration_map.cpu().numpy().astype(float),
        seed,
        max_gap,
        threshold,
        fixed_q,
        fixed_r,
    )
    device = model.transition.device
    belief = model.initial_belief(1)
    mean, covariance, q_values = [], [], []
    measurement_values = np.full((length, 2), np.nan, np.float32)
    r_values = np.full((length, 2), np.nan, np.float32)
    acquired = np.zeros(length, dtype=bool)
    next_read, last_read, used = 0, 0, 0
    decisions = []
    policy_seconds = 0.0
    start = time.perf_counter()
    for clock in range(length):
        if clock:
            belief, _ = model.predict(belief)
        if clock == next_read:
            frame = torch.as_tensor(frames[clock : clock + 1], device=device)
            measurement, r = model.encode(frame)
            belief = model.update(
                belief, measurement, r, frame.new_tensor([clock - last_read])
            )
            measurement_values[clock] = measurement[0].cpu().numpy()
            r_values[clock] = r[0].cpu().numpy()
            used += 1
            acquired[clock] = True
            last_read = clock
            policy_start = time.perf_counter()
            next_read, scores = policy.next_time(
                clock,
                used,
                belief.covariance[0].cpu().numpy().astype(float),
                model.process_variance(belief)[0].cpu().numpy().astype(float),
                r[0].cpu().numpy().astype(float),
            )
            policy_seconds += time.perf_counter() - policy_start
            decisions.append(dict(time=clock, next_time=next_read, scores=scores))
        mean.append(belief.mean[0].cpu().numpy())
        covariance.append(belief.covariance[0].cpu().numpy())
        q_values.append(model.process_variance(belief)[0].cpu().numpy())
    if used != budget:
        raise AssertionError(f"Read budget violated: {used} != {budget}.")
    means = np.stack(mean)
    decoded = model.decode(torch.as_tensor(means, device=device)).cpu().numpy()
    return dict(
        mean=means,
        covariance=np.stack(covariance),
        q=np.stack(q_values),
        measurement=measurement_values,
        r=r_values,
        acquired=acquired,
        decoded=decoded,
        decisions=decisions,
        policy_seconds=policy_seconds,
        wall_seconds=time.perf_counter() - start,
    )


def score_rollout(prediction: dict, states: np.ndarray, clean: np.ndarray) -> dict:
    """Score every physical frame, including all intervals without observations."""
    error = prediction["mean"][:, :2].astype(float) - states[:, :2]
    covariance = prediction["covariance"][:, :2, :2].astype(float) + 1e-7 * np.eye(2)
    mahalanobis = np.einsum("ti,tij,tj->t", error, np.linalg.inv(covariance), error)
    sign, logdet = np.linalg.slogdet(covariance)
    if not np.all(sign > 0):
        raise FloatingPointError("Invalid predicted position covariance.")
    nll = 0.5 * (2 * np.log(2 * np.pi) + logdet + mahalanobis)
    observed = prediction["acquired"]
    measurement_error = prediction["measurement"][observed] - states[observed, :2]
    skipped = ~observed
    return dict(
        position_mse=float(np.mean(error**2)),
        position_rmse=float(np.sqrt(np.mean(error**2))),
        skipped_position_mse=(
            float(np.mean(error[skipped] ** 2)) if skipped.any() else None
        ),
        position_nll=float(nll.mean()),
        joint_95_coverage=float(np.mean(mahalanobis <= 5.991464547)),
        clean_image_mse=float(np.mean((prediction["decoded"] - clean) ** 2)),
        measurement_mse=float(np.mean(measurement_error**2)),
        mean_forecast_r=float(prediction["r"][observed].mean()),
        mean_forecast_q=float(prediction["q"].mean()),
        observations=int(observed.sum()),
        frames=int(len(observed)),
        observation_fraction=float(observed.mean()),
        encoder_images=int(observed.sum()),
        decoder_images=int(len(observed)),
        policy_seconds=prediction["policy_seconds"],
        wall_seconds=prediction["wall_seconds"],
    )


@torch.no_grad()
def validation_mse(
    model: VideoATSModel,
    data: VideoData,
    config: VisionConfig,
    policy: str = "uniform",
    threshold: float = 0.01,
) -> float:
    """Use only validation coordinates for checkpoint or threshold selection."""
    values = []
    for index in range(len(data.frames)):
        result = rollout(
            model,
            data.frames[index],
            policy,
            config.observation_budget,
            config.seed + index,
            config.max_gap,
            threshold,
        )
        values.append(
            float(np.mean((result["mean"][:, :2] - data.states[index, :, :2]) ** 2))
        )
    return float(np.mean(values))


@torch.no_grad()
def training_noise_reference(
    model: VideoATSModel, data: VideoData, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ablation constants from training observations with a frozen model."""
    model.eval()
    device = model.transition.device
    all_q, all_r = [], []
    for start in range(0, len(data.frames), batch_size):
        frames = torch.as_tensor(data.frames[start : start + batch_size], device=device)
        belief = model.initial_belief(len(frames))
        for clock in range(frames.shape[1]):
            if clock:
                belief, _ = model.predict(belief)
            measurement, variance = model.encode(frames[:, clock])
            belief = model.update(
                belief,
                measurement,
                variance,
                frames.new_full((len(frames),), float(clock > 0)),
            )
            all_q.append(model.process_variance(belief).cpu().numpy())
            all_r.append(variance.cpu().numpy())
    return np.median(np.concatenate(all_q), axis=0), np.median(
        np.concatenate(all_r), axis=0
    )


def load_checkpoint(path: Path, device: str = "cpu") -> VideoATSModel:
    """Restore an inference model from tensor weights and primitive configuration."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("schema_version") != 1:
        raise ValueError("Unsupported vision checkpoint schema.")
    model = VideoATSModel(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()


def summarize(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """Keep conditions separate and bootstrap paired whole-clip errors vs uniform."""
    summaries, comparisons = [], []
    policies = list(dict.fromkeys(row["policy"] for row in rows))
    conditions = ["all"] + sorted({row["condition"] for row in rows})
    rng = np.random.default_rng(seed)
    for condition in conditions:
        selected = [
            row for row in rows if condition == "all" or row["condition"] == condition
        ]
        baseline = {
            row["clip_id"]: row for row in selected if row["policy"] == "uniform"
        }
        for policy in policies:
            group = [row for row in selected if row["policy"] == policy]
            if not group:
                continue
            summary = dict(
                policy=policy,
                condition=condition,
                clips=len(group),
                observations=group[0]["observations"],
            )
            for key in (
                "position_mse",
                "position_rmse",
                "position_nll",
                "joint_95_coverage",
                "clean_image_mse",
                "measurement_mse",
                "mean_forecast_q",
                "mean_forecast_r",
                "wall_seconds",
                "policy_seconds",
            ):
                summary[key if key.startswith("mean_") else f"mean_{key}"] = float(
                    np.mean([row[key] for row in group])
                )
            known_q = [
                row["true_process_variance"]
                for row in group
                if row["true_process_variance"] is not None
            ]
            summary["mean_true_process_variance"] = (
                float(np.mean(known_q)) if known_q else None
            )
            summaries.append(summary)
            if policy in {"uniform", "all_frames"}:
                continue
            differences = np.asarray(
                [
                    row["position_mse"] - baseline[row["clip_id"]]["position_mse"]
                    for row in group
                ]
            )
            means = differences[
                rng.integers(0, len(differences), (1000, len(differences)))
            ].mean(axis=1)
            low, high = np.quantile(means, [0.025, 0.975])
            comparisons.append(
                dict(
                    policy=policy,
                    condition=condition,
                    clips=len(group),
                    mse_difference_vs_uniform=float(differences.mean()),
                    exploratory_ci95_low=float(low),
                    exploratory_ci95_high=float(high),
                )
            )
    return summaries, comparisons


@record_run_failures
def run_experiment(config: VisionConfig) -> Path:
    """Execute the bounded synthetic pilot or a user-provided labeled NPZ protocol."""
    config.validate()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device(config.device)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable.")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
    output = config.output_dir()
    prepare_run(output, config, "vision_coordinate_ats")
    rng = np.random.default_rng(config.seed)
    split_names = ("train", "validation", "test")
    if config.data_root is None:
        counts = (config.train_clips, config.validation_clips, config.test_clips)
        splits = {
            name: moving_blobs(
                count,
                config.sequence_length,
                config.image_size,
                config.seed + 10000 * offset,
                name,
            )
            for offset, (name, count) in enumerate(zip(split_names, counts))
        }
    else:
        splits = {
            name: load_data(Path(config.data_root) / f"{name}.npz")
            for name in split_names
        }
    validate_splits(splits)
    train, validation, test = (splits[name] for name in split_names)
    if (
        train.frames.shape[1] != config.sequence_length
        or train.frames.shape[-1] != config.image_size
    ):
        raise ValueError(
            "Set sequence_length and image_size to match the labeled NPZ files."
        )
    data_dir = output / "data"
    data_dir.mkdir()
    identities = {}
    for name, data in splits.items():
        destination = data_dir / f"{name}.npz"
        data.save(destination)
        identities[name] = dict(
            clips=len(data.frames),
            sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        )
    model_config = dict(
        image_size=config.image_size,
        channels=train.frames.shape[2],
        hidden_size=config.hidden_size,
    )
    model = VideoATSModel(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    logs = []
    flat_frames = train.frames.reshape(-1, *train.frames.shape[2:])
    flat_clean = train.clean_frames.reshape(flat_frames.shape)
    flat_states = train.states.reshape(-1, 4)
    for epoch in range(config.warmup_epochs):
        model.train()
        losses = []
        for indices in np.array_split(
            rng.permutation(len(flat_frames)),
            max(1, int(np.ceil(len(flat_frames) / (8 * config.batch_size)))),
        ):
            frames = torch.as_tensor(flat_frames[indices], device=device)
            labels = torch.as_tensor(flat_states[indices], device=device)
            clean = torch.as_tensor(flat_clean[indices], device=device)
            measurement, variance = model.encode(frames)
            reconstruction = model.decode(labels)
            loss = gaussian_nll(
                labels[:, :2], measurement, torch.diag_embed(variance)
            ).mean()
            loss = (
                loss
                + 10 * ((reconstruction - clean).square() * (1 + 20 * clean)).mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        logs.append(
            dict(
                phase="warmup",
                epoch=epoch + 1,
                training_loss=float(np.mean(losses)),
                validation_position_mse=None,
            )
        )
    best_mse = float("inf")
    best_epoch = 0
    for epoch in range(config.epochs):
        model.train()
        losses = []
        for indices in np.array_split(
            rng.permutation(len(train.frames)),
            max(1, int(np.ceil(len(train.frames) / config.batch_size))),
        ):
            frames = torch.as_tensor(train.frames[indices], device=device)
            clean = torch.as_tensor(train.clean_frames[indices], device=device)
            states = torch.as_tensor(train.states[indices], device=device)
            mask = torch.as_tensor(
                training_mask(
                    len(indices), config.sequence_length, config.observation_budget, rng
                ),
                device=device,
            )
            loss_terms = sequence_loss(model, frames, clean, states, mask)
            optimizer.zero_grad(set_to_none=True)
            loss_terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(float(loss_terms["total"].detach()))
        selected_mse = validation_mse(model, validation, config)
        if not np.isfinite(selected_mse):
            raise FloatingPointError("Validation error is nonfinite.")
        logs.append(
            dict(
                phase="joint",
                epoch=epoch + 1,
                training_loss=float(np.mean(losses)),
                validation_position_mse=selected_mse,
            )
        )
        print(
            f"epoch={epoch + 1}/{config.epochs} validation_position_mse={selected_mse:.6f}",
            flush=True,
        )
        if selected_mse < best_mse:
            best_mse, best_epoch = selected_mse, epoch + 1
            state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            torch.save(
                dict(
                    schema_version=1,
                    model_config=model_config,
                    state_dict=state,
                    selected_epoch=best_epoch,
                    validation_position_mse=best_mse,
                ),
                output / "model.pt",
            )
    model = load_checkpoint(output / "model.pt", config.device)
    thresholds = (0.002, 0.01, 0.05, 0.2, 1.0)
    threshold_scores = {
        str(value): validation_mse(
            model, validation, config, "total_uncertainty", value
        )
        for value in thresholds
    }
    threshold = min(thresholds, key=lambda value: (threshold_scores[str(value)], value))
    fixed_q, fixed_r = training_noise_reference(model, train, config.batch_size)
    write_csv(output / "training.csv", logs)
    protocol = dict(
        schema_version=1,
        evidence=(
            "synthetic implementation pilot"
            if config.data_root is None
            else "labeled tracking benchmark"
        ),
        model_config=model_config,
        state_units="normalized image coordinates; velocity per frame",
        q_meaning="per-frame acceleration variance under fixed damped spring dynamics",
        r_meaning="effective encoder coordinate error variance; not raw pixel noise",
        acquisition="receding-horizon integrated position variance with gap-capped uniform continuation; frozen Q/R forecasts within each call",
        relation_to_scalar_ats="task-risk adaptation; not parameter information gain over the original Q/R grid",
        primary_metric="mean clip position MSE over every physical frame; lower is better",
        split_roles=dict(
            train="gradient fitting and ablation constants",
            validation="checkpoint and total-uncertainty threshold",
            test="frozen final scoring only",
        ),
        data=identities,
        selected_epoch=best_epoch,
        validation_position_mse=best_mse,
        total_uncertainty_threshold=threshold,
        validation_threshold_scores=threshold_scores,
        fixed_q=fixed_q.tolist(),
        fixed_r=fixed_r.tolist(),
        budget="first frame counts; exact acquisition count per clip; all_frames is unmatched",
        compute="same checkpoint and acquisition budget; policy runtime differs and is reported; no equal-compute claim",
        noise_forecast="R of last observed image forecasts unread images; Q uses causal observed history",
        limitations=[
            "coordinate and clean-image supervision",
            "fixed known linear motion mean",
            "diagonal acceleration and measurement noise",
            "no guarantee of Q/R identification or real-video performance",
            "bootstrap intervals are exploratory with no multiplicity correction",
        ],
    )
    write_json(output / "protocol.json", protocol)
    rows, traces, arrays = [], [], {}
    first_decoded = None
    policies = (
        "uniform",
        "random",
        "total_uncertainty",
        "ats",
        "ats_fixed_q",
        "ats_fixed_r",
        "all_frames",
    )
    for policy_name in policies:
        policy_arrays = {
            key: []
            for key in ("mean", "covariance", "q", "measurement", "r", "acquired")
        }
        for index in range(len(test.frames)):
            budget = (
                config.sequence_length
                if policy_name == "all_frames"
                else config.observation_budget
            )
            prediction = rollout(
                model,
                test.frames[index],
                policy_name,
                budget,
                config.seed + 20000 + index,
                config.max_gap,
                threshold,
                fixed_q,
                fixed_r,
            )
            metrics = score_rollout(
                prediction, test.states[index], test.clean_frames[index]
            )
            if policy_name == "ats" and index == 0:
                first_decoded = prediction["decoded"]
            row = dict(
                policy=policy_name,
                clip_id=str(test.clip_ids[index]),
                condition=str(test.conditions[index]),
                **metrics,
            )
            row["true_process_variance"] = (
                float(test.process_variances[index].mean())
                if test.process_variances is not None
                else None
            )
            rows.append(row)
            traces.append(
                dict(
                    policy=policy_name,
                    clip_id=str(test.clip_ids[index]),
                    decisions=prediction["decisions"],
                )
            )
            for key in policy_arrays:
                policy_arrays[key].append(prediction[key])
        for key, values in policy_arrays.items():
            arrays[f"{policy_name}__{key}"] = np.stack(values)
        print(
            f"evaluated policy={policy_name} clips={len(test.frames)} reads={budget}",
            flush=True,
        )
    summaries, comparisons = summarize(rows, config.seed + 40000)
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "paired_comparisons.csv", comparisons)
    write_json(output / "decisions.json", traces)
    np.savez_compressed(output / "predictions.npz", **arrays)
    plot_results(output, summaries, test, arrays, first_decoded)
    complete_run(
        output,
        selected_epoch=best_epoch,
        model_parameters=sum(p.numel() for p in model.parameters()),
        test_clips=len(test.frames),
        policies=list(policies),
        evidence=protocol["evidence"],
    )
    return output
