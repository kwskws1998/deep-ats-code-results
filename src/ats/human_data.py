"""Read public behavioral records without importing fitted author estimates or demographic variables."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class BucketParticipant:
    """Condition-aligned continuous predictions; outcomes occur after each prediction."""

    participant: str
    experiment: int
    responses: np.ndarray
    outcomes: np.ndarray
    presentation_order: np.ndarray


@dataclass(frozen=True)
class ObserveBlock:
    """One 50-decision block; -2 denotes a missed response and 0 an observe choice."""

    participant: str
    session: str
    block: str
    choices: np.ndarray
    feedback: np.ndarray


def sha256_file(path: Path) -> str:
    """Identify raw data independently of a mutable filesystem timestamp."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bucket_participants(root: Path) -> tuple[list[BucketParticipant], dict]:
    """Use the authors' condition-aligned columns and retain source trial order without exclusions."""
    participants, metadata = [], {}
    for experiment in (1, 2):
        path = root / "unpacked" / f"experiment{experiment}" / "data.mat"
        content = loadmat(path, simplify_cells=True)
        outcomes = np.asarray(content["meta_data"]["bag"], dtype=float)
        if outcomes.shape != (50, 4) or not np.isfinite(outcomes).all():
            raise ValueError("Expected 50 outcomes in four condition-aligned columns.")
        entries = content["data"]
        for index, entry in enumerate(entries):
            responses = np.asarray(entry["bucket"], dtype=float)
            order = np.asarray(entry["randomization_order"], dtype=int)
            if responses.shape != outcomes.shape or not np.isfinite(responses).all():
                raise ValueError("Invalid or missing bucket predictions.")
            if np.any((responses < 10) | (responses > 90)) or np.any(responses % 2):
                raise ValueError(
                    "Responses do not follow the released two-unit bucket grid."
                )
            if sorted(order.tolist()) != [0, 1, 2, 3]:
                raise ValueError("Invalid presentation order.")
            participants.append(
                BucketParticipant(
                    f"bucket_e{experiment}_{index + 1:04d}",
                    experiment,
                    responses,
                    outcomes.copy(),
                    order,
                )
            )
        metadata[f"experiment{experiment}"] = {
            "raw_file": str(path),
            "sha256": sha256_file(path),
            "participants": len(entries),
            "trials_per_participant": 200,
            "outcomes_shared_by_participants": True,
            "true_volatility_for_descriptive_labels_only": np.asarray(
                content["meta_data"]["true_vol"]
            ).tolist(),
            "true_stochasticity_for_descriptive_labels_only": np.asarray(
                content["meta_data"]["true_sto"]
            ).tolist(),
        }
    metadata["shared_stimuli_across_experiments"] = np.array_equal(
        participants[0].outcomes,
        next(p.outcomes for p in participants if p.experiment == 2),
    )
    return participants, metadata


def load_observe_blocks(root: Path) -> tuple[list[ObserveBlock], dict]:
    """Keep behavioral training and scanner test files separate and mask hidden or missing outcomes."""
    directories = list((root / "unpacked").glob("ObserveBet-*"))
    if len(directories) != 1:
        raise ValueError("Expected one pinned ObserveBet repository.")
    records, raw_files = [], {}
    for directory in sorted((directories[0] / "TaskData").glob("subj*")):
        for path in sorted(directory.glob("behav.*.csv")):
            if re.fullmatch(r"behav\.(?:behav)?\d{3}\.csv", path.name) is None:
                raise ValueError(
                    f"Unrecognized behavioral session filename: {path.name}"
                )
            values = np.loadtxt(path, delimiter=",")
            if (
                values.shape != (50, 2)
                or not np.isfinite(values).all()
                or np.any(values != values.astype(int))
            ):
                raise ValueError(f"Invalid behavioral block: {path}")
            choice, feedback = values.T.astype(int)
            if not set(choice) <= {-2, -1, 0, 1} or not set(feedback) <= {-1, 0, 1}:
                raise ValueError("Unexpected response coding.")
            if np.any(feedback[choice != 0] != 0) or np.any(feedback[choice == 0] == 0):
                raise ValueError(
                    "Feedback is inconsistent with the chosen observation action."
                )
            session = "train" if path.name.startswith("behav.behav") else "test"
            records.append(
                ObserveBlock(directory.name, session, path.stem, choice, feedback)
            )
            raw_files[str(path)] = sha256_file(path)
    if not records:
        raise ValueError("No public observe-or-bet blocks found.")
    subjects = sorted({row.participant for row in records})
    for participant in subjects:
        if {r.session for r in records if r.participant == participant} != {
            "train",
            "test",
        }:
            raise ValueError(
                "Each participant needs disjoint training and test sessions."
            )
    return records, {
        "participants": len(subjects),
        "blocks": len(records),
        "raw_sha256": raw_files,
        "train_trials": sum(len(r.choices) for r in records if r.session == "train"),
        "test_trials": sum(len(r.choices) for r in records if r.session == "test"),
        "missed_trials": sum(int(np.sum(r.choices == -2)) for r in records),
        "feedback_rule": "Only observed outcomes enter inference; misses advance the task clock without feedback.",
    }
