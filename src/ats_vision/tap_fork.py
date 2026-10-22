"""Explicit continuation into a fresh run after changing execution efficiency.

Exact resume still rejects any changed source/configuration. A fork preserves
weights, Adam state, RNG, completed steps and the LR schedule, but records that
floating-point execution changed and never overwrites the parent run.
"""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
from pathlib import Path

import torch
import numpy as np

from ats.artifacts import prepare_run
from ats_vision.experiment import write_json
from ats_vision.tap_checkpoint import load_diagnostic_checkpoint


def prepare_training_fork(config, identity, source):
    """Validate legacy/current state before publishing a new resumable run."""
    source = Path(source).resolve()
    output = Path(config.output_root).resolve()
    if output == source.parent or source.is_relative_to(output):
        raise ValueError("A fork requires a separate output directory.")
    _, saved, provenance = load_diagnostic_checkpoint(source, config.data_manifest)
    required = {
        "optimizer",
        "numpy_rng",
        "torch_rng",
        "cuda_rng",
        "step",
        "best_step",
        "best_key",
        "logs",
    }
    if not required <= saved.keys() or not 0 < saved["step"] <= config.steps:
        raise ValueError(
            "Fork requires a resumable last.pt within the original step budget."
        )
    original = saved["identity"]["config"]
    current = asdict(config)
    execution_only = {
        "output_root",
        "data_manifest",
        "max_hours",
        "threads",
        "data_cache_clips",
        "query_batch_size",
        "execution_backend",
        "encoder_chunk_size",
        "validation_query_batch_size",
    }
    for key, value in original.items():
        if key not in execution_only and current.get(key) != value:
            raise ValueError(f"Fork cannot change scientific training setting: {key}")
    if saved["identity"].get("torch_version") != str(torch.__version__):
        raise ValueError(
            "Fork requires the original PyTorch version for optimizer/RNG compatibility."
        )
    if saved["identity"].get("numpy_version") != str(np.__version__):
        raise ValueError(
            "Fork requires the original NumPy version for RNG compatibility."
        )
    best = saved.get("best_snapshot")
    best_bytes = None
    if best is None:
        best_path = source.parent / "model.pt"
        best_bytes = best_path.read_bytes()
        best = torch.load(io.BytesIO(best_bytes), map_location="cpu", weights_only=True)
    if (
        best.get("selected_step") != saved["best_step"]
        or best.get("identity") != saved["identity"]
        or best.get("model_config") != provenance["model_config"]
    ):
        raise ValueError(
            "Parent best/last disagree; stop the parent and use a consistent snapshot."
        )
    if config.device == "cuda" and len(saved["cuda_rng"]) != torch.cuda.device_count():
        raise ValueError("Fork requires the same visible CUDA device count.")
    saved = deepcopy(saved)
    best = deepcopy(best)
    for row in saved["logs"]:
        row.setdefault(
            "execution_backend", original.get("execution_backend", "reference")
        )
    record = dict(
        **provenance,
        parent_identity=saved["identity"],
        inherited_step=saved["step"],
        inherited_best_step=saved["best_step"],
        inherited_optimizer_and_rng=True,
        exact_replay=False,
        execution_changes={
            key: dict(
                before=original.get(key),
                after=(
                    str(current[key])
                    if isinstance(current[key], Path)
                    else current[key]
                ),
            )
            for key in execution_only
            if key in current and key != "data_manifest"
        },
        legacy_best_sha256=(
            hashlib.sha256(best_bytes).hexdigest() if best_bytes else None
        ),
    )
    best.update(identity=identity, selection_complete=False, fork_provenance=record)
    best.pop("selection", None)
    saved.update(
        schema_version=1,
        kind="tap_ats_training_snapshot",
        identity=identity,
        model_config=provenance["model_config"],
        best_snapshot=best,
        fork_provenance=record,
    )
    prepare_run(output, config, "tapvid_training_execution_fork")
    write_json(output / "training_fork.json", record)
    from ats_vision.tap_training import atomic_checkpoint

    atomic_checkpoint(output / "model.pt", best)
    atomic_checkpoint(output / "last.pt", saved)
    return record
