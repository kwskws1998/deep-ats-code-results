"""Immutable checkpoint snapshots for explicitly labeled, source-checked diagnostics."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import torch

from ats_vision.tap_data import sha256
from ats_vision.tap_model import TAPATSModel

OPERATOR_SOURCES = ("src/ats_vision/tap_model.py", "src/ats_vision/model.py")


def load_diagnostic_checkpoint(path: Path, data_manifest: Path, device="cpu"):
    """Accept selected or resumable checkpoints only when executed model sources match.

    The full fitting source inventory may differ in a separate diagnostic
    checkout. This exception is restricted to diagnostics, never final testing
    or resume. Read bytes once so a live atomic replacement cannot mix snapshots.
    """
    snapshot = Path(path).read_bytes()
    saved = torch.load(io.BytesIO(snapshot), map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or not {"identity", "state_dict"} <= saved.keys():
        raise ValueError("Unsupported diagnostic checkpoint.")
    kind = saved.get("kind")
    if kind not in {
        None,
        "tap_ats",
        "tap_ats_training_snapshot",
        "tap_ats_diagnostic_snapshot",
    }:
        raise ValueError("Unsupported diagnostic checkpoint kind.")
    if kind is None and not {"optimizer", "step", "logs"} <= saved.keys():
        raise ValueError(
            "Untyped checkpoint is not a recognized legacy training snapshot."
        )
    identity = saved["identity"]
    if identity["data_manifest_sha256"] != sha256(data_manifest):
        raise ValueError(
            "Diagnostics require the dataset manifest frozen during fitting."
        )
    project = Path(__file__).resolve().parents[2]
    for relative in OPERATOR_SOURCES:
        if identity["sources"].get(relative) != sha256(project / relative):
            raise ValueError(f"Model operator source changed: {relative}")
    backend = identity["config"].get("execution_backend", "reference")
    if backend == "shared":
        for relative in (
            "src/ats_vision/tap_fast_model.py",
            "src/ats_vision/tap_fast_math.py",
        ):
            if identity["sources"].get(relative) != sha256(project / relative):
                raise ValueError(f"Shared operator source changed: {relative}")
    config = saved.get("model_config") or {
        key: identity["config"][key] for key in ("feature_dim", "hidden_size")
    }
    if backend == "shared":
        from ats_vision.tap_fast_model import SharedTAPATSModel

        model = SharedTAPATSModel(**config).to(device)
    elif backend == "reference":
        model = TAPATSModel(**config).to(device)
    else:
        raise ValueError("Unknown checkpoint execution backend.")
    model.load_state_dict(saved["state_dict"], strict=True)
    provenance = dict(
        checkpoint=str(Path(path).resolve()),
        checkpoint_sha256=hashlib.sha256(snapshot).hexdigest(),
        checkpoint_kind=kind or "legacy_training_snapshot",
        step=saved.get("selected_step", saved.get("step")),
        selection_complete=bool(saved.get("selection_complete", False)),
        compatibility="model operator sources match; separate diagnostic code is not exact fitting-run replay",
        fitting_torch_version=identity.get("torch_version"),
        diagnostic_torch_version=str(torch.__version__),
        model_config=config,
        fitting_sources=identity["sources"],
    )
    return model.eval(), saved, provenance
