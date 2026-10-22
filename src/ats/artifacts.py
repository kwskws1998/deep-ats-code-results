"""Run provenance and protection against silently replacing scientific results."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
import os
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path


def atomic_write_json(path: Path, value) -> None:
    """Publish a complete single-writer JSON record or preserve the previous record."""
    path = Path(path)
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(encoded)
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def json_value(value):
    """Convert configuration objects into portable JSON values."""
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def source_fingerprints(project_root: Path) -> dict[str, str]:
    """Identify implementation, entrypoint, test, and dependency-specification sources."""
    paths = [project_root / "pyproject.toml"]
    for directory in ("src", "scripts", "tests"):
        paths.extend((project_root / directory).rglob("*.py"))
    return {
        str(path.relative_to(project_root)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(paths)
        if path.is_file()
    }


def validate_resume_identity(
    manifest, config, current_sources, recorded_raw, current_raw
):
    """Reject mixed checkpoints after changes to code, raw inputs, fitting settings, or core dependencies."""
    if manifest.get("status") not in {"running", "failed"}:
        raise ValueError("Resume requires an unfinished run.")
    if (
        manifest.get("config") != config
        or manifest.get("source_sha256") != current_sources
    ):
        raise ValueError("Resume requires the original configuration and source files.")
    expected = {
        k: v["sha256"] for k, v in recorded_raw.items() if k.startswith("experiment")
    }
    actual = {
        k: v["sha256"] for k, v in current_raw.items() if k.startswith("experiment")
    }
    if not expected or expected != actual:
        raise ValueError("Raw data changed since the checkpoint run.")
    for name in ("numpy", "scipy", "scikit-learn"):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        if manifest.get("dependencies", {}).get(name) != version:
            raise ValueError(
                f"Core dependency changed since the checkpoint run: {name}."
            )


def record_run_failures(function):
    """Record failures of newly created runs without changing preexisting results."""

    @wraps(function)
    def wrapped(config, *args, **kwargs):
        path = config.output_dir() / "manifest.json"
        existed = path.exists()
        try:
            return function(config, *args, **kwargs)
        except Exception as error:
            if not existed and path.exists():
                manifest = json.loads(path.read_text())
                manifest.update(
                    status="failed", error_type=type(error).__name__, error=str(error)
                )
                atomic_write_json(path, manifest)
            raise

    return wrapped


def prepare_run(output_dir: Path, config, kind: str) -> None:
    """Require a fresh output directory and persist configuration and source identity."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(path.name != ".DS_Store" for path in output_dir.iterdir()):
        raise FileExistsError(
            f"Run directory is not empty: {output_dir}. Choose a new run name."
        )
    project_root = Path(__file__).resolve().parents[2]
    versions = {}
    for name in (
        "numpy",
        "scipy",
        "scikit-learn",
        "threadpoolctl",
        "matplotlib",
        "gymnasium",
        "torch",
        "Pillow",
        "opencv-python",
        "tapnet",
        "pyhgf",
        "jax",
        "dm-control",
        "realworldrl-suite",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    manifest = {
        "schema_version": 3,
        "kind": kind,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": json_value(config),
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": versions,
        "source_root": str(project_root),
        "source_sha256": source_fingerprints(project_root),
    }
    spec = importlib.util.find_spec("pyhgf")
    if spec is not None and spec.origin is not None:
        dependency_root = Path(spec.origin).parent
        manifest["dependency_source_sha256"] = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                dependency_root / "model/hgf.py",
                dependency_root / "model/network.py",
                dependency_root / "updates/prediction/continuous.py",
            )
            if path.is_file()
        }
    atomic_write_json(output_dir / "manifest.json", manifest)


def complete_run(output_dir: Path, **metadata) -> None:
    """Mark a successful run and fingerprint the produced data artifacts."""
    path = Path(output_dir) / "manifest.json"
    manifest = json.loads(path.read_text())
    if source_fingerprints(Path(manifest["source_root"])) != manifest["source_sha256"]:
        raise RuntimeError(
            "Source files changed during this run; results cannot be marked complete."
        )
    manifest.update(json_value(metadata))
    manifest["status"] = "completed"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["artifact_sha256"] = {
        str(item.relative_to(output_dir)): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(Path(output_dir).rglob("*"))
        if item.is_file() and item != path
    }
    atomic_write_json(path, manifest)


def verify_run(output_dir: Path, check_current_source: bool = False) -> list[str]:
    """Return explicit integrity failures for one completed run."""
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text())
    failures = []
    if manifest.get("status") != "completed":
        failures.append(f"Run status is {manifest.get('status')}, not completed.")
    expected = manifest.get("artifact_sha256", {})
    actual_paths = {
        str(path.relative_to(output_dir))
        for path in output_dir.rglob("*")
        if path.is_file() and path != output_dir / "manifest.json"
    }
    for relative, digest in expected.items():
        path = output_dir / relative
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            failures.append(f"Artifact missing or changed: {relative}")
    if actual_paths - set(expected):
        failures.append(f"Unrecorded artifacts: {sorted(actual_paths - set(expected))}")
    if check_current_source:
        if (
            "source_root" not in manifest
            or source_fingerprints(Path(manifest["source_root"]))
            != manifest["source_sha256"]
        ):
            failures.append("Current project source does not match the run.")
        for relative, digest in manifest.get("dependency_source_sha256", {}).items():
            path = Path(relative)
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            ):
                failures.append(
                    f"Current dependency source missing or changed: {relative}"
                )
    return failures


def register_derived_figures(output_dir: Path) -> None:
    """Record regenerated figures separately while preserving the original data provenance."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "completed":
        return
    for relative, digest in manifest.get("artifact_sha256", {}).items():
        if Path(relative).parts[0] == "figures":
            continue
        path = output_dir / relative
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise RuntimeError(
                f"Cannot register figures after a data artifact changed: {relative}"
            )
    hashes = {
        key: value
        for key, value in manifest.get("artifact_sha256", {}).items()
        if Path(key).parts[0] != "figures"
    }
    hashes.update(
        {
            str(path.relative_to(output_dir)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted((output_dir / "figures").rglob("*"))
            if path.is_file()
        }
    )
    manifest["artifact_sha256"] = hashes
    manifest["figure_source_sha256"] = source_fingerprints(
        Path(__file__).resolve().parents[2]
    )
    manifest["figures_generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(manifest_path, manifest)
