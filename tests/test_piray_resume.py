"""Prevent reusing scientific checkpoints after material inputs or dependencies change."""

from copy import deepcopy
import importlib.metadata

import pytest

from ats.artifacts import validate_resume_identity


@pytest.mark.parametrize(
    "changed", ["none", "status", "configuration", "source", "raw", "dependency"]
)
def test_resume_identity_requires_the_original_scientific_inputs(changed):
    config = {"particles": 10000}
    sources = {"src/model.py": "abc"}
    raw = {"experiment1": {"sha256": "raw_1"}, "experiment2": {"sha256": "raw_2"}}
    recorded = deepcopy(raw)
    manifest = {
        "status": "failed",
        "config": config.copy(),
        "source_sha256": sources.copy(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "scipy", "scikit-learn"]
        },
    }
    if changed == "status":
        manifest["status"] = "completed"
    elif changed == "configuration":
        manifest["config"]["particles"] = 500
    elif changed == "source":
        manifest["source_sha256"]["src/model.py"] = "other"
    elif changed == "raw":
        raw["experiment1"]["sha256"] = "other"
    elif changed == "dependency":
        manifest["dependencies"]["scipy"] = "other"
    if changed == "none":
        validate_resume_identity(manifest, config, sources, recorded, raw)
    else:
        with pytest.raises(ValueError):
            validate_resume_identity(manifest, config, sources, recorded, raw)
