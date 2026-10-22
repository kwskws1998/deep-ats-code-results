"""Post-prediction diagnosis, explicit oracle controls, and final-test isolation."""

import csv
import json

import numpy as np
import pytest
import torch

from ats.artifacts import verify_run
from ats_vision.tap_data import sha256
from ats_vision.tap_diagnostics import full_observation_trace
from ats_vision.tap_failure_audit import (
    audited_query,
    encoder_geometry,
    extra_component_rows,
    failure_audit,
    gradient_conflicts,
    heatmap_frame_statistics,
    mechanism_controls,
    sampled_data_profile,
)
from ats_vision.tap_model import TAPATSModel
from ats_vision.tap_training import TAPTrainConfig, train_tap
from test_tapvid import fixture_clip, create_manifest


@pytest.fixture(autouse=True)
def single_thread():
    """Keep generated fixtures bounded on the host CPU."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_constructed_multipeak_control_is_an_explicit_mechanism_example():
    record = mechanism_controls()
    assert record["global_mean_error_px"] == 64
    assert record["peak_error_px"] == record["local_mean_error_px"] == 0
    assert "constructed" in record["kind"]
    assert "not its frequency" in record["limitation"]


def test_encoder_geometry_and_cell_probability_match_direct_calculations():
    assert encoder_geometry(TAPATSModel(8, 8))["receptive_field_px"] == 17
    logits = torch.arange(16.0).reshape(1, 4, 4)
    result = heatmap_frame_statistics(logits, np.array([0.9, 0.9]))
    assert result["target_cell_rank"] == 1
    assert result["target_cell_is_peak"]
    assert result["target_cell_cross_entropy"] == pytest.approx(
        float(torch.nn.functional.cross_entropy(logits.flatten(1), torch.tensor([15]))),
        abs=1e-6,
    )
    assert heatmap_frame_statistics(logits)["target_cell_rank"] is None


def test_reducer_does_not_change_predictions_weights_rng_or_parameter_gradients():
    torch.manual_seed(7)
    model = TAPATSModel(8, 8)
    clip = fixture_clip()
    ids, queries = clip.queries()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    expected = full_observation_trace(model, clip.frames, queries[0])
    actual, frames, identity = audited_query(model, clip, int(ids[0]), queries[0])
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual[key])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(before[key], value, atol=0, rtol=0)
    torch.testing.assert_close(rng, torch.get_rng_state(), atol=0, rtol=0)
    assert all(parameter.grad is None for parameter in model.parameters())

    partial = gradient_conflicts(model, clip, int(ids[0]), queries[0], 6, fraction=0.5)
    assert (
        partial["per_module_gradient_norms"]["visibility"]["forecast_visibility_head"]
        > 0
    )
    assert len(frames) == len(clip.frames) - 1
    assert "self_peak_error_px" in identity
    profile = sampled_data_profile(clip, ids, queries)
    assert profile["queries"] == 3
    assert 0 < profile["future_visible_fraction"] <= 1


def test_gradient_probe_reports_weighted_objective_without_weight_updates():
    model = TAPATSModel(8, 8)
    clip = fixture_clip()
    ids, queries = clip.queries()
    before = {name: parameter.clone() for name, parameter in model.state_dict().items()}
    result = gradient_conflicts(model, clip, int(ids[0]), queries[0], 6)
    assert result["defined"]
    assert result["total_unclipped_gradient_norm"] > 0
    assert len(result["pairwise_cosines"]) == 6
    assert (
        next(row for row in result["components"] if row["component"] == "measurement")[
            "objective_weight"
        ]
        == 0.1
    )
    for pair in result["pairwise_cosines"]:
        if pair["cosine"] is not None:
            assert -1.000001 <= pair["cosine"] <= 1.000001
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_label_control_uses_actual_feature_grid_and_preserves_query_exclusion():
    model = TAPATSModel(8, 8)
    clip = fixture_clip()
    ids, queries = clip.queries()
    rows = [full_observation_trace(model, clip.frames, query) for query in queries]
    traces = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    result = extra_component_rows(clip, ids, queries, traces, 4)
    oracle = next(row for row in result if row["condition"].startswith("oracle_label"))
    assert oracle["value"] == "4x4"
    assert any(row["condition"] == "within_training_max_horizon" for row in result)
    assert any(row["condition"] == "beyond_training_max_horizon" for row in result)


def test_failure_audit_never_reads_final_test_and_preserves_parent(
    tmp_path, monkeypatch
):
    import ats_vision.tap_failure_audit as audit
    import ats_vision.tap_data as data

    manifest = create_manifest(tmp_path / "data")
    config = TAPTrainConfig(
        manifest,
        tmp_path / "fit",
        steps=1,
        queries_per_step=1,
        validation_every=1,
        validation_queries=1,
        validation_clips=1,
        feature_dim=8,
        hidden_size=8,
        threads=1,
    )
    train_tap(config)
    original = {p.name: sha256(p) for p in config.output_root.iterdir() if p.is_file()}
    real = sha256

    def no_test(path):
        """Reject even checksum access to the final-test archive."""
        assert str(path).split("/")[-1] != "test.npz"
        return real(path)

    monkeypatch.setattr(audit, "sha256", no_test)
    monkeypatch.setattr(data, "sha256", no_test)
    root = tmp_path / "audit"
    failure_audit(
        manifest,
        config.output_root / "last.pt",
        root,
        max_clips=1,
        max_queries=3,
        include_lk=False,
        threads=1,
    )
    summary = json.loads((root / "audit_summary.json").read_text())
    assert not summary["final_test_accessed"]
    assert not summary["weights_changed"]
    assert not summary["cause_proven"]
    with (root / "sampled_data.csv").open() as stream:
        assert {row["split"] for row in csv.DictReader(stream)} == {
            "train",
            "validation",
        }
    assert (root / "diagnostic_snapshot.pt").exists()
    assert verify_run(root, check_current_source=True) == []
    for name, digest in original.items():
        assert real(config.output_root / name) == digest
