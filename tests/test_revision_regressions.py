"""Regression checks for scientific scoring, temporal budgets, HGF, and RL boundaries."""

import csv
import json
from dataclasses import replace
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from ats.artifacts import (
    prepare_run,
    complete_run,
    verify_run,
    register_derived_figures,
)
from ats.beliefs import GridBelief
from ats.config import AgentConfig, GridConfig, SimulationConfig, default_conditions
from ats.envs import TemporalSamplingEnv
from ats.experiments import (
    run_episode,
    run_experiment,
    save_example_posterior,
    stable_seed,
)
from ats.models import GridPerceptualModel, PyHGFContinuousModel, normal_nll
from ats.policies import active_temporal_sampling_scores, sampling_cost
from deep_ats_planner.deep_ppo import (
    ActorCritic,
    PPOConfig,
    generalized_advantage_estimate,
    make_encoder,
)
from deep_ats_planner.config import DeepPlannerConfig
from deep_ats_planner.planner import bellman_target
from deep_ats_planner.rwrl_wrapper import RWRLGymWrapper
from deep_ats_planner.training import transition_return


def test_packages_use_the_same_belief_and_model_classes():
    from deep_ats_planner.beliefs import GridBelief as DeepBelief
    from deep_ats_planner.models import PyHGFContinuousModel as DeepHGF

    assert DeepBelief is GridBelief
    assert DeepHGF is PyHGFContinuousModel


@pytest.mark.parametrize("dt", [1, 2, 4, 8])
def test_hgf_zero_forecast_error_preserves_latent_mean(dt):
    model = PyHGFContinuousModel(initial_volatility=4.0)
    model.reset(1.0)
    assert model.estimate().volatility == pytest.approx(4.0)
    before = model._state()
    prediction = model.predict(dt)
    assert model._state() is before
    assert prediction.mean == pytest.approx(1.0)
    model.update(1.0, dt)
    assert float(model._state()[0]["expected_mean"]) == pytest.approx(prediction.mean)
    assert float(model._state()[1]["mean"]) == pytest.approx(1.0)


def test_hgf_long_gap_matches_explicit_missing_unit_observations():
    model = PyHGFContinuousModel()
    model.reset(0.3)
    direct = PyHGFContinuousModel()
    direct.reset(0.3)
    direct.model.input_data(
        np.array([0.0, 0.0, 0.0, -0.5]),
        time_steps=np.ones(4),
        observed=np.array([0, 0, 0, 1]),
        record_trajectories=False,
    )
    model.update(-0.5, 4)
    for level in (1, 2, 3):
        for key in ("mean", "precision"):
            assert float(model._state()[level][key]) == pytest.approx(
                float(direct._state()[level][key]), rel=1e-5
            )


def test_hgf_invalid_states_raise_instead_of_being_clipped():
    model = PyHGFContinuousModel()
    model.reset(0.0)
    model.model.attributes[1]["precision"] = -1.0
    with pytest.raises(FloatingPointError):
        model.estimate()


def test_exact_predictive_mixture_matches_manually_weighted_densities():
    model = GridPerceptualModel("test", np.array([0.1, 4.0]), np.array([0.2, 9.0]), 1.0)
    model.reset(0.0)
    y, dt = 5.0, 4
    variance = model.belief.predictive_variance_by_hypothesis(dt)
    density = np.sum(
        model.belief.posterior
        * np.exp(-0.5 * y**2 / variance)
        / np.sqrt(2 * np.pi * variance)
    )
    assert model.negative_log_likelihood(y, dt) == pytest.approx(-np.log(density))
    prediction = model.predict(dt)
    assert (
        abs(normal_nll(y, prediction.mean, prediction.variance) + np.log(density))
        > 0.05
    )


def test_logged_variance_and_nll_use_selected_horizon_before_update():
    config = SimulationConfig(episodes=1, steps=1, grid=GridConfig(size=5))
    condition = default_conditions()[0]
    row = run_episode(condition, "fixed_long", config, 0, 17)[0]
    environment = TemporalSamplingEnv(
        condition, np.random.default_rng(stable_seed(17, "env")), max_time=8
    )
    v, s = config.grid.make_grids()
    belief = GridBelief(v, s, config.agent.initial_variance)
    belief.reset(environment.reset().observation)
    assert row["dt"] == 8
    assert row["predictive_variance"] == pytest.approx(
        belief.mixture_predictive_variance(8)
    )
    assert row["negative_log_likelihood"] == pytest.approx(
        -belief.predictive_log_density(row["observation"], 8)
    )


def test_common_physical_trajectory_is_independent_of_observation_schedule():
    condition = default_conditions()[2]
    short = TemporalSamplingEnv(condition, np.random.default_rng(18), max_time=8)
    long = TemporalSamplingEnv(condition, np.random.default_rng(18), max_time=8)
    assert short.reset() == long.reset()
    for _ in range(8):
        result_short = short.step(1)
    result_long = long.step(8)
    assert result_short.latent_state == result_long.latent_state
    assert result_short.observation == result_long.observation


@pytest.mark.parametrize(
    "agent",
    [
        "active_temporal",
        "fixed_short",
        "fixed_long",
        "fixed_dt_2",
        "fixed_dt_4",
        "passive_grid",
        "random_interval",
    ],
)
def test_exact_physical_budget_and_tracking_loss(agent):
    config = SimulationConfig(
        episodes=1,
        steps=2,
        time_budget=11,
        grid=GridConfig(size=5),
        agent=AgentConfig(eig_samples=16),
    )
    rows = run_episode(default_conditions()[0], agent, config, 0, 19)
    assert rows[-1]["time"] == 11
    assert sum(row["dt"] for row in rows) == 11
    assert rows[-1]["cumulative_tracking_squared_error"] >= 0
    assert rows[-1]["cumulative_sampling_cost"] == pytest.approx(
        11 * config.agent.wait_cost + len(rows) * config.agent.observation_rate_cost
    )


def test_posterior_example_is_the_prespecified_main_episode(tmp_path):
    config = SimulationConfig(
        episodes=1, steps=4, grid=GridConfig(size=5), agent=AgentConfig(eig_samples=16)
    )
    save_example_posterior(config, tmp_path)
    rows = run_episode(
        default_conditions()[0],
        "active_temporal",
        config,
        0,
        stable_seed(config.seed, default_conditions()[0].name, 0),
    )
    with np.load(tmp_path / "example_posterior.npz") as saved:
        assert int(saved["observation_count"]) == config.steps
        assert "selected_candidate" not in saved
        assert "prespecified" in str(saved["selection_rule"])
        np.testing.assert_allclose(
            saved["ratio_history"], [row["posterior_log_ratio"] for row in rows]
        )


def test_posterior_colorbar_uses_probability_heatmap(monkeypatch, tmp_path):
    import ats.figures as figures

    config = SimulationConfig(
        episodes=1, steps=3, grid=GridConfig(size=5), agent=AgentConfig(eig_samples=8)
    )
    save_example_posterior(config, tmp_path)
    target = tmp_path / "figures"
    target.mkdir()
    captured = []
    original = figures.plt.colorbar

    def record(mappable=None, **kwargs):
        captured.append(mappable)
        return original(mappable, **kwargs)

    monkeypatch.setattr(figures.plt, "colorbar", record)
    figures.plot_example_posterior(tmp_path, target)
    with np.load(tmp_path / "example_posterior.npz") as saved:
        np.testing.assert_array_equal(
            np.asarray(captured[0].get_array()), saved["posterior"]
        )
        assert captured[0].get_clim()[1] == pytest.approx(
            float(saved["posterior"].max())
        )
        left, right, bottom, top = captured[0].get_extent()
        s, v = np.log10(saved["stochasticity_grid"]), np.log10(saved["volatility_grid"])
        assert left + (right - left) / (2 * len(s)) == pytest.approx(s[0])
        assert bottom + (top - bottom) / (2 * len(v)) == pytest.approx(v[0])


def test_acquisition_common_random_numbers_are_invariant_to_action_order():
    v, s = GridConfig(size=5).make_grids()
    belief = GridBelief(v, s)
    belief.reset(0.0)
    args = (0.05, 0.08, 128, "entropy")
    a = active_temporal_sampling_scores(
        belief, (1, 2, 4, 8), *args, np.random.default_rng(7)
    )
    b = active_temporal_sampling_scores(
        belief, (8, 4, 2, 1), *args, np.random.default_rng(7)
    )
    assert a == b


def test_per_observation_cost_is_linear_in_count_and_elapsed_time():
    assert sum(sampling_cost(dt, 0.1, 0.2) for dt in (1, 2, 4)) == pytest.approx(
        0.1 * 7 + 0.2 * 3
    )
    assert sampling_cost(4, 0.1, 0.2, "legacy_rate") == pytest.approx(0.45)


def test_ppo_timeout_bootstraps_and_stops_trace_before_next_episode():
    rewards = torch.tensor([[1.0], [100.0]])
    values = torch.tensor([[2.0], [3.0]])
    next_values = torch.tensor([[10.0], [9.0]])
    terminated = torch.tensor([[0.0], [1.0]])
    ends = torch.ones_like(terminated)
    advantages, returns = generalized_advantage_estimate(
        rewards, values, next_values, terminated, ends, 0.9, 0.95
    )
    assert returns[0].item() == pytest.approx(10.0)
    assert returns[1].item() == pytest.approx(100.0)
    assert advantages[0].item() == pytest.approx(8.0)


@pytest.mark.parametrize("mode", ["ats", "none", "history", "ema"])
def test_ppo_feature_controls_have_identical_parameter_counts(mode):
    config = PPOConfig(feature_mode=mode, ats_projection_count=2)
    encoder = make_encoder(3, config, 0)
    reference = make_encoder(3, replace(config, feature_mode="ats"), 0)
    assert encoder.feature_dim == reference.feature_dim
    torch.manual_seed(12)
    a = ActorCritic(3 + encoder.feature_dim, 1, 16)
    torch.manual_seed(12)
    b = ActorCritic(3 + reference.feature_dim, 1, 16)
    assert sum(p.numel() for p in a.parameters()) == sum(
        p.numel() for p in b.parameters()
    )
    for left, right in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(left, right)


def test_projection_features_are_identical_across_vector_environment_indices():
    config = PPOConfig(seed=123, ats_projection_count=2)
    a, b = [make_encoder(3, config, index) for index in (0, 1)]
    a.reset(np.zeros(3))
    b.reset(np.zeros(3))
    np.testing.assert_array_equal(
        a.update(np.array([1.0, -1.0, 0.5])), b.update(np.array([1.0, -1.0, 0.5]))
    )


def test_history_control_retains_previous_observations():
    encoder = make_encoder(2, PPOConfig(feature_mode="history", history_length=3))
    encoder.reset(np.array([1.0, 2.0]))
    features = encoder.update(np.array([3.0, 4.0]))
    np.testing.assert_array_equal(features[:6], [3, 4, 1, 2, 1, 2])


class StubDMEnv:
    """Minimal simulator for checking dm_env boundary conversion."""

    def __init__(self, discount):
        self.discount = discount

    def action_spec(self):
        return SimpleNamespace(shape=(1,), minimum=-1.0, maximum=1.0)

    def reset(self):
        return SimpleNamespace(observation=np.zeros(1))

    def step(self, action):
        return SimpleNamespace(
            observation=np.ones(1),
            reward=0.0,
            discount=self.discount,
            last=lambda: True,
        )


@pytest.mark.parametrize(
    "discount,expected", [(0.0, (True, False)), (1.0, (False, True))]
)
def test_dm_env_boundary_respects_discount(discount, expected):
    result = RWRLGymWrapper(StubDMEnv(discount)).step(np.zeros(1))
    assert result[2:4] == expected


def test_semimarkov_backup_and_endpoint_reward_discount():
    rewards, values, dones, durations = (
        torch.tensor([1.0]),
        torch.tensor([2.0]),
        torch.tensor([0.0]),
        torch.tensor([4.0]),
    )
    assert bellman_target(
        rewards, values, dones, durations, 0.5, "physical_time"
    ).item() == pytest.approx(1.125)
    config = DeepPlannerConfig(
        gamma=0.5, agent=AgentConfig(wait_cost=0.1, observation_rate_cost=0.2)
    )
    assert transition_return(config, 1.0, 4) == pytest.approx(
        0.5**3 * 0.8 - 0.1 * (1 + 0.5 + 0.25 + 0.125)
    )


def test_existing_run_is_never_silently_overwritten(tmp_path):
    prepare_run(tmp_path, AgentConfig(), "test")
    before = (tmp_path / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        prepare_run(tmp_path, AgentConfig(), "test")
    assert (tmp_path / "manifest.json").read_bytes() == before


def test_artifact_verification_detects_tampering(tmp_path):
    prepare_run(tmp_path, AgentConfig(), "integrity")
    (tmp_path / "data.csv").write_text("value\n1\n")
    complete_run(tmp_path)
    assert verify_run(tmp_path, check_current_source=True) == []
    (tmp_path / "data.csv").write_text("value\n2\n")
    assert any("data.csv" in failure for failure in verify_run(tmp_path))


def test_registering_figures_preserves_data_identity_and_rejects_changed_data(tmp_path):
    prepare_run(tmp_path, AgentConfig(), "figures")
    (tmp_path / "data.csv").write_text("value\n1\n")
    complete_run(tmp_path)
    original = json.loads((tmp_path / "manifest.json").read_text())["source_sha256"]
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "example.txt").write_text("derived illustration")
    register_derived_figures(tmp_path)
    assert verify_run(tmp_path) == []
    assert (
        json.loads((tmp_path / "manifest.json").read_text())["source_sha256"]
        == original
    )
    (tmp_path / "data.csv").write_text("value\n2\n")
    with pytest.raises(RuntimeError):
        register_derived_figures(tmp_path)


def test_failed_run_has_an_explicit_failure_status(monkeypatch, tmp_path):
    import ats.experiments as experiments

    def fail(*args, **kwargs):
        raise FloatingPointError("intentional regression probe")

    monkeypatch.setattr(experiments, "run_episode", fail)
    config = SimulationConfig(
        output_root=tmp_path, run_name="failure", episodes=1, steps=1
    )
    with pytest.raises(FloatingPointError):
        experiments.run_experiment(config)
    manifest = json.loads((config.output_dir() / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error_type"] == "FloatingPointError"


def test_hgf_reported_gain_matches_its_mean_update():
    model = PyHGFContinuousModel()
    model.reset(0.3)
    prediction = model.predict(8)
    estimate = model.update(-0.7, 8)
    expected = prediction.mean + estimate.learning_rate * (-0.7 - prediction.mean)
    assert float(model._state()[1]["mean"]) == pytest.approx(expected, abs=1e-5)


def test_cli_preserves_smoke_grid_and_exposes_seeds_and_feature_controls():
    from ats.cli import build_parser as ats_parser
    from deep_ats_planner.cli import (
        build_parser,
        config_from_args,
        ppo_config_from_args,
    )

    args = build_parser().parse_args(["smoke", "--seed", "91"])
    config = config_from_args(args, smoke=True)
    assert config.grid.size == 9
    assert config.seed == 91
    ppo = ppo_config_from_args(
        build_parser().parse_args(
            ["ppo-train", "--seed", "92", "--feature-mode", "history"]
        )
    )
    assert ppo.seed == 92 and ppo.resolved_feature_mode == "history"
    scalar = ats_parser().parse_args(
        ["run", "--time-budget", "37", "--observation-cost", "0.2"]
    )
    assert scalar.time_budget == 37 and scalar.observation_rate_cost == 0.2


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 0.0])
def test_invalid_gaussian_variances_raise(bad):
    with pytest.raises(FloatingPointError):
        normal_nll(0.0, 0.0, bad)


def test_new_run_manifest_and_matched_controls(tmp_path):
    config = SimulationConfig(
        output_root=tmp_path,
        run_name="provenance",
        episodes=1,
        steps=3,
        grid=GridConfig(size=5),
        agent=AgentConfig(eig_samples=8),
        include_ablation=False,
        include_model_comparison=False,
        include_competitive_comparison=False,
    )
    directory = run_experiment(config)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["source_sha256"]
    assert manifest["config"]["seed"] == config.seed
    with (directory / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    agents = {row["agent"] for row in rows}
    assert {
        "fixed_dt_2",
        "fixed_dt_4",
        "matched_histogram",
        "yoked_independent",
    } <= agents
    assert (directory / "paired_comparisons.csv").exists()


class TimeoutEnv(gym.Env):
    """Return an identifiable final observation before a different reset observation."""

    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-100.0, 100.0, (1,), dtype=np.float32)
        self.closed = False

    def reset(self, *, seed=None, options=None):
        return np.array([-7.0], dtype=np.float32), {}

    def step(self, action):
        return np.array([3.0], dtype=np.float32), 0.0, False, True, {}

    def close(self):
        self.closed = True


def test_ppo_rollout_uses_final_timeout_observation_instead_of_reset(
    monkeypatch, tmp_path
):
    import deep_ats_planner.deep_ppo as ppo

    captured = []
    environments = []
    original = ppo.generalized_advantage_estimate

    def capture(rewards, values, next_values, terminated, ends, gamma, gae_lambda):
        captured.append(
            (next_values.cpu().clone(), terminated.cpu().clone(), ends.cpu().clone())
        )
        return original(
            rewards, values, next_values, terminated, ends, gamma, gae_lambda
        )

    def make_environment(*args):
        env = TimeoutEnv()
        environments.append(env)
        return env

    monkeypatch.setattr(ppo, "make_continuous_env", make_environment)
    monkeypatch.setattr(
        ppo.RunningMeanStd,
        "normalize",
        lambda self, values: np.asarray(values, dtype=np.float32),
    )
    monkeypatch.setattr(
        ppo.ActorCritic, "get_value", lambda self, observation: observation[:, :1]
    )
    monkeypatch.setattr(ppo, "generalized_advantage_estimate", capture)
    config = PPOConfig(
        output_root=tmp_path,
        run_name="timeout",
        feature_mode="none",
        total_timesteps=2,
        num_envs=1,
        num_steps=2,
        num_minibatches=1,
        update_epochs=1,
        hidden_size=8,
        evaluation_episodes=1,
        device="cpu",
    )
    directory = ppo.train_ppo(config)
    torch.testing.assert_close(captured[0][0], torch.full((2, 1), 3.0))
    assert torch.count_nonzero(captured[0][1]).item() == 0
    assert torch.all(captured[0][2] == 1)
    assert all(env.closed for env in environments)
    checkpoint = torch.load(directory / "deep_ats_ppo.pt", weights_only=True)
    assert checkpoint["config"]["feature_mode"] == "none"
