import numpy as np

from deep_ats_planner.beliefs import GridBelief
from deep_ats_planner.config import ExperimentCondition, GridConfig, deep_smoke_config
from deep_ats_planner.planner import DeepATSPlanner
from deep_ats_planner.rl_envs import TemporalBanditEnv
from deep_ats_planner.state import belief_features, feature_names
from deep_ats_planner.training import run_deep_planner


def test_temporal_bandit_env_returns_valid_reward_and_observation():
    env = TemporalBanditEnv(
        ExperimentCondition("test", volatility=0.25, stochasticity=1.0),
        np.random.default_rng(7),
    )
    initial = env.reset()
    result = env.step(action=1, dt=2)

    assert initial.dt == 0
    assert result.dt == 2
    assert result.action == 1
    assert result.reward in (0.0, 1.0)
    assert 0.0 <= result.reward_probability <= 1.0
    assert np.isfinite(result.observation)


def test_belief_features_are_finite_and_named():
    grid = GridConfig(size=5)
    volatility, stochasticity = grid.make_grids()
    belief = GridBelief(volatility, stochasticity, initial_variance=12.0)
    belief.reset(initial_observation=0.0)
    features = belief_features(
        belief=belief,
        action_space=(1, 2, 4, 8),
        task_action_count=2,
        last_observation=0.0,
        last_reward=0.0,
        last_action=-1,
        last_dt=1,
    )

    assert features.shape == (len(feature_names(2, (1, 2, 4, 8))),)
    assert np.all(np.isfinite(features))


def test_deep_planner_selects_valid_joint_action():
    rng = np.random.default_rng(11)
    planner = DeepATSPlanner(
        input_dim=len(feature_names(2)),
        task_action_count=2,
        action_space=(1, 2, 4, 8),
        hidden_size=16,
        learning_rate=1e-3,
        gamma=0.9,
        replay_capacity=100,
    )
    state = np.zeros(len(feature_names(2)), dtype=np.float32)
    action, dt, action_index, q_values = planner.select(state, epsilon=0.0, rng=rng)

    assert action in (0, 1)
    assert dt in (1, 2, 4, 8)
    assert 0 <= action_index < 8
    assert q_values.shape == (8,)


def test_deep_planner_smoke_writes_outputs(tmp_path):
    config = deep_smoke_config("pytest_deep_smoke")
    config = config.__class__(
        **{
            **config.__dict__,
            "output_root": tmp_path,
            "episodes": 1,
            "steps": 6,
            "warmup_steps": 2,
            "batch_size": 2,
        }
    )
    output_dir = run_deep_planner(config)

    assert (output_dir / "deep_steps.csv").exists()
    assert (output_dir / "deep_episodes.csv").exists()
    assert (output_dir / "deep_summary.csv").exists()
    assert (output_dir / "deep_ats_planner.pt").exists()
