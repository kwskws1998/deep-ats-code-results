import numpy as np
import torch

from deep_ats_planner.ats_encoder import ATSFeatureSpec, RandomProjectionATSEncoder
from deep_ats_planner.config import AgentConfig, GridConfig
from deep_ats_planner.deep_ppo import ActorCritic, PPOConfig, train_ppo
from deep_ats_planner.device import select_device
from deep_ats_planner.rwrl_reference import (
    best_reference_rows,
    write_rwrl_reference_csv,
)


def test_select_device_accepts_cpu_override():
    assert select_device("cpu").type == "cpu"


def test_random_projection_ats_encoder_returns_finite_features():
    encoder = RandomProjectionATSEncoder(
        observation_dim=3,
        grid=GridConfig(size=5),
        agent=AgentConfig(action_space=(1, 2, 4), initial_variance=8.0),
        spec=ATSFeatureSpec(projection_count=2, projection_seed=17),
    )
    initial = encoder.reset(np.array([0.1, -0.2, 0.3], dtype=np.float32))
    updated = encoder.update(np.array([0.2, -0.1, 0.4], dtype=np.float32), dt=2)

    assert initial.shape == (encoder.feature_dim,)
    assert updated.shape == (encoder.feature_dim,)
    assert np.all(np.isfinite(updated))


def test_actor_critic_shapes_for_continuous_action():
    model = ActorCritic(observation_dim=7, action_dim=2, hidden_size=16)
    observation = torch.zeros((4, 7), dtype=torch.float32)
    action, logprob, entropy, value = model.get_action_and_value(observation)

    assert action.shape == (4, 2)
    assert logprob.shape == (4,)
    assert entropy.shape == (4,)
    assert value.shape == (4, 1)


def test_deep_ats_ppo_smoke_writes_outputs(tmp_path):
    config = PPOConfig(
        run_name="pytest_deep_ats_ppo_smoke",
        output_root=tmp_path,
        env_id="Pendulum-v1",
        seed=123,
        total_timesteps=32,
        num_envs=1,
        num_steps=16,
        num_minibatches=1,
        update_epochs=1,
        hidden_size=16,
        ats_projection_count=2,
        device="cpu",
    )
    output_dir = train_ppo(config)

    assert (output_dir / "ppo_metrics.csv").exists()
    assert (output_dir / "deep_ats_ppo.pt").exists()


def test_rwrl_reference_baselines_write_csv(tmp_path):
    output = tmp_path / "rwrl_reference.csv"
    best_rows = best_reference_rows()
    write_rwrl_reference_csv(output, best_rows)

    text = output.read_text()
    assert "algorithm,difficulty,task,mean_return,std_return" in text
    assert "D4PG,easy,quadruped:walk,787.73,86.95" in text
    assert len(best_rows) == 16
