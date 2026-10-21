import numpy as np

from ats.config import ExperimentCondition
from ats.envs import TemporalSamplingEnv


def test_environment_seed_reproducibility():
    condition = ExperimentCondition("test", volatility=2.0, stochasticity=1.5)
    env_a = TemporalSamplingEnv(condition, np.random.default_rng(7))
    env_b = TemporalSamplingEnv(condition, np.random.default_rng(7))

    assert env_a.reset() == env_b.reset()
    assert env_a.step(4) == env_b.step(4)


def test_process_variance_increases_with_dt():
    condition = ExperimentCondition("test", volatility=2.0, stochasticity=1.0)
    env = TemporalSamplingEnv(condition, np.random.default_rng(9))
    small = env.sample_process_increments(dt=1, size=20_000)
    large = env.sample_process_increments(dt=8, size=20_000)

    assert np.var(large) > np.var(small) * 6.0
