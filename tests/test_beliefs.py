import numpy as np

from ats.beliefs import GridBelief
from ats.config import ExperimentCondition, GridConfig
from ats.envs import TemporalSamplingEnv


def test_belief_update_normalizes_and_stays_finite():
    volatility_grid, stochasticity_grid = GridConfig(size=7).make_grids()
    belief = GridBelief(volatility_grid, stochasticity_grid)
    belief.reset(initial_observation=0.0)
    summary = belief.update(observation=1.0, dt=2)

    assert np.isclose(np.sum(belief.posterior), 1.0)
    assert np.all(np.isfinite(belief.posterior))
    assert np.isfinite(summary.expected_log_ratio)
    assert np.isfinite(summary.learning_rate)


def test_belief_recovers_noise_ratio_sanity():
    condition = ExperimentCondition("high_v_low_s", volatility=4.0, stochasticity=1.0)
    env = TemporalSamplingEnv(condition, np.random.default_rng(123))
    volatility_grid, stochasticity_grid = GridConfig(size=17).make_grids()
    belief = GridBelief(volatility_grid, stochasticity_grid)
    initial = env.reset()
    belief.reset(initial.observation)
    true_log_ratio = np.log(condition.volatility / condition.stochasticity)

    for step in range(120):
        dt = 8 if step % 2 == 0 else 1
        result = env.step(dt)
        summary = belief.update(result.observation, dt)

    assert abs(summary.expected_log_ratio - true_log_ratio) < 1.5
