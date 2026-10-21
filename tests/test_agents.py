import numpy as np

from ats.agents import default_agent_names, make_agent
from ats.beliefs import GridBelief
from ats.config import AgentConfig, GridConfig
from ats.policies import active_temporal_sampling_scores


def test_agents_choose_allowed_intervals():
    config = AgentConfig()
    volatility_grid, stochasticity_grid = GridConfig(size=9).make_grids()
    belief = GridBelief(volatility_grid, stochasticity_grid)
    belief.reset(0.0)

    for index, name in enumerate(default_agent_names()):
        agent = make_agent(name, config, seed=index)
        for step in range(8):
            assert agent.choose_dt(belief, step) in config.action_space


def test_active_scores_are_finite():
    config = AgentConfig()
    volatility_grid, stochasticity_grid = GridConfig(size=9).make_grids()
    belief = GridBelief(volatility_grid, stochasticity_grid)
    belief.reset(0.0)
    scores = active_temporal_sampling_scores(
        belief,
        config.action_space,
        config.wait_cost,
        config.observation_rate_cost,
        config.eig_samples,
        config.information_objective,
        np.random.default_rng(3),
    )

    assert set(scores) == set(config.action_space)
    assert all(np.isfinite(value) for value in scores.values())
