from dataclasses import replace

import numpy as np

from ats.competitors import make_competitive_models, uncertainty_sampling_score
from ats.config import ExperimentCondition, GridConfig, SimulationConfig
from ats.models import ModelPrediction


def test_competitive_models_choose_allowed_intervals_and_update():
    condition = ExperimentCondition("test", volatility=0.25, stochasticity=1.0)
    config = SimulationConfig(episodes=1, steps=3, grid=GridConfig(size=7))
    config = replace(config, include_ablation=False)
    models = make_competitive_models(condition, config, seed=11)

    for model in models:
        model.reset(initial_observation=0.0)
        dt = model.choose_dt(step=0)
        prediction = model.predict(dt)
        estimate = model.update(observation=1.0, dt=dt)

        assert dt in config.agent.action_space
        assert np.isfinite(prediction.mean)
        assert np.isfinite(prediction.variance)
        assert prediction.variance > 0.0
        assert np.isfinite(estimate.learning_rate)


def test_uncertainty_sampling_score_is_finite():
    score = uncertainty_sampling_score(
        ModelPrediction(mean=0.0, variance=2.0),
        dt=2,
        wait_cost=0.05,
        observation_rate_cost=0.08,
    )
    assert np.isfinite(score)
