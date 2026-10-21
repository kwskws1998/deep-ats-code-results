import numpy as np

from ats.config import ExperimentCondition, GridConfig, SimulationConfig
from ats.models import make_model_suite, normal_nll


def test_perceptual_models_predict_and_update_finite():
    condition = ExperimentCondition("test", volatility=0.25, stochasticity=1.0)
    config = SimulationConfig(grid=GridConfig(size=7), episodes=1, steps=4)
    models = make_model_suite(condition, config)

    for model in models:
        model.reset(initial_observation=0.0)
        prediction = model.predict(dt=2)
        estimate = model.update(observation=1.0, dt=2)

        assert np.isfinite(prediction.mean)
        assert np.isfinite(prediction.variance)
        assert prediction.variance > 0.0
        assert np.isfinite(normal_nll(1.0, prediction.mean, prediction.variance))
        assert np.isfinite(estimate.learning_rate)
