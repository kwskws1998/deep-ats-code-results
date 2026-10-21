from dataclasses import replace

from ats.config import GridConfig, smoke_config
from ats.artifacts import verify_run
from ats.experiments import run_experiment
from ats.figures import generate_all_figures


def test_smoke_pipeline_writes_csvs_and_figures(tmp_path):
    config = smoke_config("pytest_smoke")
    config = replace(
        config,
        output_root=tmp_path,
        episodes=1,
        steps=5,
        grid=GridConfig(size=7),
        ablation_episodes=1,
    )
    output_dir = run_experiment(config)
    fig_dir = generate_all_figures(output_dir)

    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "trajectories.csv").exists()
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "wait_cost_ablation_summary.csv").exists()
    assert (output_dir / "model_comparison_metrics.csv").exists()
    assert (output_dir / "model_comparison_summary.csv").exists()
    assert (output_dir / "competitive_model_metrics.csv").exists()
    assert (output_dir / "competitive_model_summary.csv").exists()
    for index in range(1, 13):
        assert (fig_dir / f"fig{index:02d}_{_figure_name(index)}.png").exists()
    assert verify_run(output_dir, check_current_source=True) == []


def _figure_name(index: int) -> str:
    names = {
        1: "task_schematic",
        2: "example_posterior",
        3: "ratio_error",
        4: "prediction_mse",
        5: "dt_choices",
        6: "learning_rate",
        7: "wait_cost_ablation",
        8: "pareto_tradeoff",
        9: "model_comparison_nll",
        10: "model_comparison_ratio",
        11: "competitive_model_loss",
        12: "competitive_model_dt",
    }
    return names[index]
