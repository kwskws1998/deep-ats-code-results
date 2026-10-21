"""Compatibility exports from the canonical ATS implementation."""

from ats.figures import (
    read_csv as read_csv,
    ensure_fig_dir as ensure_fig_dir,
    save_current as save_current,
    plot_task_schematic as plot_task_schematic,
    plot_example_posterior as plot_example_posterior,
    plot_ratio_error as plot_ratio_error,
    plot_prediction_mse as plot_prediction_mse,
    plot_dt_choices as plot_dt_choices,
    plot_learning_rate as plot_learning_rate,
    plot_wait_cost_ablation as plot_wait_cost_ablation,
    plot_pareto_tradeoff as plot_pareto_tradeoff,
    finite_metric as finite_metric,
    plot_model_comparison_nll as plot_model_comparison_nll,
    plot_model_comparison_ratio as plot_model_comparison_ratio,
    plot_competitive_model_loss as plot_competitive_model_loss,
    plot_competitive_model_dt as plot_competitive_model_dt,
    generate_all_figures as generate_all_figures,
    AGENT_LABELS as AGENT_LABELS,
    MODEL_LABELS as MODEL_LABELS,
    COMPETITIVE_MODEL_LABELS as COMPETITIVE_MODEL_LABELS,
)
