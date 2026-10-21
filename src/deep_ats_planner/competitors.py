"""Compatibility exports from the canonical ATS implementation."""

from ats.competitors import (
    stable_seed as stable_seed,
    finite_mean as finite_mean,
    uncertainty_sampling_score as uncertainty_sampling_score,
    BaseCompetitiveModel as BaseCompetitiveModel,
    ATSJointActiveModel as ATSJointActiveModel,
    FixedScheduleJointModel as FixedScheduleJointModel,
    UncertaintyActiveModel as UncertaintyActiveModel,
    make_competitive_models as make_competitive_models,
    make_competitive_model as make_competitive_model,
    run_competitive_model_comparison as run_competitive_model_comparison,
    summarize_competitive_model_comparison as summarize_competitive_model_comparison,
    competitive_model_names as competitive_model_names,
)
