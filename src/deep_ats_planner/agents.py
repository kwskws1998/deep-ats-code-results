"""Compatibility exports from the canonical ATS implementation."""

from ats.agents import (
    BaseAgent as BaseAgent,
    FixedShortAgent as FixedShortAgent,
    FixedLongAgent as FixedLongAgent,
    RandomIntervalAgent as RandomIntervalAgent,
    PassiveGridAgent as PassiveGridAgent,
    ActiveTemporalSampler as ActiveTemporalSampler,
    make_agent as make_agent,
    default_agent_names as default_agent_names,
)
