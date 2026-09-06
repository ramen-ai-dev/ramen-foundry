"""Ramen Foundry: LangGraph governance nodes and workflow templates."""

from .core import RamenGovernedNode, RamenToolNode, ToolInvocation
from .templates import (
    DbShieldAgent,
    DevboxShieldAgent,
    EU_AI_ACT_PROXY_BIAS_POLICY_ID,
    ResumeScreeningAgent,
    ResumeScreeningRequest,
    ResumeScreeningResult,
    SHIELD_CORE_IT_BUNDLE_ID,
    ScoutShieldAgent,
)

__all__ = [
    "DbShieldAgent",
    "DevboxShieldAgent",
    "EU_AI_ACT_PROXY_BIAS_POLICY_ID",
    "RamenGovernedNode",
    "RamenToolNode",
    "ResumeScreeningAgent",
    "ResumeScreeningRequest",
    "ResumeScreeningResult",
    "SHIELD_CORE_IT_BUNDLE_ID",
    "ScoutShieldAgent",
    "ToolInvocation",
]

__version__ = "0.1.1"
