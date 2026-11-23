"""Ramen Foundry: LangGraph governance nodes and workflow templates.

Public names load on first access (PEP 562), so ``import ramen_foundry`` and
submodules that do not need LangGraph, such as ``ramen_foundry.data``, stay
light. ``from ramen_foundry import DevboxShieldAgent`` works as before.
"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs only
    from .core import (
        BaseEpisodicMemoryStore,
        CorrectionExemplar,
        Ed25519TokenVerifier,
        EvidenceResult,
        EvidenceVerifier,
        JSONFileMemoryStore,
        RamenGovernedNode,
        RamenSteerNode,
        RamenToolNode,
        RemoteForgeMemoryStore,
        SQLiteMemoryStore,
        ToolInvocation,
    )
    from .templates import (
        AUTHORIZATION_GATE_BUNDLE_ID,
        ApprovalBinding,
        ApprovalEvidence,
        AuthorizationEvidence,
        AuthorizationGateAgent,
        ClaimEvidence,
        ClaimRequest,
        CommercialLendingAgent,
        DbShieldAgent,
        DevboxShieldAgent,
        ELIGIBILITY_GATE_BUNDLE_ID,
        EU_AI_ACT_PROXY_BIAS_POLICY_ID,
        EligibilityCondition,
        EligibilityDecision,
        EligibilityGateAgent,
        ErpClearance,
        ExclusionHit,
        ExclusionRule,
        ExclusionRuleSet,
        GovernedToolAgent,
        INDUSTRIAL_IOT_ACTUATION_INVARIANCE_BUNDLE_ID,
        IndustrialAutomationAgent,
        InspectionEvidence,
        ManipulationActionType,
        ManipulationDispatchPayload,
        PROVENANCE_GATE_BUNDLE_ID,
        PlatformLimits,
        PlatformSchedule,
        PolicySystemRecord,
        ProvenanceGateAgent,
        ReceiptEvidenceVerifier,
        ResumeScreeningAgent,
        ResumeScreeningRequest,
        ResumeScreeningResult,
        SHIELD_CORE_IT_BUNDLE_ID,
        ScoutShieldAgent,
        VendorExposure,
        VerifiedReceipt,
        manipulation_payload_problem,
    )
    from .trial import TrialComplianceClient, describe_trial_limit

_CORE_NAMES = frozenset(
    (
        "BaseEpisodicMemoryStore",
        "CorrectionExemplar",
        "Ed25519TokenVerifier",
        "EvidenceResult",
        "EvidenceVerifier",
        "JSONFileMemoryStore",
        "RamenGovernedNode",
        "RamenSteerNode",
        "RamenToolNode",
        "RemoteForgeMemoryStore",
        "SQLiteMemoryStore",
        "ToolInvocation",
    )
)
_TRIAL_NAMES = frozenset(("TrialComplianceClient", "describe_trial_limit"))
_TEMPLATE_NAMES = frozenset(
    (
        "AUTHORIZATION_GATE_BUNDLE_ID",
        "ApprovalBinding",
        "ApprovalEvidence",
        "AuthorizationEvidence",
        "AuthorizationGateAgent",
        "ClaimEvidence",
        "ClaimRequest",
        "CommercialLendingAgent",
        "DbShieldAgent",
        "DevboxShieldAgent",
        "ELIGIBILITY_GATE_BUNDLE_ID",
        "EU_AI_ACT_PROXY_BIAS_POLICY_ID",
        "EligibilityCondition",
        "EligibilityDecision",
        "EligibilityGateAgent",
        "ErpClearance",
        "ExclusionHit",
        "ExclusionRule",
        "ExclusionRuleSet",
        "GovernedToolAgent",
        "INDUSTRIAL_IOT_ACTUATION_INVARIANCE_BUNDLE_ID",
        "IndustrialAutomationAgent",
        "InspectionEvidence",
        "ManipulationActionType",
        "ManipulationDispatchPayload",
        "PROVENANCE_GATE_BUNDLE_ID",
        "PlatformLimits",
        "PlatformSchedule",
        "PolicySystemRecord",
        "ProvenanceGateAgent",
        "ReceiptEvidenceVerifier",
        "ResumeScreeningAgent",
        "ResumeScreeningRequest",
        "ResumeScreeningResult",
        "SHIELD_CORE_IT_BUNDLE_ID",
        "ScoutShieldAgent",
        "VendorExposure",
        "VerifiedReceipt",
        "manipulation_payload_problem",
    )
)

__all__ = [
    "AUTHORIZATION_GATE_BUNDLE_ID",
    "ApprovalBinding",
    "ApprovalEvidence",
    "AuthorizationEvidence",
    "AuthorizationGateAgent",
    "BaseEpisodicMemoryStore",
    "ClaimEvidence",
    "ClaimRequest",
    "CommercialLendingAgent",
    "CorrectionExemplar",
    "DbShieldAgent",
    "DevboxShieldAgent",
    "ELIGIBILITY_GATE_BUNDLE_ID",
    "EU_AI_ACT_PROXY_BIAS_POLICY_ID",
    "Ed25519TokenVerifier",
    "EligibilityCondition",
    "EligibilityDecision",
    "EligibilityGateAgent",
    "ErpClearance",
    "EvidenceResult",
    "EvidenceVerifier",
    "ExclusionHit",
    "ExclusionRule",
    "ExclusionRuleSet",
    "GovernedToolAgent",
    "INDUSTRIAL_IOT_ACTUATION_INVARIANCE_BUNDLE_ID",
    "IndustrialAutomationAgent",
    "InspectionEvidence",
    "JSONFileMemoryStore",
    "ManipulationActionType",
    "ManipulationDispatchPayload",
    "PROVENANCE_GATE_BUNDLE_ID",
    "PlatformLimits",
    "PlatformSchedule",
    "PolicySystemRecord",
    "ProvenanceGateAgent",
    "RamenGovernedNode",
    "RamenSteerNode",
    "RamenToolNode",
    "ReceiptEvidenceVerifier",
    "RemoteForgeMemoryStore",
    "ResumeScreeningAgent",
    "ResumeScreeningRequest",
    "ResumeScreeningResult",
    "SHIELD_CORE_IT_BUNDLE_ID",
    "SQLiteMemoryStore",
    "ScoutShieldAgent",
    "ToolInvocation",
    "TrialComplianceClient",
    "VendorExposure",
    "VerifiedReceipt",
    "describe_trial_limit",
    "manipulation_payload_problem",
]

__version__ = "0.2.8"


def __getattr__(name: str) -> Any:
    if name in _CORE_NAMES:
        module = import_module(".core", __name__)
    elif name in _TEMPLATE_NAMES:
        module = import_module(".templates", __name__)
    elif name in _TRIAL_NAMES:
        module = import_module(".trial", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(module, name)
    globals()[name] = value  # cache: later lookups bypass __getattr__
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
