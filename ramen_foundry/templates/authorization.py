"""Governed financial authorization gates: budget clearance and dual control."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Annotated, Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from ramen_ai import RamenClient

from ramen_foundry.core import ToolInvocation
from ramen_foundry.core.evidence import EvidenceVerifier

from ._base import GovernedToolAgent

AUTHORIZATION_GATE_BUNDLE_ID = "ramen__financial_authorization_invariance"
AUTHORIZATION_GATE_TOOL_NAMES = frozenset({"approve_expenditure", "release_payment"})
DEFAULT_DUAL_CONTROL_THRESHOLD_USD = 10_000.0
DEFAULT_REQUIRED_APPROVERS = 2

_NonBlank = Annotated[str, Field(pattern=r"\S")]
_Money = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class ApprovalEvidence(BaseModel):
    """One approver's evidence that they authorize a specific request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    approver_id: _NonBlank
    approval_token: _NonBlank


class AuthorizationEvidence(BaseModel):
    """Evidence a host must resolve into the arguments of a gated tool call.

    Extra keys (beneficiary details, memo fields, and so on) are allowed and are
    passed to the policy evaluation unchanged.
    """

    model_config = ConfigDict(extra="allow", strict=True)

    request_id: _NonBlank
    requested_by: _NonBlank
    amount_usd: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    budget_id: _NonBlank
    budget_remaining_usd: _Money
    budget_clearance_token: _NonBlank
    approvals: list[ApprovalEvidence] = Field(default_factory=list)


class AuthorizationGateAgent(GovernedToolAgent):
    """Gate budget-clearance and dual-control decisions before a spend is committed.

    ``approve_expenditure`` and ``release_payment`` both carry the evidence
    contract in :class:`AuthorizationEvidence`. Before the ramen-ai backend is
    contacted, the template blocks locally (fail closed) when:

    * the evidence is missing, mistyped, or malformed;
    * ``amount_usd`` exceeds ``budget_remaining_usd``;
    * ``amount_usd`` is at or above ``dual_control_threshold_usd`` and fewer than
      ``required_approvers`` distinct approvers are present, an approver is listed
      twice, or the requester appears among the approvers.

    When a ``budget_verifier`` is supplied, ``budget_clearance_token`` must verify
    and bind ``request_id``, ``budget_id`` and ``amount_usd``. When an
    ``approval_verifier`` is supplied, every approval token must verify and bind
    ``request_id``, ``approver_id`` and ``amount_usd``. Without a verifier the
    token is only required to be present, and the bundle evaluates it. Use
    :class:`~ramen_foundry.core.evidence.Ed25519TokenVerifier` or any other
    :class:`~ramen_foundry.core.evidence.EvidenceVerifier`.

    Passing local checks never releases a tool by itself. The bundle verdict and
    its Ed25519 receipt are still required. Hosts keep token issuance, key
    custody, ledger authorization and the tool implementations.
    """

    bundle_id = AUTHORIZATION_GATE_BUNDLE_ID
    allowed_tool_names = AUTHORIZATION_GATE_TOOL_NAMES
    node_name = "governed_authorization_action"

    def __init__(
        self,
        *,
        client: RamenClient,
        tools: Mapping[str, BaseTool],
        budget_verifier: EvidenceVerifier | None = None,
        approval_verifier: EvidenceVerifier | None = None,
        dual_control_threshold_usd: float = DEFAULT_DUAL_CONTROL_THRESHOLD_USD,
        required_approvers: int = DEFAULT_REQUIRED_APPROVERS,
        provider_key: str | None = None,
        provider_name: str | None = None,
    ) -> None:
        if (
            isinstance(dual_control_threshold_usd, bool)
            or not isinstance(dual_control_threshold_usd, (int, float))
            or not math.isfinite(dual_control_threshold_usd)
            or dual_control_threshold_usd < 0
        ):
            raise ValueError("dual_control_threshold_usd must be a finite, non-negative number")
        if (
            isinstance(required_approvers, bool)
            or not isinstance(required_approvers, int)
            or required_approvers < 1
        ):
            raise ValueError("required_approvers must be a positive integer")
        for name, verifier in (
            ("budget_verifier", budget_verifier),
            ("approval_verifier", approval_verifier),
        ):
            if verifier is not None and not isinstance(verifier, EvidenceVerifier):
                raise TypeError(f"{name} must implement EvidenceVerifier")
        self._budget_verifier = budget_verifier
        self._approval_verifier = approval_verifier
        self._threshold = float(dual_control_threshold_usd)
        self._required_approvers = required_approvers
        super().__init__(
            client=client,
            tools=tools,
            provider_key=provider_key,
            provider_name=provider_name,
        )

    def _preflight(self, invocation: ToolInvocation) -> str | None:
        try:
            evidence = AuthorizationEvidence.model_validate(invocation.arguments)
        except ValidationError as error:
            # Report field paths only; never echo submitted values such as tokens.
            fields = sorted({".".join(str(part) for part in item["loc"]) for item in error.errors()})
            return (
                "Authorization evidence is incomplete or malformed "
                f"(fields: {', '.join(fields)}); supply every required field."
            )

        if evidence.amount_usd > evidence.budget_remaining_usd:
            return (
                "Requested amount exceeds the remaining cleared budget; "
                "request a smaller amount or obtain additional budget clearance."
            )

        if self._budget_verifier is not None:
            result = self._budget_verifier.verify(
                evidence.budget_clearance_token,
                expected={
                    "request_id": evidence.request_id,
                    "budget_id": evidence.budget_id,
                    "amount_usd": evidence.amount_usd,
                },
            )
            if not result.valid:
                return f"Budget clearance token rejected: {result.reason}"

        if evidence.amount_usd >= self._threshold:
            approver_ids = [approval.approver_id for approval in evidence.approvals]
            if len(set(approver_ids)) != len(approver_ids):
                return "Dual control requires distinct approvers; an approver is listed more than once."
            if evidence.requested_by in approver_ids:
                return "Dual control requires approvers independent of the requester."
            if len(approver_ids) < self._required_approvers:
                return (
                    f"Dual control requires {self._required_approvers} distinct approvers at or above "
                    f"{self._threshold:,.2f} USD; {len(approver_ids)} supplied."
                )
            if self._approval_verifier is not None:
                for approval in evidence.approvals:
                    result = self._approval_verifier.verify(
                        approval.approval_token,
                        expected={
                            "request_id": evidence.request_id,
                            "approver_id": approval.approver_id,
                            "amount_usd": evidence.amount_usd,
                        },
                    )
                    if not result.valid:
                        return f"Approval token for approver {approval.approver_id!r} rejected: {result.reason}"
        return None
