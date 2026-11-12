"""Governed financial authorization gates: budget clearance and dual control.

The ``ramen__financial_authorization_invariance`` bundle judges platform-delivered
facts (limits, ERP clearance state, vendor exposure, approvals bound to the
payment). A call that omits them is blocked by the bundle, so this template
requires them up front and checks the arithmetic locally.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Annotated, Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from ramen_ai import RamenClient

from ramen_foundry.core import ToolInvocation
from ramen_foundry.core.evidence import EvidenceVerifier, claims_equal

from ._base import GovernedToolAgent

AUTHORIZATION_GATE_BUNDLE_ID = "ramen__financial_authorization_invariance"
AUTHORIZATION_GATE_TOOL_NAMES = frozenset({"approve_expenditure", "release_payment"})
DEFAULT_REQUIRED_APPROVERS = 2

_NonBlank = Annotated[str, Field(pattern=r"\S")]
_Money = Annotated[float, Field(ge=0, allow_inf_nan=False)]
_PositiveMoney = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class _Context(BaseModel):
    """Nested evidence object. Unknown keys are kept and sent to the bundle unchanged."""

    model_config = ConfigDict(extra="allow", strict=True)


class PlatformLimits(_Context):
    """Limits delivered by the finance platform. Never limits asserted by the agent."""

    source: _NonBlank
    per_order_usd: _PositiveMoney
    per_vendor_cumulative_usd: _PositiveMoney
    dual_approval_threshold_usd: _Money
    required_approvers: Annotated[int, Field(ge=1)] | None = None


class ErpClearance(_Context):
    """Budget clearance as confirmed by the finance system of record."""

    id: _NonBlank
    status: _NonBlank
    cost_center: _NonBlank
    fiscal_period: _NonBlank
    spend_category: _NonBlank
    balance_usd: _Money
    confirmed_by: _NonBlank
    confirmation_token: _NonBlank | None = None


class VendorExposure(_Context):
    """Open, pending and paid exposure for the vendor group, from the finance platform."""

    source: _NonBlank
    vendor_group: _NonBlank
    open_pending_paid_usd: _Money


class ApprovalBinding(_Context):
    """What an approver's approval is bound to. Must equal the payment."""

    amount_usd: _PositiveMoney
    payee: _NonBlank
    payee_account: _NonBlank


class ApprovalEvidence(_Context):
    """One human approval obtained out of band and bound to this exact payment."""

    approver_id: _NonBlank
    role: _NonBlank
    approval_token: _NonBlank
    approved_out_of_band: bool
    bound_to: ApprovalBinding


class AuthorizationEvidence(_Context):
    """The complete evidence contract of ``approve_expenditure`` and ``release_payment``.

    Extra keys are allowed and are evaluated unchanged.
    """

    request_id: _NonBlank
    requested_by: _NonBlank
    amount_usd: _PositiveMoney
    payee: _NonBlank
    payee_account: _NonBlank
    cost_center: _NonBlank
    fiscal_period: _NonBlank
    spend_category: _NonBlank
    platform_limits: PlatformLimits
    erp_clearance: ErpClearance
    vendor_exposure: VendorExposure
    approvals: list[ApprovalEvidence] = Field(default_factory=list)


def _same_dimension(requested: str, cleared: str) -> bool:
    return requested.strip().casefold() == cleared.strip().casefold()


class AuthorizationGateAgent(GovernedToolAgent):
    """Gate budget-clearance and dual-control decisions before a spend is committed.

    Both tools carry :class:`AuthorizationEvidence`. Before the ramen-ai backend
    is contacted, the call is blocked locally (fail closed) when:

    * any required evidence is missing or malformed, including platform limits,
      ERP clearance state, vendor exposure, payee and payee account;
    * the ERP clearance is not ``OPEN``, or its cost center, fiscal period or spend
      category differs from the request's own ``cost_center``, ``fiscal_period`` and
      ``spend_category`` (compared ignoring case and surrounding space);
    * ``amount_usd`` exceeds the platform per-order limit or the ERP clearance
      balance, or ``amount_usd`` plus the vendor exposure exceeds the platform
      cumulative cap;
    * any presented approval is not out of band, or is not bound to the same
      amount, payee and payee account as the payment;
    * ``amount_usd`` reaches the dual-approval threshold and fewer than the
      required number of distinct approvers are present, an approver repeats,
      or the requester is among the approvers.

    The dual-approval threshold defaults to the platform's own limit. A
    ``dual_control_threshold_usd`` given here can only be stricter (the lower value
    applies), and ``required_approvers`` can only raise the count (the platform's
    number, or 2 if it sent none, is the floor).

    ``budget_verifier`` checks ``erp_clearance.confirmation_token`` and binds
    ``request_id``, ``clearance_id`` and ``amount_usd``. ``approval_verifier``
    checks every ``approval_token`` and binds ``request_id``, ``approver_id`` and
    ``amount_usd``. A configured verifier makes its token mandatory. Passing local
    checks never releases a tool: the bundle verdict and its Ed25519 receipt are
    still required. Hosts keep token issuance, key custody, ledger authorization
    and the tool implementations.
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
        dual_control_threshold_usd: float | None = None,
        required_approvers: int | None = None,
        provider_key: str | None = None,
        provider_name: str | None = None,
    ) -> None:
        if dual_control_threshold_usd is not None and (
            isinstance(dual_control_threshold_usd, bool)
            or not isinstance(dual_control_threshold_usd, (int, float))
            or not math.isfinite(dual_control_threshold_usd)
            or dual_control_threshold_usd < 0
        ):
            raise ValueError("dual_control_threshold_usd must be None or a finite, non-negative number")
        if required_approvers is not None and (
            isinstance(required_approvers, bool)
            or not isinstance(required_approvers, int)
            or required_approvers < 1
        ):
            raise ValueError("required_approvers must be None or a positive integer")
        for name, verifier in (
            ("budget_verifier", budget_verifier),
            ("approval_verifier", approval_verifier),
        ):
            if verifier is not None and not isinstance(verifier, EvidenceVerifier):
                raise TypeError(f"{name} must implement EvidenceVerifier")
        self._budget_verifier = budget_verifier
        self._approval_verifier = approval_verifier
        self._local_threshold = None if dual_control_threshold_usd is None else float(dual_control_threshold_usd)
        self._local_required = required_approvers
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
                f"(fields: {', '.join(fields)}); supply the full platform context."
            )

        amount = evidence.amount_usd
        limits, clearance, exposure = evidence.platform_limits, evidence.erp_clearance, evidence.vendor_exposure

        if amount > limits.per_order_usd:
            return "Requested amount exceeds the platform per-order limit; reduce the amount or obtain a higher limit."
        if clearance.status.strip().upper() != "OPEN":
            return "The ERP budget clearance is not open; obtain an open clearance before committing funds."
        if not all(
            _same_dimension(requested, cleared)
            for requested, cleared in (
                (evidence.cost_center, clearance.cost_center),
                (evidence.fiscal_period, clearance.fiscal_period),
                (evidence.spend_category, clearance.spend_category),
            )
        ):
            return (
                "The ERP clearance does not match the request's cost center, fiscal period and spend "
                "category; obtain a clearance for this spend."
            )
        if amount > clearance.balance_usd:
            return "Requested amount exceeds the remaining ERP clearance balance."
        if exposure.open_pending_paid_usd + amount > limits.per_vendor_cumulative_usd:
            return "Requested amount would push cumulative vendor exposure past the platform cap."

        if self._budget_verifier is not None:
            result = self._budget_verifier.verify(
                clearance.confirmation_token,
                expected={
                    "request_id": evidence.request_id,
                    "clearance_id": clearance.id,
                    "amount_usd": amount,
                },
            )
            if not result.valid:
                return f"Budget clearance token rejected: {result.reason}"

        approver_ids = [approval.approver_id for approval in evidence.approvals]
        for approval in evidence.approvals:
            if approval.approved_out_of_band is not True:
                return f"Approval by {approval.approver_id!r} was not obtained out of band."
            binding = approval.bound_to
            if not (
                claims_equal(binding.amount_usd, amount)
                and binding.payee == evidence.payee
                and binding.payee_account == evidence.payee_account
            ):
                return (
                    f"Approval by {approval.approver_id!r} is not bound to this payment's "
                    "amount, payee and payee account."
                )

        threshold = limits.dual_approval_threshold_usd
        if self._local_threshold is not None:
            threshold = min(threshold, self._local_threshold)
        required = max(limits.required_approvers or DEFAULT_REQUIRED_APPROVERS, self._local_required or 0)
        if amount >= threshold:
            if len(set(approver_ids)) != len(approver_ids):
                return "Dual control requires distinct approvers; an approver is listed more than once."
            if evidence.requested_by in approver_ids:
                return "Dual control requires approvers independent of the requester."
            if len(approver_ids) < required:
                return (
                    f"Dual control requires {required} distinct approvers at or above "
                    f"{threshold:,.2f} USD; {len(approver_ids)} supplied."
                )
        if self._approval_verifier is not None:
            for approval in evidence.approvals:
                result = self._approval_verifier.verify(
                    approval.approval_token,
                    expected={
                        "request_id": evidence.request_id,
                        "approver_id": approval.approver_id,
                        "amount_usd": amount,
                    },
                )
                if not result.valid:
                    return f"Approval token for approver {approval.approver_id!r} rejected: {result.reason}"
        return None


__all__ = [
    "AUTHORIZATION_GATE_BUNDLE_ID",
    "AUTHORIZATION_GATE_TOOL_NAMES",
    "ApprovalBinding",
    "ApprovalEvidence",
    "AuthorizationEvidence",
    "AuthorizationGateAgent",
    "ErpClearance",
    "PlatformLimits",
    "VendorExposure",
]
