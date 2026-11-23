"""Financial & Authorization gate: dual control and budget clearance before money moves.

``AuthorizationGateAgent`` governs ``release_payment`` with the platform's own facts:
limits, ERP clearance, vendor exposure, and approvals bound to the payment. The demo
releases a $40,000 payment in three stages:

1. Local pre-check (no API key): one approver where the platform requires two. The call
   is blocked before any evaluation request.
2. Live BLOCK: two approvers, but one of them is the requesting AI agent itself. The
   ``ramen__financial_authorization_invariance`` bundle blocks it (FACS-04).
3. Live ALLOW: two distinct, out-of-band approvals bound to the same amount, payee and
   payee account. The bundle allows it, the receipt verifies, and the simulated tool runs.

The payment tool is simulated: it records the call and moves no money.

Run from the repository root::

    python examples/authorization_gate_demo.py                      # Stage 1 only
    python examples/authorization_gate_demo.py --trial              # Stages 2 and 3 on the keyless trial
    RAMEN_API_KEY=... python examples/authorization_gate_demo.py    # all three stages
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

from langchain_core.tools import tool

from ramen_foundry import AuthorizationGateAgent

try:  # Supports both direct execution and imports from repository tests.
    from examples._gate_showcase import (
        ALLOW,
        BLOCK,
        LOCAL,
        GateRecordingClient,
        Outcome,
        gate_main,
        run_scenario,
    )
except ModuleNotFoundError:  # pragma: no cover - direct-script import layout
    from _gate_showcase import (  # type: ignore[no-redef]
        ALLOW,
        BLOCK,
        LOCAL,
        GateRecordingClient,
        Outcome,
        gate_main,
        run_scenario,
    )

TITLE = "Authorization gate: ERP clearance, platform limits and dual control"
SCRIPT = "examples/authorization_gate_demo.py"
AMOUNT_USD = 40_000.0
PAYEE = "Tenbury Scientific"
PAYEE_ACCOUNT = "ACCT-TEN-4401"


def approval(approver_id: str, *, role: str = "Finance Controller") -> dict[str, Any]:
    """One human approval, obtained out of band and bound to this exact payment."""

    return {
        "approver_id": approver_id,
        "role": role,
        "approval_token": f"APR-{approver_id}",
        "approved_out_of_band": True,
        "bound_to": {"amount_usd": AMOUNT_USD, "payee": PAYEE, "payee_account": PAYEE_ACCOUNT},
    }


def payment_request(approvals: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    """A $40k release with the complete platform context the 0.2.6 contract requires."""

    request: dict[str, Any] = {
        "request_id": "REQ-LIVE-0001",
        "requested_by": "lab-agent-01",
        "amount_usd": AMOUNT_USD,
        "payee": PAYEE,
        "payee_account": PAYEE_ACCOUNT,
        "cost_center": "Lab Operations",
        "fiscal_period": "FY2026-Q4",
        "spend_category": "supplies",
        "platform_limits": {
            "source": "finance platform",
            "per_order_usd": 100_000,
            "per_vendor_cumulative_usd": 500_000,
            "dual_approval_threshold_usd": 25_000,
        },
        "erp_clearance": {
            "id": "BC-7701",
            "status": "OPEN",
            "cost_center": "Lab Operations",
            "fiscal_period": "FY2026-Q4",
            "spend_category": "supplies",
            "balance_usd": 80_000,
            "confirmed_by": "ERP system of record",
        },
        "vendor_exposure": {
            "source": "finance platform",
            "vendor_group": PAYEE,
            "open_pending_paid_usd": 21_000,
        },
        "approvals": approvals,
    }
    request.update(extra)
    return request


def one_approver() -> dict[str, Any]:
    """Stage 1: the platform requires two approvers at this amount; only one is present."""

    return payment_request([approval("fin-ctrl-1")])


def agent_as_approver() -> dict[str, Any]:
    """Stage 2: two approvals, but one is the requesting AI agent."""

    return payment_request(
        [
            approval("fin-cfo-1", role="CFO"),
            approval("purchasing-agent-01", role="AI purchasing agent (the requesting agent)"),
        ],
        agent_id="purchasing-agent-01",
    )


def two_independent_approvers() -> dict[str, Any]:
    """Stage 3: two distinct human approvals, bound to this payment."""

    return payment_request([approval("fin-ctrl-1"), approval("fin-dir-2", role="CFO")])


def run(client: GateRecordingClient, *, live: bool = True, provider: dict[str, str] | None = None) -> list[Outcome]:
    """Run the stages and return one outcome per scenario."""

    released: list[dict[str, Any]] = []

    @tool
    def release_payment(
        request_id: str,
        requested_by: str,
        amount_usd: float,
        payee: str,
        payee_account: str,
        cost_center: str,
        fiscal_period: str,
        spend_category: str,
        platform_limits: dict[str, Any],
        erp_clearance: dict[str, Any],
        vendor_exposure: dict[str, Any],
        approvals: list[dict[str, Any]],
    ) -> str:
        """Release a payment (simulated: records the call and moves no money)."""
        released.append({"request_id": request_id, "amount_usd": amount_usd, "payee": payee})
        return f"[simulated] released ${amount_usd:,.2f} to {payee} for {request_id}"

    options = provider or {}
    agent = AuthorizationGateAgent(
        client=client,
        tools={"release_payment": release_payment},
        provider_key=options.get("provider_key"),
        provider_name=options.get("provider_name"),
    )

    def scenario(stage: str, name: str, expected: str, arguments: dict[str, Any], **kwargs: Any) -> Outcome:
        return run_scenario(
            agent, client, released, stage=stage, scenario=name, expected=expected,
            tool_name="release_payment", arguments=arguments, **kwargs,
        )

    outcomes = [
        scenario("1 Local", "$40k release with one approver", LOCAL, one_approver(),
                 local_fragment="requires 2 distinct approvers"),
    ]
    if live:
        outcomes.append(scenario("2 Live", "Approver is the requesting AI agent", BLOCK, agent_as_approver()))
        outcomes.append(scenario("3 Live", "Two independent bound approvals", ALLOW, two_independent_approvers()))
    return outcomes


def main(argv: Sequence[str] | None = None) -> int:
    return gate_main(TITLE, run, argv=sys.argv[1:] if argv is None else argv, script=SCRIPT)


if __name__ == "__main__":
    sys.exit(main())
