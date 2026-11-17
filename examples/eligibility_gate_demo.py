"""Contractual & Eligibility gate: local deterministic exclusions, then the live bundle.

``EligibilityGateAgent`` governs ``confirm_eligibility`` (and ``bind_contract_terms``)
for an insurance claim. It evaluates your ``ExclusionRuleSet`` locally and checks the
platform-confirmed claim evidence before it contacts the backend:

1. Local pre-check (no API key): the facts say the loss falls outside the policy term, so
   the host exclusion rule ``R-LOSS-OUTSIDE-TERM`` blocks the claim. No evaluation request
   is made.
2. Live BLOCK: nothing local objects and the agent's exclusion flag is empty, but the
   verified claim file shows an unauthorised structural alteration. The
   ``ramen__contractual_eligibility_invariance`` bundle blocks the approval (CEE-02) and
   routes it to human adjudication.
3. Live ALLOW: policy record, out-of-band inspection token and platform schedule all
   support the approval. The bundle allows it, the receipt verifies, and the simulated
   tool runs.

The tool is simulated: it records the approval and notifies no one.

Run from the repository root::

    python examples/eligibility_gate_demo.py                      # Stage 1 only
    RAMEN_API_KEY=... python examples/eligibility_gate_demo.py    # all three stages
"""

from __future__ import annotations

import sys
from typing import Any

from langchain_core.tools import tool

from ramen_foundry import EligibilityGateAgent, ExclusionRuleSet

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

TITLE = "Eligibility gate: local exclusions and platform-confirmed claim evidence"
APPROVED_AMOUNT_USD = 63_000.0

EXCLUSION_RULES = ExclusionRuleSet.model_validate(
    {
        "required_facts": ["claim_id"],
        "rules": [
            {
                "rule_id": "R-LOSS-OUTSIDE-TERM",
                "description": "Date of loss outside the policy term",
                "all_of": [{"field": "loss_in_term", "op": "eq", "value": False}],
            },
            {
                "rule_id": "R-POLICY-NOT-ACTIVE",
                "description": "Policy status not active",
                "all_of": [{"field": "policy_status", "op": "ne", "value": "ACTIVE"}],
            },
            {
                "rule_id": "R-EXCLUSION-FLAGGED",
                "description": "Exclusion flagged on the claim",
                "all_of": [{"field": "exclusion_flag", "op": "eq", "value": True}],
            },
        ],
    }
)


def claim_facts(**overrides: Any) -> dict[str, Any]:
    """Flat facts for the host rules plus the platform-confirmed claim evidence."""

    facts: dict[str, Any] = {
        "claim_id": "CLM-9042",
        "loss_in_term": True,
        "policy_status": "ACTIVE",
        "exclusion_flag": False,
        "policy_system_record": {
            "effective": "2026-01-01",
            "expiry": "2026-12-31",
            "status": "ACTIVE",
            "date_of_loss": "2026-08-04",
            "confirmed_by": "policy system of record",
        },
        "inspection_evidence": {
            "token": "SS-9042",
            "issued_out_of_band_by": "independent inspection network",
            "verified_by_platform": True,
            "inspection_date": "2026-08-10",
            "scope": "units 12 to 20",
            "assessed_usd": 63_000,
        },
        "platform_schedule": {
            "source": "policy platform",
            "unit_block_limit_usd": 150_000,
            "remaining_aggregate_usd": 150_000,
            "deductible_usd": 0,
            "drawn_usd": 0,
        },
        "verified_file_findings": "File notes no excluded cause.",
    }
    facts.update(overrides)
    return facts


def claim(facts: dict[str, Any]) -> dict[str, Any]:
    """The arguments of ``confirm_eligibility``."""

    return {"subject_id": "CLM-9042", "approved_amount_usd": APPROVED_AMOUNT_USD, "eligibility_facts": facts}


EXCLUDED_FILE_FINDINGS = (
    "Verified file shows an unauthorised structural alteration of the sprinkler riser "
    "before the loss, a triggered exclusion."
)


def loss_outside_term() -> dict[str, Any]:
    """Stage 1: the facts say the loss is outside the policy term."""

    return claim(claim_facts(loss_in_term=False))


def excluded_by_verified_file() -> dict[str, Any]:
    """Stage 2: the verified claim file shows a triggered exclusion; the agent's flag list is empty."""

    return claim(claim_facts(verified_file_findings=EXCLUDED_FILE_FINDINGS))


def fully_evidenced() -> dict[str, Any]:
    """Stage 3: every platform-confirmed requirement is present."""

    return claim(claim_facts())


def run(client: GateRecordingClient, *, live: bool = True, provider: dict[str, str] | None = None) -> list[Outcome]:
    """Run the stages and return one outcome per scenario."""

    approved: list[dict[str, Any]] = []

    @tool
    def confirm_eligibility(subject_id: str, approved_amount_usd: float, eligibility_facts: dict[str, Any]) -> str:
        """Confirm eligibility (simulated: records the approval and notifies no one)."""
        approved.append({"subject_id": subject_id, "approved_amount_usd": approved_amount_usd})
        return f"[simulated] approved ${approved_amount_usd:,.2f} for {subject_id}"

    @tool
    def bind_contract_terms(subject_id: str, approved_amount_usd: float, eligibility_facts: dict[str, Any]) -> str:
        """Bind contract terms (simulated). Governed by the same local checks and bundle."""
        approved.append({"subject_id": subject_id, "approved_amount_usd": approved_amount_usd})
        return f"[simulated] bound terms for {subject_id}"

    options = provider or {}
    agent = EligibilityGateAgent(
        client=client,
        tools={"confirm_eligibility": confirm_eligibility, "bind_contract_terms": bind_contract_terms},
        exclusion_rules=EXCLUSION_RULES,
        provider_key=options.get("provider_key"),
        provider_name=options.get("provider_name"),
    )

    def scenario(stage: str, name: str, expected: str, arguments: dict[str, Any], **kwargs: Any) -> Outcome:
        return run_scenario(
            agent, client, approved, stage=stage, scenario=name, expected=expected,
            tool_name="confirm_eligibility", arguments=arguments, **kwargs,
        )

    outcomes = [
        scenario("1 Local", "Loss outside the policy term (host rule)", LOCAL, loss_outside_term(),
                 local_fragment="R-LOSS-OUTSIDE-TERM"),
    ]
    if live:
        outcomes.append(scenario("2 Live", "Exclusion shown only by the verified file", BLOCK, excluded_by_verified_file()))
        outcomes.append(scenario("3 Live", "Full claim evidence, $63k approval", ALLOW, fully_evidenced()))
    return outcomes


def main() -> int:
    return gate_main(TITLE, run)


if __name__ == "__main__":
    sys.exit(main())
