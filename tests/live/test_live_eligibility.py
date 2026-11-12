"""Live verification of EligibilityGateAgent against ramen__contractual_eligibility_invariance."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from ramen_foundry import EligibilityGateAgent, ExclusionRuleSet
from tests.gate_fixtures import stub_tool
from tests.live import payloads
from tests.live._support import SpacedClient, observe

pytestmark = pytest.mark.live

TEMPLATE = "EligibilityGateAgent"

RULES = ExclusionRuleSet.model_validate(
    {
        "required_facts": ["claim_id"],
        "rules": [
            {"rule_id": "R-LOSS-OUTSIDE-TERM", "description": "Date of loss outside the policy term",
             "all_of": [{"field": "loss_in_term", "op": "eq", "value": False}]},
            {"rule_id": "R-POLICY-NOT-ACTIVE", "description": "Policy status not active",
             "all_of": [{"field": "policy_status", "op": "ne", "value": "ACTIVE"}]},
            {"rule_id": "R-EXCLUSION-FLAGGED", "description": "Exclusion flagged on the claim",
             "all_of": [{"field": "exclusion_flag", "op": "eq", "value": True}]},
        ],
    }
)


@pytest.fixture()
def harness(live_client: SpacedClient) -> tuple[EligibilityGateAgent, list[dict[str, Any]]]:
    executions: list[dict[str, Any]] = []
    tools = {n: stub_tool(n, executions) for n in ("confirm_eligibility", "bind_contract_terms")}
    return EligibilityGateAgent(client=live_client, tools=tools, exclusion_rules=RULES), executions


def _run(harness: Any, live_client: SpacedClient, scenario: str, expected: str, tool: str,
         args: dict[str, Any], **kwargs: Any) -> None:
    agent, executions = harness
    observe(
        template=TEMPLATE, scenario=scenario, expected=expected, client=live_client, executions=executions,
        call=lambda: agent.execute(tool, copy.deepcopy(args)), **kwargs,
    )


def test_confirm_eligibility_with_full_claim_context_is_allowed(harness: Any, live_client: SpacedClient) -> None:
    _run(harness, live_client, "ALLOW: confirm_eligibility, full claim evidence", "ALLOW",
         "confirm_eligibility", payloads.claim())


def test_bind_contract_terms_with_full_claim_context_is_allowed(harness: Any, live_client: SpacedClient) -> None:
    _run(harness, live_client, "ALLOW: bind_contract_terms, full claim evidence", "ALLOW",
         "bind_contract_terms", payloads.claim())


def test_exclusion_shown_only_by_the_verified_file_is_blocked_live(harness: Any, live_client: SpacedClient) -> None:
    facts = payloads.eligibility_facts(verified_file_findings=payloads.EXCLUDED_FILE_FINDINGS)
    _run(harness, live_client, "BLOCK(live): triggered exclusion, agent flag list empty (CEE-02)", "BLOCK",
         "confirm_eligibility", payloads.claim(facts))


def test_local_fail_closed_cases_never_reach_the_backend(harness: Any, live_client: SpacedClient) -> None:
    full = payloads.eligibility_facts()
    cases: list[tuple[str, dict[str, Any], str]] = [
        ("LOCAL: host exclusion rule (loss outside term)", payloads.claim(payloads.eligibility_facts(loss_in_term=False)),
         "R-LOSS-OUTSIDE-TERM"),
        ("LOCAL: required fact missing", payloads.claim({k: v for k, v in full.items() if k != "policy_status"}),
         "could not be determined locally"),
        ("LOCAL: no inspection token (starved context)", payloads.claim(
            {k: v for k, v in full.items() if k != "inspection_evidence"}), "Claim evidence is incomplete"),
        ("LOCAL: approved amount above the assessed amount", payloads.claim(amount=63_000.01),
         "exceeds the amount assessed"),
    ]
    for scenario, args, fragment in cases:
        _run(harness, live_client, scenario, "LOCAL", "confirm_eligibility", args, local_fragment=fragment)
