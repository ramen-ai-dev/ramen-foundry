"""Offline tests for the keyless trial adapter and the receipt metadata on tool messages."""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import tool
from ramen_ai import RamenClient, TrialLimitError

import ramen_foundry
from ramen_foundry import DevboxShieldAgent, IndustrialAutomationAgent, TrialComplianceClient, describe_trial_limit
from tests.gate_fixtures import FakeClient
from tests.trial_fixtures import TrialServer

ROOT = Path(__file__).resolve().parents[1]
SAFE = {
    "robot_id": "COBOT-1",
    "action_type": "PICK_AND_PLACE",
    "target_object": "bracket",
    "destination_target": "fixture A3",
    "commanded_velocity_mps": 0.25,
    "commanded_force_nm": 35.0,
    "human_proximity_meters": 1.8,
    "active_hazard_flags": [],
    "scene_context_id": "S-1",
}
UNSAFE = {**SAFE, "destination_target": "active burner", "scene_context_id": "S-2"}


def _agent(client: Any, ran: list[str]) -> IndustrialAutomationAgent:
    @tool
    def dispatch_manipulation(
        robot_id: str,
        action_type: str,
        target_object: str,
        destination_target: str,
        commanded_velocity_mps: float,
        commanded_force_nm: float,
        human_proximity_meters: float,
        active_hazard_flags: list[str],
        scene_context_id: str,
    ) -> str:
        """Canary."""
        ran.append(scene_context_id)
        return "ran"

    return IndustrialAutomationAgent(client=client, tools={"dispatch_manipulation": dispatch_manipulation})


def _metadata(command: Any) -> dict[str, Any]:
    return dict(command.update["messages"][-1].response_metadata)


def test_the_raw_trial_client_is_not_an_evaluate_compliance_client() -> None:
    assert not hasattr(RamenClient.trial(), "evaluate_compliance")


def test_allow_and_block_pass_through_a_governed_template_with_trial_receipts() -> None:
    server, ran = TrialServer(), []
    with server.client() as client:
        agent = _agent(client, ran)
        allowed = agent.execute("dispatch_manipulation", SAFE)
        blocked = agent.execute("dispatch_manipulation", UNSAFE)
    assert allowed.update["governance_error"] is None and ran == ["S-1"]
    assert blocked.update["governance_error"] == "Route this call to human review."
    assert allowed.goto == blocked.goto == "__end__"
    for command, verdict in ((allowed, 1), (blocked, 0)):
        metadata = _metadata(command)
        assert metadata["receipt_verified"] is True and metadata["receipt_problem"] is None
        assert metadata["receipt"]["id"].startswith("trial_")
        assert json.loads(metadata["receipt"]["canonical_payload"])["verdict"] == verdict
        assert metadata["policy_ids"] == ["trial-policy-1"]


def test_requests_carry_one_bundle_no_credentials_no_context_and_fit_the_limit() -> None:
    server = TrialServer()
    with server.client() as client:
        _agent(client, []).execute("dispatch_manipulation", SAFE)
        client.evaluate_compliance(
            "inspect",
            bundle_ids=["ramen__shield_core_it"],
            context={"tool_name": "x"},
            provider_key="provider-secret-value",
            provider_name="openai",
        )
    for request, content in zip(server.requests, server.bodies, strict=True):
        assert set(request.headers) & {"authorization", "x-provider-key", "x-provider"} == set()
        body = json.loads(content)
        assert set(body) == {"input", "bundle_ids"} and len(body["bundle_ids"]) == 1
        assert len(content) <= 2048
    assert b"provider-secret-value" not in b"".join(server.bodies)


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"bundle_ids": None}, "exactly one bundle"),
        ({"bundle_ids": ["a", "b"]}, "exactly one bundle"),
        ({"bundle_ids": ["a"], "policy_ids": ["p"]}, "policy_ids"),
    ],
)
def test_unsupported_arguments_fail_before_any_request(kwargs: dict[str, Any], fragment: str) -> None:
    server = TrialServer()
    with server.client() as client, pytest.raises(ValueError, match=fragment):
        client.evaluate_compliance("text", **kwargs)
    assert server.requests == []


def test_an_exhausted_allowance_is_kept_typed_and_fails_closed() -> None:
    server, ran = TrialServer(limit_after=0), []
    with server.client() as client:
        command = _agent(client, ran).execute("dispatch_manipulation", SAFE)
        limit = client.last_limit
    assert ran == []
    assert "Governance evaluation unavailable" in command.update["governance_error"]
    assert _metadata(command) == {}
    assert isinstance(limit, TrialLimitError) and limit.scope == "network"
    text = describe_trial_limit(limit)
    assert "scope: network" in text and "2030-01-01 00:00 UTC" in text and "https://ramenai.dev" in text


def test_describe_trial_limit_ignores_a_signup_url_outside_ramenai() -> None:
    server = TrialServer(limit_after=0)
    with server.client() as client, pytest.raises(TrialLimitError) as caught:
        client.evaluate_compliance("x", bundle_ids=["ramen__shield_core_it"])
    caught.value.signup_url = "https://evil.example/pricing"
    assert "evil.example" not in describe_trial_limit(caught.value)


def test_an_oversized_call_is_refused_without_a_request() -> None:
    server, ran = TrialServer(), []
    with server.client() as client:
        command = _agent(client, ran).execute("dispatch_manipulation", {**SAFE, "target_object": "x" * 2100})
    assert server.requests == [] and ran == []
    assert "limit is 2048" in command.update["governance_error"]


def test_the_adapter_records_the_remaining_allowance() -> None:
    server = TrialServer(remaining=7)
    with server.client() as client:
        client.evaluate_compliance("hello", bundle_ids=["ramen__shield_core_it"])
        assert client.last_quota == {"limit": 10, "remaining": 7, "reset_epoch_seconds": 1893456000}


def test_an_existing_client_and_options_cannot_be_combined() -> None:
    with pytest.raises(TypeError):
        TrialComplianceClient(RamenClient.trial(), timeout=5.0)


def test_the_adapter_is_exported_lazily_and_imports_neither_langgraph_nor_pandas() -> None:
    assert ramen_foundry.TrialComplianceClient is TrialComplianceClient
    code = textwrap.dedent(
        """
        import sys
        import ramen_foundry.trial
        loaded = {"langgraph", "pandas"} & set(sys.modules)
        raise SystemExit(f"unexpected imports: {loaded}" if loaded else 0)
        """
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout


# --- receipt metadata on tool messages (all templates share RamenToolNode) ---


def test_metadata_reports_a_verified_allow_and_a_verified_block() -> None:
    for allowed in (True, False):
        ran: list[str] = []
        command = _agent(FakeClient(allowed=allowed), ran).execute("dispatch_manipulation", SAFE)
        metadata = _metadata(command)
        assert metadata["receipt_verified"] is True
        assert metadata["receipt_problem"] is None
        assert set(metadata) == {"receipt", "receipt_verified", "policy_ids", "receipt_problem"}
        assert (ran == ["S-1"]) is allowed


def test_metadata_reports_why_an_allow_was_not_verified() -> None:
    class UnverifiedClient(FakeClient):
        def evaluate_compliance(self, input_text: str, **kwargs: Any) -> dict[str, Any]:
            verdict = super().evaluate_compliance(input_text, **kwargs)
            return {**verdict, "receipt_verified": False, "receipt_reason": "client could not verify"}

    ran: list[str] = []
    command = _agent(UnverifiedClient(), ran).execute("dispatch_manipulation", SAFE)
    metadata = _metadata(command)
    assert ran == []
    assert metadata["receipt_verified"] is False
    assert metadata["receipt_problem"] == "client could not verify"
    assert metadata["receipt"] is not None


def test_metadata_is_empty_when_nothing_was_evaluated() -> None:
    client = FakeClient()
    command = _agent(client, []).execute("dispatch_manipulation", {**SAFE, "action_type": "STAB"})
    assert client.calls == [] and _metadata(command) == {}


def test_a_devbox_shield_message_carries_the_same_keys() -> None:
    @tool
    def inspect_directory(path: str) -> str:
        """Inspect."""
        return path

    agent = DevboxShieldAgent(client=FakeClient(), tools={"inspect_directory": inspect_directory})
    command = agent.execute({"name": "inspect_directory", "arguments": {"path": "./build"}, "tool_call_id": "t1"})
    assert _metadata(command)["receipt_verified"] is True
