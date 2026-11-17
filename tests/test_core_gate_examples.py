"""Offline tests for the Core Architecture gate demos in examples/."""

from __future__ import annotations

import importlib
import re
import socket
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import mock

import pytest

from examples import _gate_showcase as showcase
from examples import (
    authorization_gate_demo,
    eligibility_gate_demo,
    kinematic_gate_demo,
    provenance_gate_demo,
)
from examples.benchmark_asimov_robotics import build_benchmark_cases
from ramen_foundry import manipulation_payload_problem
from tests.live import payloads as live
from tests.live.test_live_eligibility import RULES as LIVE_ELIGIBILITY_RULES
from tests.receipt_fixtures import signed_data

REPO = Path(__file__).resolve().parents[1]
SECRET = "test-key-SECRET-0123456789"
BYOK_SECRET = "provider-key-SECRET-9876543210"


class ScriptedClient(showcase.GateRecordingClient):
    """A recording client whose verdict is decided from the evaluated text, with real receipts."""

    def __init__(self, decide: Callable[[str], bool]) -> None:
        super().__init__(SECRET, min_interval=0.0)
        self._decide = decide
        self.calls: list[dict[str, Any]] = []

    def _evaluate(self, input_text: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        allowed = self._decide(input_text)
        return {
            "allowed": allowed,
            "receipt_verified": True,
            "receipt_reason": None,
            "steering": None if allowed else "Blocked by the bundle: route to human review.",
            "data": signed_data(input_text, allowed=allowed),
        }


def _blocked_when(marker: str) -> Callable[[str], bool]:
    return lambda text: marker not in text


# module, verdict rule, evaluations a full run makes, live outcome count
DEMOS: dict[str, tuple[ModuleType, Callable[[str], bool], int]] = {
    "authorization": (authorization_gate_demo, _blocked_when("(the requesting agent)"), 2),
    "eligibility": (eligibility_gate_demo, _blocked_when("unauthorised structural alteration"), 2),
    "provenance": (provenance_gate_demo, _blocked_when("batch_subjects"), 4),  # two evidence mints
    "kinematic": (kinematic_gate_demo, _blocked_when("active burner"), 2),
}
IDS = list(DEMOS)


def _demo(name: str) -> tuple[ModuleType, Callable[[str], bool], int]:
    return DEMOS[name]


@pytest.mark.parametrize("name", IDS)
def test_stage_one_runs_without_a_key_or_any_evaluation(name: str) -> None:
    module, _, _ = _demo(name)
    with showcase.OfflineGateClient() as client:
        outcomes = module.run(client, live=False)

    assert outcomes, "stage 1 must produce at least one outcome"
    assert {o.expected for o in outcomes} == {showcase.LOCAL}
    assert all(o.decision == "BLOCKED locally" for o in outcomes)
    assert client.evaluations == []


@pytest.mark.parametrize("name", IDS)
def test_a_full_run_ends_in_one_block_and_one_allow_with_verified_receipts(name: str) -> None:
    module, decide, evaluations = _demo(name)
    client = ScriptedClient(decide)

    outcomes = module.run(client, live=True)

    live_outcomes = [o for o in outcomes if o.decision.endswith("(live)")]
    assert [o.decision for o in live_outcomes] == ["BLOCK (live)", "ALLOW (live)"]
    assert all("Schema V5 Ed25519 verified, kid=ramen_pk_v1" in o.receipt for o in live_outcomes)
    assert len(client.evaluations) == evaluations
    assert [o.stage[0] for o in outcomes] == sorted(o.stage[0] for o in outcomes)
    assert "tool ran once" in live_outcomes[1].note


@pytest.mark.parametrize("name", IDS)
def test_an_unexpected_verdict_fails_the_run_instead_of_passing_quietly(
    name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    module, _, _ = _demo(name)
    everything_allowed = ScriptedClient(lambda text: True)

    with pytest.raises(showcase.ExpectationError, match="expected BLOCK but the bundle returned ALLOW"):
        module.run(everything_allowed, live=True)

    code = showcase.gate_main(
        module.TITLE, module.run, environment={"RAMEN_API_KEY": SECRET},
        client_factory=lambda key: ScriptedClient(lambda text: True),
    )
    assert code == 1
    assert "UNEXPECTED RESULT" in capsys.readouterr().out


@pytest.mark.parametrize("name", IDS)
def test_main_without_a_key_runs_stage_one_exits_zero_and_never_opens_a_socket(
    name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    module, _, _ = _demo(name)

    def no_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a demo attempted network access without a key")

    with mock.patch.object(socket.socket, "connect", no_network):
        code = showcase.gate_main(module.TITLE, module.run, environment={})

    out = capsys.readouterr().out
    assert code == 0
    assert "Stage 1 only" in out
    assert "Set RAMEN_API_KEY to run Stages 2 and 3" in out
    assert "BLOCKED locally" in out


@pytest.mark.parametrize("name", IDS)
def test_main_with_a_key_prints_the_table_and_never_the_key(
    name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    module, decide, _ = _demo(name)
    clients: list[ScriptedClient] = []

    def factory(key: str) -> ScriptedClient:
        assert key == SECRET
        clients.append(ScriptedClient(decide))
        return clients[-1]

    code = showcase.gate_main(
        module.TITLE, module.run,
        environment={"RAMEN_API_KEY": SECRET, "OPENAI_API_KEY": BYOK_SECRET},
        client_factory=factory,
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "BLOCK (live)" in captured.out and "ALLOW (live)" in captured.out
    assert "Provider mode: OpenAI BYOK" in captured.out
    for secret in (SECRET, BYOK_SECRET):
        assert secret not in captured.out + captured.err
    # The BYOK settings reach the evaluation of the governed call.
    governed = [c for c in clients[0].calls if c.get("bundle_ids") != ["ramen__shield_core_it"]]
    assert governed and all(c["provider_name"] == "openai" and c["provider_key"] == BYOK_SECRET for c in governed)


def test_an_api_error_is_reported_without_the_key(capsys: pytest.CaptureFixture[str]) -> None:
    class Exploding(ScriptedClient):
        def _evaluate(self, input_text: str, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError(f"HTTP 500 for key {SECRET}")

    code = showcase.gate_main(
        authorization_gate_demo.TITLE, authorization_gate_demo.run,
        environment={"RAMEN_API_KEY": SECRET}, client_factory=lambda key: Exploding(lambda text: True),
    )

    out = capsys.readouterr().out
    assert code == 1
    assert "UNEXPECTED RESULT" in out and "Governance evaluation unavailable" in out
    assert SECRET not in out and "***" in out


def test_an_error_outside_the_governed_call_is_reported_without_the_key(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def factory(key: str) -> ScriptedClient:
        raise RuntimeError(f"cannot connect with {key}")

    code = showcase.gate_main(
        authorization_gate_demo.TITLE, authorization_gate_demo.run,
        environment={"RAMEN_API_KEY": SECRET}, client_factory=factory,
    )

    out = capsys.readouterr().out
    assert code == 1
    assert "ERROR: RuntimeError" in out
    assert SECRET not in out


def test_the_provenance_demo_hands_the_derived_report_to_the_tool() -> None:
    _, decide, _ = _demo("provenance")
    outcomes = provenance_gate_demo.run(ScriptedClient(decide), live=True)

    allowed = outcomes[-1]
    assert "bound_subject=IND-5120" in allowed.note
    assert "record_type=KYC_CLEARANCE" in allowed.note
    assert any(o.scenario == "Receipt tampered after signing" and o.decision == "BLOCKED locally" for o in outcomes)


@pytest.mark.parametrize("name", IDS)
def test_importing_a_demo_has_no_side_effects(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    module, _, _ = _demo(name)
    importlib.reload(module)
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


# --------------------------------------------------------------- payload drift guard
# The demos inline their payloads so readers see the schema. These tests pin them to the
# payloads that passed the serial live gate (tests/live/payloads.py and the ASIMOV cases).


def test_authorization_payloads_match_the_live_gate() -> None:
    demo = authorization_gate_demo
    assert demo.two_independent_approvers() == live.authorization_dual(40_000.0)
    assert demo.agent_as_approver() == live.authorization_agent_as_approver()
    assert demo.AMOUNT_USD == 40_000.0


def test_eligibility_payloads_and_rules_match_the_live_gate() -> None:
    demo = eligibility_gate_demo
    assert demo.fully_evidenced() == live.claim()
    assert demo.excluded_by_verified_file() == live.claim_with_excluded_file()
    assert demo.loss_outside_term() == live.claim(live.eligibility_facts(loss_in_term=False))
    assert demo.EXCLUSION_RULES.model_dump() == LIVE_ELIGIBILITY_RULES.model_dump()


def test_provenance_payloads_match_the_live_gate() -> None:
    demo = provenance_gate_demo
    items = [{"payload": "p", "receipt": {"id": "r"}}]
    assert (demo.RECORD_TYPE, demo.SUBJECT_ID, demo.ARTIFACT_SHA256) == (
        live.RECORD_TYPE, live.SUBJECT_ID, live.ARTIFACT_SHA256,
    )
    for step in ("scan", "review"):
        assert demo.evidence_call(step) == live.evidence_call(step)
    assert demo.submission(items) == live.submission(items)
    assert demo.batch_without_per_subject_evidence(items) == live.batch_submission(items)


def test_kinematic_payloads_match_the_asimov_cases_and_the_local_contract() -> None:
    demo = kinematic_gate_demo
    safe, hazard = build_benchmark_cases()[0], build_benchmark_cases()[1]
    assert demo.collaborative_transfer() == safe.payload and safe.expected_allowed
    assert demo.aerosol_on_burner() == hazard.payload and not hazard.expected_allowed
    assert manipulation_payload_problem(demo.collaborative_transfer()) is None
    assert manipulation_payload_problem(demo.aerosol_on_burner()) is None
    for name, malformed in demo.malformed_dispatches():
        assert manipulation_payload_problem(malformed) is not None, name


# ------------------------------------------------------------- documentation guards


def _python_block_under(markdown: str, heading: str) -> str:
    section = markdown.split(heading, 1)[1]
    block = re.search(r"```python\n(.*?)```", section, re.S)
    assert block, f"no python block under {heading}"
    return block.group(1)


def test_the_authorization_readme_snippet_is_self_contained(monkeypatch: pytest.MonkeyPatch) -> None:
    import ramen_ai

    monkeypatch.setenv("RAMEN_API_KEY", SECRET)
    snippet = _python_block_under((REPO / "README.md").read_text(encoding="utf-8"), "### authorization-gate")

    def allow_everything(self: Any, input_text: str, **kwargs: Any) -> dict[str, Any]:
        return {
            "allowed": True, "receipt_verified": True, "receipt_reason": None, "steering": None,
            "data": signed_data(input_text, allowed=True),
        }

    monkeypatch.setattr(ramen_ai.RamenClient, "evaluate_compliance", allow_everything)
    namespace: dict[str, Any] = {}
    exec(compile(snippet, "README.md:authorization-gate", "exec"), namespace)  # noqa: S102

    assert namespace["command"].update["governance_error"] is None
    assert namespace["command"].update["messages"][0].content.startswith("[simulated]")


@pytest.mark.parametrize("document", ["README.md", "examples/README.md"])
def test_every_example_file_the_docs_mention_exists(document: str) -> None:
    text = (REPO / document).read_text(encoding="utf-8")
    names = set(re.findall(r"examples/([A-Za-z0-9_./-]+\.(?:py|md|json))", text))
    if document == "examples/README.md":
        names |= set(re.findall(r"\]\(([A-Za-z0-9_-]+\.(?:py|md))\)", text))
    assert names, "the document should reference example files"
    for name in sorted(names):
        assert (REPO / "examples" / name).exists(), f"{document} references missing examples/{name}"


def test_the_core_architectures_table_covers_all_four_pillars() -> None:
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    section = readme.split("## Core Architectures", 1)[1].split("\n## ", 1)[0]
    for pillar, script in (
        ("Financial & Authorization", "authorization_gate_demo.py"),
        ("Contractual & Eligibility", "eligibility_gate_demo.py"),
        ("Compliance & Provenance", "provenance_gate_demo.py"),
        ("Cyber-Physical Kinematic", "kinematic_gate_demo.py"),
    ):
        assert pillar in section and script in section, pillar
    assert "benchmark_asimov_robotics.py" in section
    assert 'href="#core-architectures"' in readme.split("---", 1)[0]
