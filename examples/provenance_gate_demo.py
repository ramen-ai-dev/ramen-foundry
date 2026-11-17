"""Compliance & Provenance gate: evidence verified locally, report written by the template.

``ProvenanceGateAgent`` governs ``submit_filing`` and ``release_report``. Evidence is a set
of Schema V5 Ed25519 receipts. The template verifies each one locally (signature, signed
ALLOW verdict, freshness, and binding to the subject, record type and artifact) and writes
the ``verifier_report`` the bundle evaluates itself. A caller cannot supply one.

1. Local pre-check (no API key): the caller attaches its own ``verifier_report``. The call is
   blocked as self-attestation before any evaluation request. With a key, a second local
   scenario tampers with a receipt after signing.
2. Live BLOCK: one evidence set is offered for a batch of subjects. The
   ``ramen__compliance_provenance_invariance`` bundle blocks it (CPE-02).
3. Live ALLOW: receipts bound to the subject, record type and artifact. The bundle allows
   the filing, the receipt verifies, and the simulated tool receives the derived report.

With a key the demo first mints two evidence receipts by evaluating a benign
``inspect_directory`` call under ``ramen__shield_core_it``. In production the receipts come
from your own governed steps.

The filing tool is simulated: it records the submission and files nothing.

Run from the repository root::

    python examples/provenance_gate_demo.py                      # Stage 1 only
    RAMEN_API_KEY=... python examples/provenance_gate_demo.py    # all stages
"""

from __future__ import annotations

import hashlib
import json
import sys
from typing import Any

from langchain_core.tools import tool

from ramen_foundry import ProvenanceGateAgent
from ramen_foundry.core.receipts import signed_verdict_problem

try:  # Supports both direct execution and imports from repository tests.
    from examples._gate_showcase import (
        ALLOW,
        BLOCK,
        LOCAL,
        ExpectationError,
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
        ExpectationError,
        GateRecordingClient,
        Outcome,
        gate_main,
        run_scenario,
    )

TITLE = "Provenance gate: Schema V5 receipts as evidence, derived verifier report"
RECORD_TYPE = "KYC_CLEARANCE"
SUBJECT_ID = "IND-5120"
ARTIFACT_SHA256 = hashlib.sha256(b"kyc-onboarding-record-IND-5120-v1").hexdigest()
EVIDENCE_BUNDLE = "ramen__shield_core_it"


def evidence_call(step: str) -> str:
    """The exact text evaluated to mint one evidence receipt for this submission."""

    return json.dumps(
        {
            "tool": "inspect_directory",
            "arguments": {
                "path": "./build",
                "step": step,
                "subject_id": SUBJECT_ID,
                "record_type": RECORD_TYPE,
                "artifact_sha256": ARTIFACT_SHA256,
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def submission(items: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    """A submission bound to the subject, the record type and the artifact."""

    payload: dict[str, Any] = {
        "record_type": RECORD_TYPE,
        "subject_id": SUBJECT_ID,
        "artifact_sha256": ARTIFACT_SHA256,
        "evidence": items,
    }
    payload.update(extra)
    return payload


def self_attested() -> dict[str, Any]:
    """Stage 1: the caller writes its own verifier report. Rejected before any receipt is read."""

    return submission([], verifier_report={"status": "valid", "bound_subject": SUBJECT_ID})


def batch_without_per_subject_evidence(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Stage 2: one evidence set offered for a batch of subjects."""

    return submission(
        items,
        batch_subjects=[SUBJECT_ID, "IND-7777"],
        note="one evidence set covers both subjects in this batch",
    )


def mint_evidence(client: GateRecordingClient, step: str) -> dict[str, Any]:
    """Obtain one real ALLOW receipt and require that it verifies."""

    text = evidence_call(step)
    result = client.evaluate_compliance(
        text, bundle_ids=[EVIDENCE_BUNDLE], context={"tool_name": "inspect_directory"}
    )
    if result.get("allowed") is not True:
        raise ExpectationError(f"evidence step {step!r} was not allowed: {result.get('steering')}")
    problem = signed_verdict_problem(result, text, 1, None)
    if problem is not None:
        raise ExpectationError(f"evidence receipt for {step!r} did not verify: {problem}")
    return {"payload": text, "receipt": result["data"]["receipt"]}


def run(client: GateRecordingClient, *, live: bool = True, provider: dict[str, str] | None = None) -> list[Outcome]:
    """Run the stages and return one outcome per scenario."""

    filed: list[dict[str, Any]] = []

    @tool
    def submit_filing(
        record_type: str,
        subject_id: str,
        artifact_sha256: str,
        evidence: list[dict[str, Any]],
        verifier_report: dict[str, Any] | None = None,
    ) -> str:
        """Submit a regulated filing (simulated: records the submission and files nothing)."""
        filed.append({"subject_id": subject_id, "verifier_report": verifier_report})
        return f"[simulated] filed {record_type} for {subject_id}"

    options = provider or {}
    agent = ProvenanceGateAgent(
        client=client,
        tools={"submit_filing": submit_filing},
        token_required_record_types=[RECORD_TYPE],
        provider_key=options.get("provider_key"),
        provider_name=options.get("provider_name"),
    )

    def scenario(stage: str, name: str, expected: str, arguments: dict[str, Any], **kwargs: Any) -> Outcome:
        return run_scenario(
            agent, client, filed, stage=stage, scenario=name, expected=expected,
            tool_name="submit_filing", arguments=arguments, **kwargs,
        )

    outcomes = [
        scenario("1 Local", "Caller-written verifier_report", LOCAL, self_attested(),
                 local_fragment="Self-attestation is not accepted"),
    ]
    if not live:
        return outcomes

    evidence = [mint_evidence(client, "scan"), mint_evidence(client, "review")]
    tampered = {"payload": evidence[0]["payload"] + " ", "receipt": evidence[0]["receipt"]}
    outcomes.append(
        scenario("1 Local", "Receipt tampered after signing", LOCAL, submission([tampered]),
                 local_fragment="payload_hash")
    )
    outcomes.append(
        scenario("2 Live", "One evidence set for a batch of subjects", BLOCK, batch_without_per_subject_evidence(evidence))
    )
    allowed = scenario("3 Live", "Receipts bound to subject, type and artifact", ALLOW, submission(evidence))
    report = filed[-1]["verifier_report"] or {}
    receipts = ", ".join(str(r["id"])[:8] for r in report.get("receipts", []))
    outcomes.append(
        Outcome(
            allowed.stage, allowed.scenario, allowed.expected, allowed.decision, allowed.receipt, allowed.latency_ms,
            f"tool ran once with derived report: bound_subject={report.get('bound_subject')}, "
            f"record_type={report.get('record_type')}, receipts=[{receipts}]",
        )
    )
    return outcomes


def main() -> int:
    return gate_main(TITLE, run)


if __name__ == "__main__":
    sys.exit(main())
