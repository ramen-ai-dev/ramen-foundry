"""Tests for ProvenanceGateAgent: receipt-derived provenance with no forge dependency."""

from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
import socket
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest import mock

import httpx

import ramen_foundry.templates.provenance as provenance_module
from ramen_foundry import (
    PROVENANCE_GATE_BUNDLE_ID,
    ProvenanceGateAgent,
    ReceiptEvidenceVerifier,
)
from ramen_foundry.templates.provenance import DERIVED_KEYS
from tests.gate_fixtures import FakeClient, stub_tool
from tests.receipt_fixtures import TEST_PUBLIC_KEYS, sign_receipt

ARTIFACT = b"quarterly-report-v3"
DIGEST = hashlib.sha256(ARTIFACT).hexdigest()
RECORD_TYPE = "KYC_CLEARANCE"
SUBJECT = "IND-5120"


def _stamp(delta: timedelta = timedelta(0)) -> str:
    return (datetime.now(timezone.utc) + delta).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _item(
    *,
    verdict: int = 1,
    kid: str = "ramen_pk_v1",
    step: str = "scan",
    subject: str | None = SUBJECT,
    record_type: str | None = RECORD_TYPE,
    digest: str | None = DIGEST,
    timestamp: str | None = None,
) -> dict[str, Any]:
    arguments: dict[str, Any] = {"step": step}
    if subject is not None:
        arguments["subject_id"] = subject
    if record_type is not None:
        arguments["record_type"] = record_type
    if digest is not None:
        arguments["artifact_sha256"] = digest
    payload = json.dumps({"tool": "verify_step", "arguments": arguments}, sort_keys=True, separators=(",", ":"))
    return {"payload": payload, "receipt": sign_receipt(payload, verdict=verdict, kid=kid, timestamp=timestamp)}


def _args(evidence: Any, **overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "record_type": RECORD_TYPE,
        "subject_id": SUBJECT,
        "artifact_sha256": DIGEST,
        "evidence": evidence,
    }
    arguments.update(overrides)
    return arguments


class ProvenanceGateAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executions: list[dict[str, Any]] = []
        fields = ("record_type", "subject_id", "artifact_sha256", "evidence", "verifier_report")
        self.tools = {n: stub_tool(n, self.executions, *fields) for n in ("submit_filing", "release_report")}
        self.client = FakeClient()
        self.agent = self._agent()

    def _agent(self, **kwargs: Any) -> ProvenanceGateAgent:
        kwargs.setdefault("token_required_record_types", {RECORD_TYPE, "BENEFICIAL_OWNER"})
        return ProvenanceGateAgent(client=self.client, tools=self.tools, **kwargs)  # type: ignore[arg-type]

    def _call(self, arguments: Any, *, tool_name: str = "release_report", agent: Any = None) -> Any:
        return (agent or self.agent).execute(tool_name, arguments)

    def _blocked_locally(self, command: Any) -> str:
        self.assertEqual(self.client.calls, [], "the backend must not be contacted")
        self.assertEqual(self.executions, [])
        reason = command.update["governance_error"]
        self.assertIsNotNone(reason)
        return str(reason)

    def test_bound_fresh_receipts_reach_the_bundle_and_run(self) -> None:
        command = self._call(_args([_item(step="scan"), _item(step="review")]))

        self.assertIsNone(command.update["governance_error"])
        self.assertEqual(len(self.executions), 1)
        _, options = self.client.calls[0]
        self.assertEqual(options["bundle_ids"], [PROVENANCE_GATE_BUNDLE_ID])

    def test_both_tools_are_gated(self) -> None:
        for tool_name in ("release_report", "submit_filing"):
            with self.subTest(tool_name):
                self.assertIn("at least 1 verified receipt", self._blocked_locally(self._call(_args([]), tool_name=tool_name)))

    def test_the_verifier_report_is_derived_from_the_verified_receipts(self) -> None:
        items = [_item(step="scan"), _item(step="review")]
        self._call(_args(items), tool_name="submit_filing")

        evaluated = json.loads(self.client.calls[0][0])["arguments"]
        report = evaluated["verifier_report"]
        self.assertEqual(report["record_type"], RECORD_TYPE)
        self.assertEqual(report["bound_subject"], SUBJECT)
        self.assertEqual(report["bound_artifact_sha256"], DIGEST)
        self.assertEqual((report["status"], report["revoked"], report["superseded"]), ("valid", False, False))
        self.assertIs(report["verified_out_of_band"], True)
        self.assertIn("ramen_pk_v1", report["issuer"])
        signed_ids = [json.loads(i["receipt"]["canonical_payload"])["id"] for i in items]
        self.assertEqual([r["id"] for r in report["receipts"]], signed_ids)
        self.assertIs(evaluated["record_type_token_required"], True)
        self.assertIs(evaluated["tokens_submitted_unmodified"], True)
        # A tool that declares the parameter receives exactly what the backend evaluated.
        self.assertEqual(self.executions[0]["verifier_report"], report)
        self.assertEqual(evaluated["evidence"], items)

    def test_a_caller_written_report_is_rejected_as_self_attestation(self) -> None:
        forged = {"issuer": "me", "bound_subject": SUBJECT, "status": "valid"}
        for key in sorted(DERIVED_KEYS):
            with self.subTest(key):
                reason = self._blocked_locally(self._call(_args([_item()], **{key: forged})))
                self.assertIn("Self-attestation is not accepted", reason)
                self.assertIn(key, reason)

    def test_caller_arguments_are_not_mutated(self) -> None:
        arguments = _args([_item()])
        snapshot = copy.deepcopy(arguments)
        self._call(arguments)
        self.assertEqual(arguments, snapshot)

    def test_receipts_must_be_bound_to_subject_record_type_and_artifact(self) -> None:
        other_digest = hashlib.sha256(b"a different artifact").hexdigest()
        cases = {
            "different subject": [_item(subject="IND-9999")],
            "subject missing": [_item(subject=None)],
            "different record type": [_item(record_type="BENEFICIAL_OWNER")],
            "record type missing": [_item(record_type=None)],
            "different artifact": [_item(digest=other_digest)],
            "artifact missing": [_item(digest=None)],
            "one unbound item among good": [_item(), _item(subject="IND-9999", step="x")],
        }
        for name, evidence in cases.items():
            with self.subTest(name):
                self.assertIn("not bound to the submitted", self._blocked_locally(self._call(_args(evidence))))

    def test_a_digest_mentioned_only_inside_a_string_is_not_a_binding(self) -> None:
        payload = json.dumps({"tool": "verify_step", "arguments": {"note": f"{SUBJECT} {RECORD_TYPE} {DIGEST}"}})
        item = {"payload": payload, "receipt": sign_receipt(payload, verdict=1)}
        self.assertIn("not bound to the submitted", self._blocked_locally(self._call(_args([item]))))

    def test_stale_future_and_untimestamped_receipts_are_rejected(self) -> None:
        cases = {
            "stale": (_item(timestamp=_stamp(timedelta(days=-2))), "older than the permitted age"),
            "future": (_item(timestamp=_stamp(timedelta(hours=1))), "in the future"),
            "garbage timestamp": (_item(timestamp="yesterday"), "no usable timestamp"),
            "naive timestamp": (_item(timestamp="2026-11-12T09:27:11"), "no usable timestamp"),
        }
        for name, (item, fragment) in cases.items():
            with self.subTest(name):
                self.assertIn(fragment, self._blocked_locally(self._call(_args([item]))))

    def test_receipt_age_limit_is_configurable(self) -> None:
        old = _item(timestamp=_stamp(timedelta(days=-2)))
        relaxed = self._agent(max_receipt_age_seconds=7 * 86_400)
        self.assertIsNone(self._call(_args([old]), agent=relaxed).update["governance_error"])
        for bad in (0, -1, float("inf"), True):
            with self.subTest(bad), self.assertRaises(ValueError):
                self._agent(max_receipt_age_seconds=bad)

    def test_evidence_that_is_not_a_verified_allow_receipt_is_blocked_locally(self) -> None:
        tampered = _item()
        tampered["payload"] += " "
        unsigned = _item()
        unsigned["receipt"] = {k: v for k, v in unsigned["receipt"].items() if k != "signature"}
        cases = {
            "BLOCK receipt": [_item(verdict=0)],
            "payload tampered after signing": [tampered],
            "unknown signing key": [_item(kid="not-a-pinned-key")],
            "missing signature": [unsigned],
            "no receipt": [{"payload": _item()["payload"]}],
            "no payload": [{"receipt": _item()["receipt"]}],
            "not an object": ["receipt-id-1234"],
            "one bad item among good": [_item(), _item(verdict=0)],
        }
        for name, evidence in cases.items():
            with self.subTest(name):
                self.assertIn("rejected", self._blocked_locally(self._call(_args(evidence))))

    def test_a_receipt_counts_only_once(self) -> None:
        item = _item()
        self.assertIn("repeats receipt", self._blocked_locally(self._call(_args([item, dict(item)]))))

    def test_record_type_must_be_declared_by_the_platform(self) -> None:
        for record_type in ("EXEMPT_TYPE", "", None, 7):
            with self.subTest(record_type):
                self.assertIn("token-required record types", self._blocked_locally(
                    self._call(_args([_item()], record_type=record_type))))

    def test_malformed_submissions_are_blocked_locally(self) -> None:
        self._blocked_locally(self._call(_args([_item()], artifact_sha256="NOT-A-DIGEST")))
        self._blocked_locally(self._call(_args([_item()], artifact_sha256=DIGEST.upper())))
        self._blocked_locally(self._call(_args([_item()], subject_id="  ")))
        self._blocked_locally(self._call(_args([_item()], subject_id=None)))
        self._blocked_locally(self._call(_args("evidence")))
        self._blocked_locally(self._call(_args(None)))
        self._blocked_locally(self._call(_args([_item(step=str(i)) for i in range(51)])))
        self._blocked_locally(self._call({"evidence": [_item()]}))

    def test_min_evidence_is_enforced(self) -> None:
        agent = self._agent(min_evidence=2)
        self.assertIn("at least 2", self._blocked_locally(self._call(_args([_item()]), agent=agent)))
        self.assertIsNone(self._call(_args([_item(step="a"), _item(step="b")]), agent=agent).update["governance_error"])
        for bad in (0, 51, True, 1.5):
            with self.subTest(bad), self.assertRaises(ValueError):
                self._agent(min_evidence=bad)

    def test_constructor_requires_declared_record_types(self) -> None:
        for bad in ([], (), "KYC_CLEARANCE", [" "], [3]):
            with self.subTest(bad), self.assertRaises(ValueError):
                self._agent(token_required_record_types=bad)
        with self.assertRaises(TypeError):
            ProvenanceGateAgent(client=self.client, tools=self.tools)  # type: ignore[call-arg,arg-type]
        self.assertEqual(self.agent.token_required_record_types, frozenset({RECORD_TYPE, "BENEFICIAL_OWNER"}))

    def test_local_pass_does_not_bypass_the_backend_verdict(self) -> None:
        self.client.allowed = False
        command = self._call(_args([_item()]))

        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.executions, [])
        self.assertIsNotNone(command.update["governance_error"])

    def test_compiled_graph_derives_the_report_and_blocks_self_attestation(self) -> None:
        state = self.agent.invoke(
            {"messages": [], "tool_invocation": {"name": "release_report", "arguments": _args([_item()]), "tool_call_id": "g1"}}
        )
        self.assertIsNone(state["governance_error"])
        self.assertIn("verifier_report", json.loads(self.client.calls[0][0])["arguments"])

        self.client.calls.clear()
        state = self.agent.invoke(
            {"messages": [], "tool_invocation": {"name": "release_report",
             "arguments": _args([_item()], verifier_report={"status": "valid"}), "tool_call_id": "g2"}}
        )
        self.assertIn("Self-attestation", state["governance_error"])
        self.assertEqual(self.client.calls, [])

    def test_receipts_are_checked_against_the_pinned_keys_in_force(self) -> None:
        with mock.patch("ramen_foundry.core.receipts.AUDIT_PUBLIC_KEYS", {"ramen_pk_v1": "AAAA"}):
            self._blocked_locally(self._call(_args([_item()])))
        explicit = self._agent(public_keys=TEST_PUBLIC_KEYS)
        self.client.calls.clear()
        with mock.patch("ramen_foundry.core.receipts.AUDIT_PUBLIC_KEYS", {}):
            command = self._call(_args([_item()]), agent=explicit)
        # The evidence verified under the explicit keys, so the backend was reached. The gate's own
        # check of the backend's reply still uses the default keys (now empty).
        self.assertEqual(len(self.client.calls), 1)
        self.assertNotIn("Evidence item", command.update["governance_error"])

    def test_receipt_evidence_verifier_is_usable_on_its_own(self) -> None:
        verifier = ReceiptEvidenceVerifier()
        expected = {"artifact_sha256": DIGEST, "subject_id": SUBJECT, "record_type": RECORD_TYPE}
        self.assertTrue(verifier.verify(_item(), expected=expected).valid)
        self.assertFalse(verifier.verify(_item(verdict=0), expected={}).valid)
        self.assertFalse(verifier.verify(None, expected={}).valid)
        result, facts = verifier.inspect(_item(), expected=expected)
        self.assertTrue(result.valid)
        assert facts is not None
        self.assertEqual(facts.kid, "ramen_pk_v1")


class ProvenanceHasNoForgeDependencyTests(unittest.TestCase):
    """Provenance is established from Schema V5 receipts alone."""

    FORBIDDEN_TERMS = ("forge", "audit_uri", "_ramen_provenance", "memory_store", "episodic", "exemplar")

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = Path(inspect.getsourcefile(provenance_module) or "").read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_source_never_mentions_forge_or_its_lookup_surface(self) -> None:
        lowered = self.source.lower()
        for term in self.FORBIDDEN_TERMS:
            with self.subTest(term):
                self.assertNotIn(term, lowered)

    def test_imports_exclude_the_memory_and_steering_modules_and_http_clients(self) -> None:
        imported: set[str] = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = ("." * node.level) + (node.module or "")
                imported.add(module)
                imported.update(f"{module}.{alias.name}" for alias in node.names)
        for name in imported:
            with self.subTest(name):
                self.assertNotRegex(name, r"memory|steer_node|httpx|requests|urllib|socket")
        self.assertIn("ramen_foundry.core.receipts", imported)

    def test_public_surface_has_no_memory_or_remote_parameters(self) -> None:
        parameters = set(inspect.signature(ProvenanceGateAgent.__init__).parameters)
        self.assertEqual(
            parameters,
            {
                "self", "client", "tools", "token_required_record_types", "min_evidence",
                "max_receipt_age_seconds", "public_keys", "provider_key", "provider_name",
            },
        )
        for name in dir(provenance_module):
            self.assertNotIn("Forge", name)
            self.assertNotIn("Memory", name)

    def test_a_full_gate_run_makes_no_network_calls_of_its_own(self) -> None:
        def no_network(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("ProvenanceGateAgent attempted network access")

        executions: list[dict[str, Any]] = []
        with (
            mock.patch.object(socket.socket, "connect", no_network),
            mock.patch.object(httpx.Client, "send", no_network),
            mock.patch.object(httpx.AsyncClient, "send", no_network),
        ):
            agent = ProvenanceGateAgent(
                client=FakeClient(),  # type: ignore[arg-type]
                tools={"release_report": stub_tool("release_report", executions)},
                token_required_record_types=[RECORD_TYPE],
            )
            command = agent.execute("release_report", _args([_item(), _item(step="b")]))
            blocked = agent.execute("release_report", _args([_item(verdict=0)]))

        self.assertIsNone(command.update["governance_error"])
        self.assertEqual(len(executions), 1)
        self.assertIsNotNone(blocked.update["governance_error"])


if __name__ == "__main__":
    unittest.main()
