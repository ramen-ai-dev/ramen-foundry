"""Tests for ProvenanceGateAgent: receipt-only provenance with no forge dependency."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import socket
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import httpx
from langchain_core.tools import tool

import ramen_foundry.templates.provenance as provenance_module
from ramen_foundry import (
    PROVENANCE_GATE_BUNDLE_ID,
    ProvenanceGateAgent,
    ReceiptEvidenceVerifier,
)
from tests.gate_fixtures import FakeClient
from tests.receipt_fixtures import TEST_PUBLIC_KEYS, sign_receipt

ARTIFACT = b"quarterly-report-v3"
DIGEST = hashlib.sha256(ARTIFACT).hexdigest()


def _item(*, verdict: int = 1, digest: str | None = DIGEST, kid: str = "ramen_pk_v1", step: str = "scan") -> dict[str, Any]:
    arguments: dict[str, Any] = {"step": step}
    if digest is not None:
        arguments["artifact_sha256"] = digest
    payload = json.dumps({"tool": "verify_step", "arguments": arguments}, sort_keys=True, separators=(",", ":"))
    return {"payload": payload, "receipt": sign_receipt(payload, verdict=verdict, kid=kid)}


class ProvenanceGateAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executions: list[str] = []

        @tool
        def release_report(artifact_sha256: str, evidence: list[dict[str, Any]]) -> str:
            """Release a report."""

            self.executions.append("release_report")
            return "released"

        @tool
        def submit_filing(artifact_sha256: str, evidence: list[dict[str, Any]]) -> str:
            """Submit a filing."""

            self.executions.append("submit_filing")
            return "submitted"

        self.client = FakeClient()
        self.tools = {"release_report": release_report, "submit_filing": submit_filing}
        self.agent = ProvenanceGateAgent(client=self.client, tools=self.tools)  # type: ignore[arg-type]

    def _call(self, evidence: Any, *, digest: Any = DIGEST, tool_name: str = "release_report", agent: Any = None) -> Any:
        return (agent or self.agent).execute(tool_name, {"artifact_sha256": digest, "evidence": evidence})

    def _blocked_locally(self, command: Any) -> str:
        self.assertEqual(self.client.calls, [], "the backend must not be contacted")
        self.assertEqual(self.executions, [])
        reason = command.update["governance_error"]
        self.assertIsNotNone(reason)
        return str(reason)

    def test_valid_receipt_evidence_reaches_the_bundle_and_runs(self) -> None:
        command = self._call([_item(step="scan"), _item(step="review")])

        self.assertIsNone(command.update["governance_error"])
        self.assertEqual(self.executions, ["release_report"])
        _, options = self.client.calls[0]
        self.assertEqual(options["bundle_ids"], [PROVENANCE_GATE_BUNDLE_ID])

    def test_both_tools_are_gated(self) -> None:
        for tool_name in ("release_report", "submit_filing"):
            with self.subTest(tool_name):
                self.assertIn("at least 1 verified receipt", self._blocked_locally(self._call([], tool_name=tool_name)))

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
                reason = self._blocked_locally(self._call(evidence))
                self.assertIn("rejected", reason)

    def test_a_receipt_counts_only_once(self) -> None:
        item = _item()
        reason = self._blocked_locally(self._call([item, dict(item)]))
        self.assertIn("repeats receipt", reason)

    def test_receipts_must_reference_the_artifact_digest(self) -> None:
        reason = self._blocked_locally(self._call([_item(digest=None)]))
        self.assertIn("does not reference the artifact digest", reason)

        other = hashlib.sha256(b"a different artifact").hexdigest()
        self._blocked_locally(self._call([_item()], digest=other))

    def test_digest_binding_can_be_disabled_explicitly(self) -> None:
        agent = ProvenanceGateAgent(client=self.client, tools=self.tools, bind_artifact_digest=False)  # type: ignore[arg-type]
        command = self._call([_item(digest=None)], agent=agent)
        self.assertIsNone(command.update["governance_error"])

    def test_malformed_arguments_are_blocked_locally(self) -> None:
        self._blocked_locally(self._call([_item()], digest="NOT-A-DIGEST"))
        self._blocked_locally(self._call([_item()], digest=DIGEST.upper()))
        self._blocked_locally(self._call([_item()], digest=None))
        self._blocked_locally(self._call("evidence"))
        self._blocked_locally(self._call(None))
        self._blocked_locally(self._call([_item(step=str(i)) for i in range(51)]))

    def test_min_evidence_is_enforced(self) -> None:
        agent = ProvenanceGateAgent(client=self.client, tools=self.tools, min_evidence=2)  # type: ignore[arg-type]
        self.assertIn("at least 2", self._blocked_locally(self._call([_item()], agent=agent)))
        self.assertIsNone(self._call([_item(step="a"), _item(step="b")], agent=agent).update["governance_error"])
        for bad in (0, 51, True, 1.5):
            with self.subTest(bad), self.assertRaises(ValueError):
                ProvenanceGateAgent(client=self.client, tools=self.tools, min_evidence=bad)  # type: ignore[arg-type]

    def test_local_pass_does_not_bypass_the_backend_verdict(self) -> None:
        self.client.allowed = False
        command = self._call([_item()])

        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.executions, [])
        self.assertIsNotNone(command.update["governance_error"])

    def test_receipts_are_checked_against_the_pinned_keys_in_force(self) -> None:
        with mock.patch("ramen_foundry.core.receipts.AUDIT_PUBLIC_KEYS", {"ramen_pk_v1": "AAAA"}):
            self._blocked_locally(self._call([_item()]))
        explicit = ProvenanceGateAgent(
            client=self.client, tools=self.tools, public_keys=TEST_PUBLIC_KEYS  # type: ignore[arg-type]
        )
        self.client.calls.clear()
        with mock.patch("ramen_foundry.core.receipts.AUDIT_PUBLIC_KEYS", {}):
            command = self._call([_item()], agent=explicit)
        # The evidence receipt verified under the explicit keys, so the backend was reached.
        # The gate's own check of the backend's reply still uses the default keys (now empty).
        self.assertEqual(len(self.client.calls), 1)
        self.assertNotIn("Evidence item", command.update["governance_error"])

    def test_receipt_evidence_verifier_is_usable_on_its_own(self) -> None:
        verifier = ReceiptEvidenceVerifier()
        self.assertTrue(verifier.verify(_item(), expected={"artifact_sha256": DIGEST}).valid)
        self.assertFalse(verifier.verify(_item(verdict=0), expected={}).valid)
        self.assertFalse(verifier.verify(None, expected={}).valid)


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
                "self", "client", "tools", "min_evidence", "bind_artifact_digest",
                "public_keys", "provider_key", "provider_name",
            },
        )
        for name in dir(provenance_module):
            self.assertNotIn("Forge", name)
            self.assertNotIn("Memory", name)

    def test_a_full_gate_run_makes_no_network_calls_of_its_own(self) -> None:
        def no_network(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("ProvenanceGateAgent attempted network access")

        executions: list[str] = []

        @tool
        def release_report(artifact_sha256: str, evidence: list[dict[str, Any]]) -> str:
            """Release a report."""

            executions.append("release_report")
            return "released"

        with (
            mock.patch.object(socket.socket, "connect", no_network),
            mock.patch.object(httpx.Client, "send", no_network),
            mock.patch.object(httpx.AsyncClient, "send", no_network),
        ):
            agent = ProvenanceGateAgent(client=FakeClient(), tools={"release_report": release_report})  # type: ignore[arg-type]
            command = agent.execute(
                "release_report", {"artifact_sha256": DIGEST, "evidence": [_item(), _item(step="b")]}
            )
            blocked = agent.execute("release_report", {"artifact_sha256": DIGEST, "evidence": [_item(verdict=0)]})

        self.assertIsNone(command.update["governance_error"])
        self.assertEqual(executions, ["release_report"])
        self.assertIsNotNone(blocked.update["governance_error"])


if __name__ == "__main__":
    unittest.main()
