"""Signed-ALLOW verification in RamenToolNode: forgeries never reach the host tool."""

from __future__ import annotations

import json
import unittest
from typing import Any

import httpx
from langchain_core.tools import tool
from ramen_ai import RamenClient
from ramen_ai.verifier import AUDIT_PUBLIC_KEYS as PRODUCTION_KEYS

from ramen_foundry import RamenToolNode, ToolInvocation
from ramen_foundry.core.receipts import signed_allow_problem
from tests.receipt_fixtures import TEST_PUBLIC_KEYS, sign_canonical, sign_receipt, tool_payload

ARGS = {"command": "rm -rf ./build/tmp", "working_directory": "/workspace/project"}
PAYLOAD = tool_payload("run_bash", ARGS)


def verdict(receipt: dict[str, Any] | None, *, allowed: bool = True, receipt_verified: bool = True) -> dict[str, Any]:
    """An SDK-shaped result whose unsigned fields claim whatever the caller says."""
    return {
        "allowed": allowed,
        "receipt_verified": receipt_verified,
        "receipt_reason": None if receipt_verified else "Signature does not verify against canonical_payload.",
        "steering": None,
        "policy_ids": ["policy-1"],
        "data": {"allowed": allowed, "receipt": receipt, "total_violations": []},
    }


class StaticClient:
    """Returns one fixed result: stands in for a wrapper or tampered response."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[str] = []

    def evaluate_compliance(self, input_text: str, **_: Any) -> dict[str, Any]:
        self.calls.append(input_text)
        return self.result


class RamenToolNodeSignedAllowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canary: list[str] = []

        @tool
        def run_bash(command: str, working_directory: str) -> str:
            """Canary: records that the host tool ran."""
            self.canary.append(command)
            return "ran"

        self.tools = {"run_bash": run_bash}

    def run_node(self, result: dict[str, Any], **node_options: Any) -> Any:
        node = RamenToolNode(
            client=StaticClient(result),  # type: ignore[arg-type]
            tools=self.tools,
            llm_node="assistant",
            bundle_ids=["ramen__shield_core_it"],
            **node_options,
        )
        return node({"tool_invocation": ToolInvocation(name="run_bash", arguments=ARGS, tool_call_id="c1")})

    def assert_halted(self, command: Any, reason: str) -> None:
        self.assertEqual(self.canary, [], "the host tool must not run")
        error = command.update["governance_error"]
        self.assertIn("could not be verified", error)
        self.assertIn(reason, error)
        message = command.update["messages"][0]
        self.assertEqual(message.status, "error")
        self.assertIn("Governance blocked", message.content)

    def test_genuine_signed_allow_executes_the_tool(self) -> None:
        command = self.run_node(verdict(sign_receipt(PAYLOAD, verdict=1)))
        self.assertIsNone(command.update["governance_error"])
        self.assertEqual(self.canary, [ARGS["command"]])
        self.assertEqual(command.update["messages"][0].content, "ran")

    def test_replayed_signed_block_receipt_halts(self) -> None:
        # allowed and receipt_verified both claim success; only the signed verdict is honest.
        block = sign_receipt(PAYLOAD, verdict=0)
        self.assert_halted(self.run_node(verdict(block)), "signed verdict is 0")

    def test_invalidated_signature_halts(self) -> None:
        receipt = sign_receipt(PAYLOAD, verdict=0)
        receipt["canonical_payload"] = receipt["canonical_payload"].replace('"verdict":0', '"verdict":1')
        self.assert_halted(self.run_node(verdict(receipt)), "signature does not verify")

    def test_missing_receipt_halts(self) -> None:
        self.assert_halted(self.run_node(verdict(None)), "no Schema V5 receipt")

    def test_allow_receipt_for_a_different_call_halts(self) -> None:
        other = sign_receipt(tool_payload("run_bash", {"command": "ls", "working_directory": "/"}), verdict=1)
        self.assert_halted(self.run_node(verdict(other)), "payload_hash does not match")

    def test_receipt_from_an_unpinned_key_halts(self) -> None:
        # The production key map cannot verify a receipt signed with the test key.
        command = self.run_node(verdict(sign_receipt(PAYLOAD, verdict=1)), public_keys=PRODUCTION_KEYS)
        self.assert_halted(command, "signature does not verify")
        unknown = sign_receipt(PAYLOAD, verdict=1, kid="rogue_kid")
        self.assert_halted(self.run_node(verdict(unknown)), "unknown key")

    def test_client_reported_unverified_receipt_still_halts(self) -> None:
        command = self.run_node(verdict(sign_receipt(PAYLOAD, verdict=1), receipt_verified=False))
        self.assert_halted(command, "Signature does not verify")

    def test_blocked_verdict_is_unaffected(self) -> None:
        command = self.run_node(verdict(sign_receipt(PAYLOAD, verdict=0), allowed=False))
        self.assertEqual(self.canary, [])
        self.assertEqual(command.update["governance_error"], "Tool execution was blocked by ramen-ai policy.")

    def test_real_sdk_client_also_rejects_a_flipped_response(self) -> None:
        # End to end through RamenClient: a response with allowed flipped to true
        # around a signed BLOCK receipt never releases the tool.
        block = sign_receipt(PAYLOAD, verdict=0)
        forged = {"data": {"allowed": True, "policy_ids": [], "executed_at": "", "total_violations": [],
                           "results": [], "statutory_anchors": [], "receipt": block}}
        client = RamenClient("ramen_ak_test")
        client._http = httpx.Client(
            base_url="https://api.example.test",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=forged)),
        )
        node = RamenToolNode(client=client, tools=self.tools, llm_node="assistant", bundle_ids=["b"])
        command = node({"tool_invocation": ToolInvocation(name="run_bash", arguments=ARGS, tool_call_id="c1")})
        self.assertEqual(self.canary, [])
        self.assertIn("could not be verified", command.update["governance_error"])


class SignedAllowProblemTests(unittest.TestCase):
    def test_valid_allow_passes(self) -> None:
        self.assertIsNone(signed_allow_problem(verdict(sign_receipt(PAYLOAD, verdict=1)), PAYLOAD, TEST_PUBLIC_KEYS))

    def test_signed_fields_are_checked(self) -> None:
        # Each payload is validly signed, so only the semantic check can reject it.
        good = json.loads(sign_receipt(PAYLOAD, verdict=1, receipt_id="rcpt-1")["canonical_payload"])
        cases = [
            ("schema_version", sign_canonical(good | {"schema_version": "4.0"})),
            ("signed kid", sign_canonical(good | {"kid": "test-kid"}, receipt_kid="ramen_pk_v1")),
            ("signed id", sign_canonical(good, receipt_id="rcpt-2")),
            ("verdict", sign_canonical(good | {"verdict": "ALLOWED"})),
        ]
        for expected, receipt in cases:
            with self.subTest(field=expected):
                problem = signed_allow_problem(verdict(receipt), PAYLOAD, TEST_PUBLIC_KEYS)
                self.assertIsNotNone(problem)
                self.assertIn(expected, problem)

    def test_never_raises_on_garbage(self) -> None:
        for receipt in ({"kid": "ramen_pk_v1", "signature": "!!!", "canonical_payload": "{}"},
                        {"kid": "ramen_pk_v1", "signature": "", "canonical_payload": "{}"},
                        {"kid": 7, "signature": "a", "canonical_payload": "b"}):
            self.assertIsInstance(signed_allow_problem(verdict(receipt), PAYLOAD, TEST_PUBLIC_KEYS), str)


if __name__ == "__main__":
    unittest.main()
