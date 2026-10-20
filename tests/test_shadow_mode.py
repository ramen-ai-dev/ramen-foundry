"""Shadow mode: policy BLOCKs are logged, not enforced; every other failure still fails closed."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from typing import Annotated, Any, TypedDict
from unittest import mock

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool, tool
from langgraph.graph import END, START, StateGraph, add_messages

from ramen_foundry import (
    CommercialLendingAgent,
    DbShieldAgent,
    DevboxShieldAgent,
    IndustrialAutomationAgent,
    JSONFileMemoryStore,
    RamenSteerNode,
    RamenToolNode,
    ScoutShieldAgent,
    ToolInvocation,
)
from ramen_foundry.core.langgraph_nodes import (
    SHADOW_LOG_HEADER,
    SHADOW_METADATA_KEY,
    SHADOW_MODE_ENV,
    resolve_shadow_mode,
)
from ramen_foundry.core.memory import fingerprint_task
from ramen_foundry.core.receipts import (
    VERDICT_ALLOW,
    VERDICT_BLOCK,
    signed_allow_problem,
    signed_verdict_problem,
)
from ramen_foundry.core.steer_node import PROVENANCE_KEY, REFLECTION_HEADER
from tests.receipt_fixtures import TEST_PUBLIC_KEYS, sign_canonical, sign_receipt, tool_payload

SHADOW_LOGGER = "ramen_foundry.shadow"
TOOL = "dispatch_wire"
# A distinctive argument value: it must never appear in any log output.
ARGS = {"account_id": "acct-SECRET-4242", "amount_usd": 150000.0}
PAYLOAD = tool_payload(TOOL, ARGS)
STEERING = "Obtain and attach an Ed25519 co-signer signature before dispatching."
ANCHORS = ["UCC § 4A-202", "31 CFR § 1010.410(f)"]
BLOCK_RECEIPT_ID = "rcpt-shadow-block"


def block_verdict(
    receipt: dict[str, Any] | None = None,
    *,
    receipt_verified: bool = True,
    receipt_reason: str | None = None,
) -> dict[str, Any]:
    """An SDK-shaped BLOCK carrying a signed BLOCK receipt for ``PAYLOAD`` by default."""
    if receipt is None:
        receipt = sign_receipt(PAYLOAD, verdict=0, receipt_id=BLOCK_RECEIPT_ID)
    return {
        "allowed": False,
        "receipt_verified": receipt_verified,
        "receipt_valid": receipt_verified,
        "receipt_reason": receipt_reason,
        "receipt_alert": None,
        "steering": STEERING,
        "policy_ids": ["policy-1"],
        "data": {
            "allowed": False,
            "policy_ids": ["policy-1"],
            "total_violations": [
                {
                    "rule_id": "wire-dual-control",
                    "rule_name": "Wire Dual Control",
                    "reasoning": "High-value wire lacks dual-control authorization.",
                    "recovery_instruction": STEERING,
                }
            ],
            "results": [],
            "statutory_anchors": ANCHORS,
            "receipt": receipt,
        },
    }


def allow_verdict(*, receipt_verified: bool = True) -> dict[str, Any]:
    return {
        "allowed": True,
        "receipt_verified": receipt_verified,
        "receipt_reason": None if receipt_verified else "signature mismatch",
        "steering": None,
        "policy_ids": ["policy-1"],
        "data": {"allowed": True, "total_violations": [], "receipt": sign_receipt(PAYLOAD, verdict=1)},
    }


class ScriptedClient:
    """Return one fixed verdict (or raise) and capture each evaluate_compliance call."""

    def __init__(self, result: dict[str, Any] | Exception) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def evaluate_compliance(self, input_text: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((input_text, kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def invocation_state(**extra: Any) -> dict[str, Any]:
    return {
        "task": "Wire $150,000 to Harbor Equipment LLC",
        "tool_invocation": ToolInvocation(name=TOOL, arguments=dict(ARGS), tool_call_id="call-1"),
        **extra,
    }


class ShadowModeResolutionTests(unittest.TestCase):
    def test_disabled_by_default(self) -> None:
        self.assertFalse(resolve_shadow_mode(None, {}))

    def test_truthy_and_falsy_values(self) -> None:
        for raw in ("1", "true", "yes", "on", "TRUE", " On ", "Yes"):
            with self.subTest(raw=raw):
                self.assertTrue(resolve_shadow_mode(None, {SHADOW_MODE_ENV: raw}))
        for raw in ("0", "false", "no", "off", "", "  ", "FALSE", " Off "):
            with self.subTest(raw=raw):
                self.assertFalse(resolve_shadow_mode(None, {SHADOW_MODE_ENV: raw}))

    def test_ambiguous_environment_values_raise(self) -> None:
        for raw in ("2", "maybe", "enabled", "disabled", "y", "n", "t", "f", "true!", "1.0"):
            with self.subTest(raw=raw), self.assertRaises(ValueError) as caught:
                resolve_shadow_mode(None, {SHADOW_MODE_ENV: raw})
            self.assertIn(SHADOW_MODE_ENV, str(caught.exception))

    def test_explicit_argument_overrides_environment(self) -> None:
        self.assertFalse(resolve_shadow_mode(False, {SHADOW_MODE_ENV: "true"}))
        self.assertTrue(resolve_shadow_mode(True, {SHADOW_MODE_ENV: "false"}))
        # An explicit argument wins even over an invalid environment value.
        self.assertFalse(resolve_shadow_mode(False, {SHADOW_MODE_ENV: "garbage"}))

    def test_non_boolean_argument_raises(self) -> None:
        for value in (1, 0, "true", "false", "", []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_shadow_mode(value, {})  # type: ignore[arg-type]

    def test_reads_process_environment_by_default(self) -> None:
        with mock.patch.dict(os.environ, {SHADOW_MODE_ENV: "on"}):
            self.assertTrue(resolve_shadow_mode(None))


class ShadowNodeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.executions: list[dict[str, Any]] = []
        self.tool_error: Exception | None = None

        @tool
        def dispatch_wire(account_id: str, amount_usd: float) -> str:
            """Canary: records that the host tool ran."""
            if self.tool_error is not None:
                raise self.tool_error
            self.executions.append({"account_id": account_id, "amount_usd": amount_usd})
            return "WIRE-QUEUED"

        self.tools = {TOOL: dispatch_wire}

    def tool_node(self, result: dict[str, Any] | Exception, **options: Any) -> tuple[RamenToolNode, ScriptedClient]:
        client = ScriptedClient(result)
        node = RamenToolNode(
            client=client,  # type: ignore[arg-type]
            tools=self.tools,
            llm_node="assistant",
            bundle_ids=["ramen__fintech_banking_invariance"],
            **options,
        )
        return node, client

    def steer_node(self, result: dict[str, Any] | Exception, **options: Any) -> RamenSteerNode:
        options.setdefault("planner_node", "planner")
        options.setdefault("next_node", "summariser")
        return RamenSteerNode(
            client=ScriptedClient(result),  # type: ignore[arg-type]
            tools=self.tools,
            bundle_ids=["ramen__fintech_banking_invariance"],
            **options,
        )

    def assert_no_arguments_logged(self, logs: Any) -> None:
        for line in logs.output:
            self.assertNotIn("acct-SECRET-4242", line)
            self.assertNotIn("150000", line)


class ShadowConstructionTests(ShadowNodeTestCase):
    def test_startup_warning_when_enabled(self) -> None:
        with self.assertLogs(SHADOW_LOGGER, level="WARNING") as logs:
            node, _ = self.tool_node(block_verdict(), shadow_mode=True)
        self.assertTrue(node.shadow_mode)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("SHADOW MODE ENABLED", logs.output[0])
        self.assertIn("NOT enforced", logs.output[0])

    def test_no_startup_warning_when_disabled(self) -> None:
        with self.assertNoLogs(SHADOW_LOGGER, level="WARNING"):
            node, _ = self.tool_node(block_verdict())
            steer = self.steer_node(block_verdict(), shadow_mode=False)
        self.assertFalse(node.shadow_mode)
        self.assertFalse(steer.shadow_mode)

    def test_environment_enables_both_nodes(self) -> None:
        with mock.patch.dict(os.environ, {SHADOW_MODE_ENV: "true"}), self.assertLogs(SHADOW_LOGGER, "WARNING"):
            node, _ = self.tool_node(block_verdict())
            steer = self.steer_node(block_verdict())
        self.assertTrue(node.shadow_mode)
        self.assertTrue(steer.shadow_mode)

    def test_invalid_environment_value_fails_construction(self) -> None:
        with mock.patch.dict(os.environ, {SHADOW_MODE_ENV: "sometimes"}):
            with self.assertRaises(ValueError):
                self.tool_node(block_verdict())
            with self.assertRaises(ValueError):
                self.steer_node(block_verdict())

    def test_non_boolean_constructor_argument_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.tool_node(block_verdict(), shadow_mode="true")
        with self.assertRaises(ValueError):
            self.steer_node(block_verdict(), shadow_mode=1)


class RamenToolNodeShadowTests(ShadowNodeTestCase):
    def test_block_is_enforced_when_shadow_mode_is_off(self) -> None:
        node, _ = self.tool_node(block_verdict())
        with self.assertNoLogs(SHADOW_LOGGER, level="WARNING"):
            command = node(invocation_state())
        self.assertEqual(self.executions, [])
        self.assertEqual(command.update["governance_error"], STEERING)
        self.assertEqual(command.update["messages"][0].status, "error")

    def test_verified_block_is_logged_and_the_tool_executes(self) -> None:
        node, _ = self.tool_node(block_verdict(), shadow_mode=True)
        with self.assertLogs(SHADOW_LOGGER, level="WARNING") as logs:
            command = node(invocation_state())

        self.assertEqual(self.executions, [ARGS])
        self.assertEqual(command.goto, "assistant")
        self.assertIsNone(command.update["governance_error"])
        self.assertIsNone(command.update["tool_invocation"])
        self.assertEqual(set(command.update), {"messages", "governance_error", "tool_invocation"})
        message = command.update["messages"][0]
        self.assertIsInstance(message, ToolMessage)
        self.assertEqual(message.content, "WIRE-QUEUED")
        self.assertEqual(message.status, "success")
        self.assertNotIn(REFLECTION_HEADER, message.content)

        metadata = message.response_metadata
        self.assertEqual(metadata["receipt"]["id"], BLOCK_RECEIPT_ID)
        self.assertTrue(metadata["receipt_verified"])
        self.assertEqual(metadata["policy_ids"], ["policy-1"])
        shadow = metadata[SHADOW_METADATA_KEY]
        self.assertEqual(
            shadow,
            {
                "shadow_mode": True,
                "enforced": False,
                "verdict": "BLOCK",
                "tool_name": TOOL,
                "tool_call_id": "call-1",
                "rule_id": "wire-dual-control",
                "primary_statutory_anchor": "UCC § 4A-202",
                "reason": "High-value wire lacks dual-control authorization.",
                "steering_directive": STEERING,
                "receipt_id": BLOCK_RECEIPT_ID,
                "receipt_kid": "ramen_pk_v1",
                "receipt_verified": True,
                "receipt_problem": None,
                "policy_ids": ["policy-1"],
            },
        )
        json.dumps(metadata)  # JSON-serialisable for checkpointers and tracers

        self.assertEqual(len(logs.records), 1)
        record = logs.records[0]
        self.assertEqual(record.levelno, logging.WARNING)
        text = record.getMessage()
        self.assertTrue(text.startswith(SHADOW_LOG_HEADER))
        self.assertIn(f"Receipt ID: {BLOCK_RECEIPT_ID}  (signature: VERIFIED)", text)
        self.assertIn(f"Steering Directive: {STEERING}", text)
        self.assertIn("Rule Violated: wire-dual-control (UCC § 4A-202)", text)
        self.assertIn("Action: tool executed anyway (shadow_mode=True)", text)
        self.assertEqual(getattr(record, SHADOW_METADATA_KEY), shadow)
        self.assert_no_arguments_logged(logs)

    def test_client_unverified_block_logs_unverified_and_still_passes_through(self) -> None:
        verdict = block_verdict(receipt_verified=False, receipt_reason="Signature does not verify.")
        node, _ = self.tool_node(verdict, shadow_mode=True)
        with self.assertLogs(SHADOW_LOGGER, level="WARNING") as logs:
            command = node(invocation_state())

        self.assertEqual(self.executions, [ARGS])
        self.assertIsNone(command.update["governance_error"])
        shadow = command.update["messages"][0].response_metadata[SHADOW_METADATA_KEY]
        self.assertFalse(shadow["receipt_verified"])
        self.assertEqual(shadow["receipt_problem"], "Signature does not verify.")
        self.assertIn("(signature: UNVERIFIED: Signature does not verify.)", logs.records[0].getMessage())

    def test_independent_check_catches_a_block_carrying_an_allow_receipt(self) -> None:
        # The client claims a verified BLOCK, but the signed bytes say ALLOW.
        verdict = block_verdict(sign_receipt(PAYLOAD, verdict=1, receipt_id="rcpt-allow"))
        node, _ = self.tool_node(verdict, shadow_mode=True)
        with self.assertLogs(SHADOW_LOGGER, level="WARNING") as logs:
            command = node(invocation_state())

        self.assertEqual(self.executions, [ARGS])
        shadow = command.update["messages"][0].response_metadata[SHADOW_METADATA_KEY]
        self.assertFalse(shadow["receipt_verified"])
        self.assertIn("signed verdict is 1, not 0", shadow["receipt_problem"])
        self.assertIn("UNVERIFIED", logs.records[0].getMessage())

    def test_block_receipt_for_a_different_call_is_unverified(self) -> None:
        other = sign_receipt(tool_payload(TOOL, {"account_id": "x", "amount_usd": 1.0}), verdict=0)
        node, _ = self.tool_node(block_verdict(other), shadow_mode=True)
        with self.assertLogs(SHADOW_LOGGER, level="WARNING"):
            command = node(invocation_state())
        shadow = command.update["messages"][0].response_metadata[SHADOW_METADATA_KEY]
        self.assertFalse(shadow["receipt_verified"])
        self.assertIn("payload_hash does not match", shadow["receipt_problem"])

    def test_missing_receipt_is_unverified_and_still_passes_through(self) -> None:
        verdict = block_verdict()
        verdict["data"]["receipt"] = None
        node, _ = self.tool_node(verdict, shadow_mode=True)
        with self.assertLogs(SHADOW_LOGGER, level="WARNING") as logs:
            command = node(invocation_state())

        self.assertEqual(self.executions, [ARGS])
        metadata = command.update["messages"][0].response_metadata
        self.assertIsNone(metadata["receipt"])
        self.assertIsNone(metadata[SHADOW_METADATA_KEY]["receipt_id"])
        self.assertFalse(metadata[SHADOW_METADATA_KEY]["receipt_verified"])
        self.assertIn("Receipt ID: <none>", logs.records[0].getMessage())

    def test_full_receipt_is_logged_only_at_debug(self) -> None:
        node, _ = self.tool_node(block_verdict(), shadow_mode=True)
        with self.assertLogs(SHADOW_LOGGER, level="DEBUG") as logs:
            node(invocation_state())
        debug = [r for r in logs.records if r.levelno == logging.DEBUG]
        self.assertEqual(len(debug), 1)
        self.assertIn("canonical_payload", debug[0].getMessage())
        self.assert_no_arguments_logged(logs)

    def test_evaluation_failure_still_fails_closed(self) -> None:
        node, _ = self.tool_node(RuntimeError("connection refused"), shadow_mode=True)
        with self.assertNoLogs(SHADOW_LOGGER, level="WARNING"):
            command = node(invocation_state())
        self.assertEqual(self.executions, [])
        self.assertIn("Governance evaluation unavailable", command.update["governance_error"])

    def test_unverified_allow_still_fails_closed(self) -> None:
        node, _ = self.tool_node(allow_verdict(receipt_verified=False), shadow_mode=True)
        command = node(invocation_state())
        self.assertEqual(self.executions, [])
        self.assertIn("could not be verified", command.update["governance_error"])

    def test_unregistered_tool_still_fails_closed(self) -> None:
        self.tools.clear()
        node, _ = self.tool_node(block_verdict(), shadow_mode=True)
        with self.assertLogs(SHADOW_LOGGER, level="WARNING"):
            command = node(invocation_state())
        self.assertIn("is not registered", command.update["governance_error"])
        self.assertEqual(command.update["messages"][0].status, "error")

    def test_tool_exception_still_fails_closed(self) -> None:
        self.tool_error = RuntimeError("core banking offline")
        node, _ = self.tool_node(block_verdict(), shadow_mode=True)
        with self.assertLogs(SHADOW_LOGGER, level="WARNING"):
            command = node(invocation_state())
        self.assertIn("core banking offline", command.update["governance_error"])
        self.assertEqual(command.update["messages"][0].status, "error")

    def test_verified_allow_is_unaffected_by_shadow_mode(self) -> None:
        node, _ = self.tool_node(allow_verdict(), shadow_mode=True)
        with self.assertNoLogs(SHADOW_LOGGER, level="WARNING"):
            command = node(invocation_state())
        self.assertEqual(self.executions, [ARGS])
        self.assertNotIn(SHADOW_METADATA_KEY, command.update["messages"][0].response_metadata)

    def test_byok_provider_key_and_name_are_forwarded_in_shadow_mode(self) -> None:
        node, client = self.tool_node(
            block_verdict(), shadow_mode=True, provider_key="provider-test-value", provider_name="anthropic"
        )
        with self.assertLogs(SHADOW_LOGGER, level="WARNING") as logs:
            node(invocation_state())
        _, kwargs = client.calls[0]
        self.assertEqual(kwargs["provider_key"], "provider-test-value")
        self.assertEqual(kwargs["provider_name"], "anthropic")
        for line in logs.output:
            self.assertNotIn("provider-test-value", line)

    def test_managed_mode_omits_provider_key_and_name_in_shadow_mode(self) -> None:
        node, client = self.tool_node(block_verdict(), shadow_mode=True)
        with self.assertLogs(SHADOW_LOGGER, level="WARNING"):
            node(invocation_state())
        _, kwargs = client.calls[0]
        self.assertIsNone(kwargs["provider_key"])
        self.assertIsNone(kwargs["provider_name"])


class RamenSteerNodeShadowTests(ShadowNodeTestCase):
    def test_block_dispatches_and_skips_the_steering_loop(self) -> None:
        node = self.steer_node(block_verdict(), shadow_mode=True)
        with self.assertLogs(SHADOW_LOGGER, level="WARNING") as logs:
            command = node(invocation_state(repair_turns=1))

        self.assertEqual(self.executions, [ARGS])
        self.assertEqual(command.goto, "summariser")
        update = command.update
        self.assertEqual(update["repair_turns"], 0)
        self.assertIsNone(update["pending_correction"])
        self.assertIsNone(update["governance_error"])
        self.assertIsNone(update["tool_invocation"])
        message = update["messages"][0]
        self.assertEqual(message.content, "WIRE-QUEUED")
        self.assertEqual(message.status, "success")
        self.assertNotIn(REFLECTION_HEADER, message.content)
        self.assertNotIn("Replan", message.content)

        metadata = message.response_metadata
        self.assertEqual(metadata["receipt"]["id"], BLOCK_RECEIPT_ID)
        self.assertTrue(metadata[SHADOW_METADATA_KEY]["receipt_verified"])
        provenance = metadata[PROVENANCE_KEY]
        self.assertEqual(provenance["receipt_id"], BLOCK_RECEIPT_ID)
        self.assertIsNone(provenance["exemplar_id"])
        self.assertIsNone(provenance["prevention_summary"])
        self.assertEqual(provenance["source"], "ramen-foundry-local")
        self.assertEqual(provenance["tool_name"], TOOL)
        self.assertNotIn("exemplar_id", metadata)
        json.dumps(metadata)

        text = logs.records[0].getMessage()
        self.assertIn(f"Receipt ID: {BLOCK_RECEIPT_ID}", text)
        self.assertIn(f"Steering Directive: {STEERING}", text)
        self.assert_no_arguments_logged(logs)

    def test_block_is_steered_when_shadow_mode_is_off(self) -> None:
        command = self.steer_node(block_verdict())(invocation_state())
        self.assertEqual(self.executions, [])
        self.assertEqual(command.goto, "planner")
        self.assertIn(REFLECTION_HEADER, command.update["messages"][0].content)

    def test_no_correction_exemplar_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = JSONFileMemoryStore(Path(tmp) / "memory.json")
            node = self.steer_node(block_verdict(), shadow_mode=True, memory_store=store)
            # A pending correction from an earlier enforced block must not be
            # turned into a "repair" by a shadow pass-through.
            pending = {
                "tool_name": TOOL,
                "failed_arguments": {"account_id": "acct-1", "amount_usd": 1.0},
                "rule_id": "wire-dual-control",
                "violation_reason": "earlier",
                "primary_statutory_anchor": "UCC § 4A-202",
                "steering_directive": STEERING,
            }
            with mock.patch.object(store, "record_correction") as record, self.assertLogs(SHADOW_LOGGER, "WARNING"):
                command = node(invocation_state(repair_turns=1, pending_correction=pending))
            record.assert_not_called()
            self.assertIsNone(command.update["pending_correction"])
            task_fingerprint = fingerprint_task(invocation_state()["task"])
            self.assertEqual(store.retrieve_relevant_exemplars(task_fingerprint, TOOL), [])
            self.assertIsNone(command.update["messages"][0].response_metadata[PROVENANCE_KEY]["exemplar_id"])

    def test_exhausted_repair_budget_does_not_halt_in_shadow_mode(self) -> None:
        node = self.steer_node(block_verdict(), shadow_mode=True, max_repair_turns=2)
        with self.assertLogs(SHADOW_LOGGER, level="WARNING"):
            command = node(invocation_state(repair_turns=2))
        self.assertEqual(command.goto, "summariser")
        self.assertEqual(command.update["repair_turns"], 0)
        self.assertEqual(self.executions, [ARGS])

    def test_unverified_block_still_passes_through(self) -> None:
        node = self.steer_node(block_verdict(receipt_verified=False, receipt_reason="bad sig"), shadow_mode=True)
        with self.assertLogs(SHADOW_LOGGER, level="WARNING") as logs:
            command = node(invocation_state())
        self.assertEqual(command.goto, "summariser")
        self.assertEqual(self.executions, [ARGS])
        self.assertIn("UNVERIFIED: bad sig", logs.records[0].getMessage())

    def test_tool_exception_halts_in_shadow_mode(self) -> None:
        self.tool_error = RuntimeError("core banking offline")
        node = self.steer_node(block_verdict(), shadow_mode=True)
        with self.assertLogs(SHADOW_LOGGER, level="WARNING"):
            command = node(invocation_state(repair_turns=1))
        self.assertEqual(command.goto, END)
        self.assertIn("core banking offline", command.update["governance_error"])
        self.assertEqual(command.update["repair_turns"], 1)

    def test_evaluation_failure_halts_in_shadow_mode(self) -> None:
        node = self.steer_node(RuntimeError("timeout"), shadow_mode=True)
        command = node(invocation_state())
        self.assertEqual(command.goto, END)
        self.assertEqual(self.executions, [])

    def test_shadow_update_is_accepted_by_a_compiled_graph(self) -> None:
        class State(TypedDict, total=False):
            task: str
            messages: Annotated[list[Any], add_messages]
            tool_invocation: Any
            governance_error: str | None
            repair_turns: int
            pending_correction: dict[str, Any] | None

        node = self.steer_node(block_verdict(), shadow_mode=True, planner_node="planner", next_node=END)
        graph = StateGraph(State)
        graph.add_node("planner", lambda state: {})
        graph.add_node("gate", node, destinations=("planner", END))
        graph.add_edge(START, "gate")
        with self.assertLogs(SHADOW_LOGGER, level="WARNING"):
            final = graph.compile().invoke(invocation_state())
        self.assertEqual(final["messages"][-1].content, "WIRE-QUEUED")
        self.assertEqual(final["repair_turns"], 0)
        self.assertIsNone(final["governance_error"])


class TemplateShadowIsolationTests(unittest.TestCase):
    """High-assurance templates never expose shadow mode and ignore RAMEN_SHADOW_MODE."""

    @staticmethod
    def make_tool(name: str) -> StructuredTool:
        def run(value: str = "") -> str:
            return "ran"

        return StructuredTool.from_function(func=run, name=name, description=f"Test tool {name}.")

    def test_templates_enforce_even_when_environment_enables_shadow_mode(self) -> None:
        templates = [
            (CommercialLendingAgent, "dispatch_wire"),
            (IndustrialAutomationAgent, "adjust_plc_setpoint"),
            (DevboxShieldAgent, "inspect_directory"),
            (DbShieldAgent, "run_query"),
            (ScoutShieldAgent, "fetch_url"),
        ]
        client = ScriptedClient(block_verdict())
        with mock.patch.dict(os.environ, {SHADOW_MODE_ENV: "true"}):
            for template, tool_name in templates:
                with self.subTest(template=template.__name__):
                    with self.assertNoLogs(SHADOW_LOGGER, level="WARNING"):
                        agent = template(client=client, tools={tool_name: self.make_tool(tool_name)})  # type: ignore[arg-type]
                    self.assertFalse(agent._guard.shadow_mode)

    def test_templates_do_not_accept_a_shadow_mode_argument(self) -> None:
        for template, tool_name in (
            (CommercialLendingAgent, "dispatch_wire"),
            (IndustrialAutomationAgent, "adjust_plc_setpoint"),
            (DevboxShieldAgent, "inspect_directory"),
            (DbShieldAgent, "run_query"),
            (ScoutShieldAgent, "fetch_url"),
        ):
            with self.subTest(template=template.__name__), self.assertRaises(TypeError):
                template(  # type: ignore[call-arg]
                    client=ScriptedClient(block_verdict()),
                    tools={tool_name: self.make_tool(tool_name)},
                    shadow_mode=True,
                )


class SignedVerdictProblemTests(unittest.TestCase):
    def test_signed_block_passes_for_expected_block(self) -> None:
        verdict = block_verdict()
        self.assertIsNone(signed_verdict_problem(verdict, PAYLOAD, VERDICT_BLOCK, TEST_PUBLIC_KEYS))

    def test_verdict_mismatch_is_reported_both_ways(self) -> None:
        block = block_verdict()
        allow = block_verdict(sign_receipt(PAYLOAD, verdict=1))
        self.assertIn("not 1", signed_verdict_problem(block, PAYLOAD, VERDICT_ALLOW, TEST_PUBLIC_KEYS) or "")
        self.assertIn("not 0 (BLOCK)", signed_verdict_problem(allow, PAYLOAD, VERDICT_BLOCK, TEST_PUBLIC_KEYS) or "")

    def test_signed_allow_problem_is_the_allow_specialisation(self) -> None:
        allow = block_verdict(sign_receipt(PAYLOAD, verdict=1))
        self.assertIsNone(signed_allow_problem(allow, PAYLOAD, TEST_PUBLIC_KEYS))
        self.assertIn(
            "the evaluated call was blocked",
            signed_allow_problem(block_verdict(), PAYLOAD, TEST_PUBLIC_KEYS) or "",
        )

    def test_boolean_signed_verdicts_are_rejected(self) -> None:
        good = json.loads(sign_receipt(PAYLOAD, verdict=0, receipt_id="r1")["canonical_payload"])
        as_false = {"data": {"receipt": sign_canonical(good | {"verdict": False})}}
        as_true = {"data": {"receipt": sign_canonical(good | {"verdict": True})}}
        self.assertIsNotNone(signed_verdict_problem(as_false, PAYLOAD, VERDICT_BLOCK, TEST_PUBLIC_KEYS))
        self.assertIsNotNone(signed_verdict_problem(as_true, PAYLOAD, VERDICT_ALLOW, TEST_PUBLIC_KEYS))

    def test_unknown_expected_verdict_raises(self) -> None:
        for expected in (2, -1, True, False, "0"):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                signed_verdict_problem(block_verdict(), PAYLOAD, expected, TEST_PUBLIC_KEYS)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
