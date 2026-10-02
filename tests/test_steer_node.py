"""Routing, reflection, receipt, and memory tests for RamenSteerNode."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, TypedDict
from unittest import mock

import httpx

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command

from ramen_foundry import (
    BaseEpisodicMemoryStore,
    CorrectionExemplar,
    JSONFileMemoryStore,
    RamenSteerNode,
    RemoteForgeMemoryStore,
    SQLiteMemoryStore,
    ToolInvocation,
)
from ramen_foundry.core.memory import fingerprint_task

TASK = "Wire $150,000 to Harbor Equipment LLC"
BUNDLE_ID = "ramen__fintech_banking_invariance"
ANCHORS = ["UCC § 4A-202", "31 CFR § 1010.410(f)", "FFIEC BSA/AML Manual"]
STEERING = "Obtain and attach an Ed25519 co-signer signature before dispatching."
RECEIPT = {
    "id": "rcpt-7f3a",
    "schema_version": "5",
    "kid": "ramen-ed25519-2026-09",
    "signature": "c2lnbmF0dXJl",
    "canonical_payload": '{"allowed":true}',
    "statutory_anchors": [],
    "attestation": None,
}
FAILED_ARGS = {"account_id": "acct-1", "amount_usd": 150000.0}
REPAIRED_ARGS = {**FAILED_ARGS, "co_signer_signature": "ed25519:ab"}


def blocked_verdict(
    *,
    anchors: list[str] | None = None,
    rule_id: str = "wire-dual-control",
    reasoning: str = "High-value wire lacks dual-control authorization.",
) -> dict[str, Any]:
    return {
        "allowed": False,
        "receipt_verified": True,
        "receipt_valid": True,
        "receipt_reason": None,
        "receipt_alert": None,
        "steering": STEERING,
        "policy_ids": ["policy-1"],
        "data": {
            "allowed": False,
            "policy_ids": ["policy-1"],
            "total_violations": [
                {
                    "rule_id": rule_id,
                    "rule_name": "Wire Dual Control",
                    "rule_content": "Wires >= $10,000 require a co-signer.",
                    "enforcement_level": "strict",
                    "reasoning": reasoning,
                    "recovery_instruction": STEERING,
                }
            ],
            "results": [],
            "statutory_anchors": ANCHORS if anchors is None else anchors,
            "receipt": {**RECEIPT, "id": "rcpt-block"},
        },
    }


def allowed_verdict(*, receipt_verified: bool = True) -> dict[str, Any]:
    return {
        "allowed": True,
        "receipt_verified": receipt_verified,
        "receipt_valid": receipt_verified,
        "receipt_reason": None if receipt_verified else "signature mismatch",
        "receipt_alert": None,
        "steering": None,
        "policy_ids": ["policy-1"],
        "data": {
            "allowed": True,
            "policy_ids": ["policy-1"],
            "total_violations": [],
            "results": [],
            "statutory_anchors": [],
            "receipt": dict(RECEIPT),
        },
    }


class ScriptedClient:
    """Return queued SDK verdicts and capture every evaluate_compliance call."""

    def __init__(self, *verdicts: dict[str, Any] | Exception) -> None:
        self._verdicts = list(verdicts)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def evaluate_compliance(self, input_text: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((input_text, kwargs))
        verdict = self._verdicts.pop(0) if len(self._verdicts) > 1 else self._verdicts[0]
        if isinstance(verdict, Exception):
            raise verdict
        return verdict


def carry(update: Mapping[str, Any]) -> dict[str, Any]:
    """Return the repair-loop state a blocked Command hands to the next attempt."""
    return {key: update[key] for key in ("repair_turns", "pending_correction")}


class FailingStore(BaseEpisodicMemoryStore):
    def record_correction(self, exemplar: CorrectionExemplar) -> None:
        raise OSError("disk full")

    def retrieve_relevant_exemplars(
        self, task_fingerprint: str, tool_name: str, limit: int = 3
    ) -> list[CorrectionExemplar]:
        return []


class SteerState(TypedDict, total=False):
    task: str
    messages: Annotated[list[Any], add_messages]
    tool_invocation: Any
    governance_error: str | None
    repair_turns: int
    pending_correction: dict[str, Any] | None


class RamenSteerNodeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executions: list[dict[str, Any]] = []

        @tool
        def dispatch_wire(
            account_id: str,
            amount_usd: float,
            co_signer_signature: str | None = None,
        ) -> str:
            """Dispatch a wire."""
            self.executions.append(
                {
                    "account_id": account_id,
                    "amount_usd": amount_usd,
                    "co_signer_signature": co_signer_signature,
                }
            )
            return "WIRE-QUEUED"

        self.tools = {"dispatch_wire": dispatch_wire}

    def make_node(self, client: ScriptedClient, **kwargs: Any) -> RamenSteerNode:
        options: dict[str, Any] = {
            "client": client,
            "tools": self.tools,
            "planner_node": "planner",
            "bundle_ids": [BUNDLE_ID],
        }
        options.update(kwargs)
        return RamenSteerNode(**options)

    @staticmethod
    def state(arguments: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
        return {
            "task": TASK,
            "tool_invocation": ToolInvocation(
                name="dispatch_wire", arguments=dict(arguments), tool_call_id="call-1"
            ),
            **extra,
        }

    # -- Blocked routing -------------------------------------------------------

    def test_blocked_call_is_not_dispatched_and_routes_to_planner(self) -> None:
        client = ScriptedClient(blocked_verdict())
        command = self.make_node(client)(self.state(FAILED_ARGS))

        self.assertEqual(self.executions, [])
        self.assertEqual(command.goto, "planner")
        update = command.update
        self.assertIsNone(update["tool_invocation"])
        self.assertEqual(update["repair_turns"], 1)
        self.assertEqual(
            update["pending_correction"],
            {
                "tool_name": "dispatch_wire",
                "failed_arguments": FAILED_ARGS,
                "rule_id": "wire-dual-control",
                "violation_reason": "High-value wire lacks dual-control authorization.",
                "primary_statutory_anchor": "UCC § 4A-202",
                "steering_directive": STEERING,
            },
        )
        message = update["messages"][0]
        self.assertIsInstance(message, ToolMessage)
        self.assertEqual(message.status, "error")
        self.assertEqual(message.tool_call_id, "call-1")
        self.assertEqual(
            message.content,
            "[ACTION BLOCKED BY RAMEN ACTION GATE]\n"
            "Rule Violated: wire-dual-control (UCC § 4A-202)\n"
            "Reason: High-value wire lacks dual-control authorization.\n"
            f"Steering Directive: {STEERING}\n"
            "Instruction: Replan and re-invoke the tool with parameters satisfying "
            "the steering directive.",
        )

    def test_reflection_truncates_to_single_primary_anchor(self) -> None:
        command = self.make_node(ScriptedClient(blocked_verdict()))(self.state(FAILED_ARGS))
        message = command.update["messages"][0]

        self.assertIn("UCC § 4A-202", message.content)
        for secondary in ANCHORS[1:]:
            self.assertNotIn(secondary, message.content)
            self.assertNotIn(secondary, json.dumps(message.response_metadata))
            self.assertNotIn(secondary, json.dumps(command.update["pending_correction"]))
        self.assertEqual(message.content.count("UCC § 4A-202"), 1)
        self.assertEqual(message.response_metadata["primary_statutory_anchor"], "UCC § 4A-202")

    def test_missing_anchors_fall_back_to_general_operational_boundary(self) -> None:
        for anchors in ([], ["  "]):
            with self.subTest(anchors=anchors):
                command = self.make_node(ScriptedClient(blocked_verdict(anchors=anchors)))(
                    self.state(FAILED_ARGS)
                )
                self.assertIn(
                    "Rule Violated: wire-dual-control (General Operational Boundary)",
                    command.update["messages"][0].content,
                )

    def test_repair_budget_exhaustion_routes_to_halt_node(self) -> None:
        node = self.make_node(ScriptedClient(blocked_verdict()), max_repair_turns=2)
        command = node(self.state(FAILED_ARGS, repair_turns=2))

        self.assertEqual(command.goto, END)
        self.assertEqual(command.update["repair_turns"], 2)
        self.assertIn("Repair budget exhausted after 2 of 2", command.update["messages"][0].content)
        self.assertEqual(self.executions, [])

    # -- Allowed routing -------------------------------------------------------

    def test_allowed_call_dispatches_and_exposes_full_receipt(self) -> None:
        node = self.make_node(ScriptedClient(allowed_verdict()), next_node="summariser")
        command = node(self.state(REPAIRED_ARGS, repair_turns=1))

        self.assertEqual(command.goto, "summariser")
        self.assertEqual(len(self.executions), 1)
        update = command.update
        self.assertIsNone(update["governance_error"])
        self.assertEqual(update["repair_turns"], 0)
        self.assertIsNone(update["pending_correction"])
        message = update["messages"][0]
        self.assertEqual(message.content, "WIRE-QUEUED")
        self.assertEqual(message.status, "success")
        self.assertEqual(message.response_metadata["receipt"], RECEIPT)
        self.assertTrue(message.response_metadata["receipt_verified"])

    def test_top_level_receipt_takes_precedence(self) -> None:
        verdict = allowed_verdict()
        verdict["receipt"] = {**RECEIPT, "id": "rcpt-top-level"}
        command = self.make_node(ScriptedClient(verdict))(self.state(REPAIRED_ARGS))
        self.assertEqual(
            command.update["messages"][0].response_metadata["receipt"]["id"],
            "rcpt-top-level",
        )

    def test_next_node_defaults_to_planner(self) -> None:
        command = self.make_node(ScriptedClient(allowed_verdict()))(self.state(REPAIRED_ARGS))
        self.assertEqual(command.goto, "planner")

    # -- Fail-closed paths -----------------------------------------------------

    def test_fail_closed_paths_halt_without_dispatch(self) -> None:
        cases = {
            "evaluation error": ScriptedClient(RuntimeError("timeout")),
            "unverified receipt": ScriptedClient(allowed_verdict(receipt_verified=False)),
        }
        for label, client in cases.items():
            with self.subTest(case=label):
                command = self.make_node(client)(self.state(REPAIRED_ARGS))
                self.assertEqual(command.goto, END)
                self.assertIsNotNone(command.update["governance_error"])
                self.assertEqual(command.update["messages"][0].status, "error")
        self.assertEqual(self.executions, [])

    def test_unregistered_tool_halts(self) -> None:
        state = self.state(FAILED_ARGS)
        state["tool_invocation"] = ToolInvocation(
            name="delete_ledger", arguments={}, tool_call_id="call-x"
        )
        command = self.make_node(ScriptedClient(allowed_verdict()))(state)
        self.assertEqual(command.goto, END)
        self.assertIn("not registered", command.update["governance_error"])

    # -- Provider modes --------------------------------------------------------

    def test_byok_forwards_provider_key_and_name_together(self) -> None:
        client = ScriptedClient(allowed_verdict())
        self.make_node(client, provider_key="provider-test-value", provider_name="openai")(
            self.state(REPAIRED_ARGS)
        )
        kwargs = client.calls[0][1]
        self.assertEqual(kwargs["provider_key"], "provider-test-value")
        self.assertEqual(kwargs["provider_name"], "openai")
        self.assertEqual(kwargs["bundle_ids"], [BUNDLE_ID])

    def test_managed_mode_omits_provider_key_and_name(self) -> None:
        client = ScriptedClient(allowed_verdict())
        self.make_node(client)(self.state(REPAIRED_ARGS))
        kwargs = client.calls[0][1]
        self.assertIsNone(kwargs["provider_key"])
        self.assertIsNone(kwargs["provider_name"])

    # -- Episodic memory -------------------------------------------------------

    def test_successful_repair_records_exemplar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JSONFileMemoryStore(Path(directory) / "agent_memory.json")
            client = ScriptedClient(blocked_verdict(), allowed_verdict())
            node = self.make_node(client, memory_store=store)

            blocked = node(self.state(FAILED_ARGS))
            allowed = node(self.state(REPAIRED_ARGS, **carry(blocked.update)))

            exemplars = store.retrieve_relevant_exemplars(fingerprint_task(TASK), "dispatch_wire")
            self.assertEqual(len(exemplars), 1)
            exemplar = exemplars[0]
            self.assertEqual(exemplar.failed_arguments, FAILED_ARGS)
            self.assertEqual(exemplar.repaired_arguments, REPAIRED_ARGS)
            self.assertEqual(exemplar.primary_statutory_anchor, "UCC § 4A-202")
            self.assertEqual(exemplar.steering_directive, STEERING)
            self.assertEqual(exemplar.receipt_id, RECEIPT["id"])
            self.assertEqual(exemplar.receipt, RECEIPT)
            stored = json.loads(store.path.read_text(encoding="utf-8"))["exemplars"][0]
            self.assertEqual(stored["receipt"], RECEIPT)
            self.assertEqual(
                allowed.update["messages"][0].response_metadata["exemplar_id"],
                exemplar.exemplar_id,
            )

    def test_repair_transmits_full_receipt_to_forge(self) -> None:
        store = RemoteForgeMemoryStore(base_url="https://forge.example.test", write_token="forge-test-token")
        node = self.make_node(ScriptedClient(blocked_verdict(), allowed_verdict()), memory_store=store)
        blocked = node(self.state(FAILED_ARGS))
        created = httpx.Response(201, json={"success": True}, request=httpx.Request("POST", "https://forge.example.test"))
        with mock.patch("ramen_foundry.core.memory.httpx.post", return_value=created) as post:
            allowed = node(self.state(REPAIRED_ARGS, **carry(blocked.update)))

        post.assert_called_once()
        sent = post.call_args.kwargs["json"]
        self.assertEqual(sent["receipt"], RECEIPT)
        self.assertEqual(sent["receipt_id"], RECEIPT["id"])
        self.assertEqual(sent["task_description"], TASK)
        self.assertNotIn("memory_error", allowed.update["messages"][0].response_metadata)

    def test_first_pass_success_does_not_record_exemplar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = JSONFileMemoryStore(Path(directory) / "agent_memory.json")
            node = self.make_node(ScriptedClient(allowed_verdict()), memory_store=store)
            node(self.state(REPAIRED_ARGS))
            self.assertEqual(
                store.retrieve_relevant_exemplars(fingerprint_task(TASK), "dispatch_wire"), []
            )
            self.assertFalse(store.path.exists())

    def test_memory_failure_is_surfaced_without_losing_tool_result(self) -> None:
        node = self.make_node(
            ScriptedClient(blocked_verdict(), allowed_verdict()),
            memory_store=FailingStore(),
        )
        blocked = node(self.state(FAILED_ARGS))
        with self.assertLogs("ramen_foundry.core.steer_node", level="ERROR"):
            allowed = node(self.state(REPAIRED_ARGS, **carry(blocked.update)))
        message = allowed.update["messages"][0]
        self.assertEqual(message.content, "WIRE-QUEUED")
        self.assertIn("disk full", message.response_metadata["memory_error"])

    def test_bound_memory_store_requires_task(self) -> None:
        node = self.make_node(ScriptedClient(allowed_verdict()), memory_store=FailingStore())
        state = self.state(REPAIRED_ARGS)
        state.pop("task")
        with self.assertRaises(ValueError):
            node(state)
        self.assertEqual(self.executions, [])

    # -- Constructor -----------------------------------------------------------

    def test_constructor_validation(self) -> None:
        client = ScriptedClient(allowed_verdict())
        with self.assertRaises(ValueError):
            RamenSteerNode(client=client, tools=self.tools, planner_node="planner")
        for turns in (-1, True, 1.5):
            with self.subTest(turns=turns), self.assertRaises(ValueError):
                self.make_node(client, max_repair_turns=turns)
        with self.assertRaises(TypeError):
            self.make_node(client, memory_store=object())

    # -- Closed-loop LangGraph -------------------------------------------------

    def build_loop(self, node: RamenSteerNode, planner_calls: list[int]) -> Any:
        def planner(state: Mapping[str, Any]) -> Command:
            messages = state.get("messages") or []
            last = messages[-1] if messages else None
            if isinstance(last, ToolMessage) and last.status != "error":
                return Command(goto=END)
            planner_calls.append(len(planner_calls))
            repairing = isinstance(last, ToolMessage)
            arguments = REPAIRED_ARGS if repairing else FAILED_ARGS
            call_id = f"call-{len(planner_calls)}"
            return Command(
                update={
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[{"name": "dispatch_wire", "args": arguments, "id": call_id}],
                        )
                    ],
                    "tool_invocation": ToolInvocation(
                        name="dispatch_wire", arguments=arguments, tool_call_id=call_id
                    ),
                },
                goto="steer",
            )

        graph = StateGraph(SteerState)
        graph.add_node("planner", planner, destinations=("steer", END))
        graph.add_node("steer", node, destinations=("planner", END))
        graph.add_edge(START, "planner")
        return graph.compile()

    def test_graph_self_heals_and_records_in_both_stores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stores = {
                "json": JSONFileMemoryStore(Path(directory) / "agent_memory.json"),
                "sqlite": SQLiteMemoryStore(Path(directory) / "ramen_memory.db"),
            }
            try:
                for label, store in stores.items():
                    with self.subTest(store=label):
                        self.executions.clear()
                        planner_calls: list[int] = []
                        node = self.make_node(
                            ScriptedClient(blocked_verdict(), allowed_verdict()),
                            memory_store=store,
                        )
                        final = self.build_loop(node, planner_calls).invoke(
                            {"task": TASK, "messages": [], "repair_turns": 0}
                        )

                        self.assertEqual(len(planner_calls), 2)
                        self.assertEqual(len(self.executions), 1)
                        self.assertEqual(self.executions[0]["co_signer_signature"], "ed25519:ab")
                        self.assertIsNone(final["governance_error"])
                        tool_messages = [
                            m for m in final["messages"] if isinstance(m, ToolMessage)
                        ]
                        self.assertEqual([m.status for m in tool_messages], ["error", "success"])
                        self.assertEqual(tool_messages[-1].response_metadata["receipt"], RECEIPT)
                        exemplars = store.retrieve_relevant_exemplars(
                            fingerprint_task(TASK), "dispatch_wire"
                        )
                        self.assertEqual(len(exemplars), 1)
                        self.assertEqual(exemplars[0].receipt_id, RECEIPT["id"])
                        self.assertEqual(exemplars[0].receipt, RECEIPT)
            finally:
                stores["sqlite"].close()

    def test_graph_stops_after_max_repair_turns(self) -> None:
        planner_calls: list[int] = []
        client = ScriptedClient(blocked_verdict())
        node = self.make_node(client, max_repair_turns=2)
        final = self.build_loop(node, planner_calls).invoke(
            {"task": TASK, "messages": [], "repair_turns": 0}
        )

        self.assertEqual(len(planner_calls), 3)
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(self.executions, [])
        self.assertIsNotNone(final["governance_error"])
        self.assertIn("Repair budget exhausted", final["messages"][-1].content)


if __name__ == "__main__":
    unittest.main()
