"""Closed-loop self-healing wire agent built on RamenSteerNode.

The agent is tasked with a $150,000 commercial wire that lacks UCC Article 4A
dual-control evidence. ``RamenSteerNode`` gates every ``dispatch_wire`` call
against the FinTech Banking Invariance bundle before dispatch:

* Turn 1: the unsigned wire is blocked pre-dispatch. The planner receives a
  structured reflection carrying the rule, primary statutory anchor, and the
  ramen-ai steering directive.
* Turn 2: the planner reads the directive, obtains the treasury co-signer's
  Ed25519 signature over the wire's authorization fields, and re-invokes the
  tool. The repair is allowed, the wire is dispatched, the verified Schema V5
  receipt is surfaced in the tool metadata, and the correction exemplar is
  written to ``agent_memory.json``.

The planner is deterministic so the loop is reproducible; in production it is
an LLM node that reads the same reflection message. The co-signer key is
generated per run to stand in for a treasury officer's signing device.

Run from the repository root (makes live ramen-ai calls)::

    RAMEN_API_KEY=... python examples/self_healing_agent_demo.py

``RAMEN_API_KEY`` is required. When ``OPENAI_API_KEY`` is set, evaluations use
OpenAI BYOK (Starter/Professional). When it is unset, provider key and name are
both omitted so ramen-ai uses Enterprise managed-provider inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, TypedDict

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command
from ramen_ai import RamenClient

from ramen_foundry import JSONFileMemoryStore, RamenSteerNode, ToolInvocation
from ramen_foundry.core.memory import fingerprint_task
from ramen_foundry.core.steer_node import REFLECTION_HEADER
from ramen_foundry.templates.fintech import FINTECH_BANKING_INVARIANCE_BUNDLE_ID

PLANNER_NODE = "planner"
STEER_NODE = "ramen_steer"
TASK = (
    "Disburse the approved commercial loan: wire $150,000.00 from account "
    "acct-commercial-1042 to Harbor Equipment LLC (routing 021000021, account "
    "9876543210) against GL offset 1010-commercial-loans."
)
WIRE_SIGNED_FIELDS = (
    "account_id",
    "amount_usd",
    "beneficiary_name",
    "beneficiary_routing",
    "beneficiary_account",
    "sanction_clearance_token",
    "gl_offset",
)


class SelfHealingState(TypedDict, total=False):
    """Graph state shared by the planner and RamenSteerNode."""

    task: str
    messages: Annotated[list[Any], add_messages]
    tool_invocation: ToolInvocation | Mapping[str, Any] | None
    governance_error: str | None
    repair_turns: int
    pending_correction: dict[str, Any] | None


def sanctions_clearance_token(payload: Mapping[str, Any]) -> str:
    """Return the deterministic OFAC screening token bound to the beneficiary."""
    subject = "|".join(
        (
            payload["beneficiary_name"],
            payload["beneficiary_routing"],
            payload["beneficiary_account"],
        )
    )
    digest = hashlib.sha256(subject.encode()).hexdigest().upper()
    return f"OFAC-DET-20260910-{digest[:20]}"


def unanchored_wire() -> dict[str, Any]:
    """Return the agent's first-pass wire: correct economics, no co-signer."""
    payload: dict[str, Any] = {
        "account_id": "acct-commercial-1042",
        "amount_usd": 150000.0,
        "beneficiary_name": "Harbor Equipment LLC",
        "beneficiary_routing": "021000021",
        "beneficiary_account": "9876543210",
        "gl_offset": "1010-commercial-loans",
    }
    payload["sanction_clearance_token"] = sanctions_clearance_token(payload)
    return payload


def co_sign(payload: Mapping[str, Any], co_signer: Ed25519PrivateKey) -> dict[str, Any]:
    """Attach an Ed25519 dual-control signature over the wire's authorization fields."""
    signed = dict(payload)
    authorization = json.dumps(
        {field: signed[field] for field in WIRE_SIGNED_FIELDS},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    public_key = co_signer.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    signed["co_signer_public_key"] = f"ed25519:{public_key.hex()}"
    signed["co_signer_signature"] = f"ed25519:{co_signer.sign(authorization).hex()}"
    return signed


def _reflection_field(reflection: str, label: str) -> str:
    prefix = f"{label}:"
    for line in reflection.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return ""


class ScriptedWirePlanner:
    """Deterministic planner that proposes a wire and repairs it from a reflection."""

    def __init__(self, co_signer: Ed25519PrivateKey) -> None:
        self._co_signer = co_signer
        self.turn = 0

    def __call__(self, state: Mapping[str, Any]) -> Command:
        messages = state.get("messages") or []
        last = messages[-1] if messages else None

        if isinstance(last, ToolMessage) and last.status != "error":
            return Command(
                update={"messages": [AIMessage(content=f"Wire complete: {last.content}")]},
                goto=END,
            )

        if isinstance(last, ToolMessage) and REFLECTION_HEADER in str(last.content):
            print("\n--- Planner: reading reflection ---")
            print(last.content)
            directive = _reflection_field(str(last.content), "Steering Directive")
            print(f"\nPlanner: applying steering directive -> {directive}")
            print("Planner: requesting treasury co-signer Ed25519 signature.")
            arguments = co_sign(unanchored_wire(), self._co_signer)
        else:
            arguments = unanchored_wire()

        self.turn += 1
        tool_call_id = f"dispatch-wire-turn-{self.turn}"
        print(f"\n=== Turn {self.turn}: planner calls dispatch_wire ===")
        print(json.dumps(arguments, indent=2, sort_keys=True))
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "dispatch_wire", "args": arguments, "id": tool_call_id}
                        ],
                    )
                ],
                "tool_invocation": ToolInvocation(
                    name="dispatch_wire",
                    arguments=arguments,
                    tool_call_id=tool_call_id,
                ),
            },
            goto=STEER_NODE,
        )


def build_graph(
    *,
    client: RamenClient,
    memory_store: JSONFileMemoryStore,
    planner: ScriptedWirePlanner,
    dispatched: list[dict[str, Any]],
    provider_key: str | None,
    provider_name: str | None,
) -> Any:
    """Compile the planner <-> RamenSteerNode self-healing loop."""

    @tool
    def dispatch_wire(
        account_id: str,
        amount_usd: float,
        beneficiary_name: str,
        beneficiary_routing: str,
        beneficiary_account: str,
        sanction_clearance_token: str,
        gl_offset: str,
        co_signer_public_key: str,
        co_signer_signature: str,
    ) -> str:
        """Dispatch a governed commercial-loan wire transfer (simulated core banking)."""
        dispatched.append(
            {
                "account_id": account_id,
                "amount_usd": amount_usd,
                "beneficiary_name": beneficiary_name,
                "co_signer_public_key": co_signer_public_key,
            }
        )
        reference = hashlib.sha256(co_signer_signature.encode()).hexdigest()[:12].upper()
        return f"WIRE-{reference} queued: ${amount_usd:,.2f} to {beneficiary_name}"

    steer = RamenSteerNode(
        client=client,
        tools={"dispatch_wire": dispatch_wire},
        planner_node=PLANNER_NODE,
        bundle_ids=[FINTECH_BANKING_INVARIANCE_BUNDLE_ID],
        provider_key=provider_key,
        provider_name=provider_name,
        memory_store=memory_store,
        max_repair_turns=2,
    )
    workflow = StateGraph(SelfHealingState)
    workflow.add_node(PLANNER_NODE, planner, destinations=(STEER_NODE, END))
    workflow.add_node(STEER_NODE, steer, destinations=(PLANNER_NODE, END))
    workflow.add_edge(START, PLANNER_NODE)
    return workflow.compile()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--memory-path",
        default="agent_memory.json",
        help="Episodic memory JSON file (default: ./agent_memory.json)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("RAMEN_API_KEY")
    if not api_key:
        raise RuntimeError("RAMEN_API_KEY must be supplied through the environment")
    provider_key = os.environ.get("OPENAI_API_KEY") or None
    provider_name = "openai" if provider_key else None
    mode = "BYOK (openai)" if provider_key else "Enterprise managed-provider"
    print(f"ramen-ai provider mode: {mode}")

    memory_store = JSONFileMemoryStore(Path(args.memory_path))
    planner = ScriptedWirePlanner(Ed25519PrivateKey.generate())
    dispatched: list[dict[str, Any]] = []

    with RamenClient(api_key) as client:
        graph = build_graph(
            client=client,
            memory_store=memory_store,
            planner=planner,
            dispatched=dispatched,
            provider_key=provider_key,
            provider_name=provider_name,
        )
        final_state = graph.invoke(
            {"task": TASK, "messages": [HumanMessage(content=TASK)], "repair_turns": 0}
        )

    tool_messages = [m for m in final_state["messages"] if isinstance(m, ToolMessage)]
    print("\n=== Outcome ===")
    for index, message in enumerate(tool_messages, start=1):
        status = "BLOCKED" if message.status == "error" else "ALLOWED"
        print(f"Turn {index}: {status}")

    if final_state.get("governance_error"):
        print(f"Loop halted: {final_state['governance_error']}")
        return 1
    if len(tool_messages) < 2 or tool_messages[0].status != "error":
        print("Expected the unsigned wire to be blocked before the repair was allowed.")
        return 1

    final_metadata = tool_messages[-1].response_metadata
    receipt = final_metadata.get("receipt") or {}
    print(f"Dispatched wires: {len(dispatched)}")
    print(f"Tool result: {tool_messages[-1].content}")
    print(f"Receipt verified (Ed25519): {final_metadata.get('receipt_verified')}")
    print(f"Receipt id: {receipt.get('id')}")
    print(f"Receipt schema_version: {receipt.get('schema_version')}")
    print(f"Receipt kid: {receipt.get('kid')}")
    if final_metadata.get("memory_error"):
        print(f"Episodic memory error: {final_metadata['memory_error']}")
        return 1
    print(f"Exemplar recorded: {final_metadata.get('exemplar_id')} -> {memory_store.path}")

    exemplars = memory_store.retrieve_relevant_exemplars(
        fingerprint_task(TASK), "dispatch_wire"
    )
    print(f"Zero-turn retrieval for this task: {len(exemplars)} exemplar(s)")
    if exemplars:
        latest = exemplars[0]
        print(f"  anchor: {latest.primary_statutory_anchor}")
        print(f"  steering: {latest.steering_directive}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
