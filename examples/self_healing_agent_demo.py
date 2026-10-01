"""Closed-loop self-healing credit adverse-action agent built on RamenSteerNode.

An underwriting agent must issue an adverse-action notice for a declined
commercial credit application. ``RamenSteerNode`` gates every
``issue_credit_adverse_action`` call against the credit adverse-action policy
(ECOA Regulation B / CFPB Circular 2023-03) before dispatch:

* Turn 1: the planner cites a geographic proxy
  (``REGIONAL_ECONOMIC_VOLATILITY_ZIP_CODE``) that is not supported by the
  model's attribution evidence. The call is blocked pre-dispatch, and the
  planner receives a structured reflection carrying the rule, primary
  statutory anchor, and the ramen-ai steering directive.
* Turn 2: the planner reads the directive and replaces the reason codes with
  ones attributable to the model's strongest negative SHAP factors
  (``checking_balance`` -> ``INSUFFICIENT_LIQUIDITY``, ``repayment_duration``
  -> ``EXCESSIVE_REPAYMENT_TERM``). The repair is allowed, the tool is
  dispatched, the verified Schema V5 receipt is surfaced in the tool metadata,
  and the correction exemplar is written to ``agent_memory.json``.

The planner is deterministic so the loop is reproducible; in production it is
an LLM node that reads the same reflection message. The notice tool is
simulated: it records the approved notice and sends nothing to an applicant.

Run from the repository root (makes live ramen-ai calls)::

    RAMEN_API_KEY=... python examples/self_healing_agent_demo.py

``RAMEN_API_KEY`` is required. When ``OPENAI_API_KEY`` is set, evaluations use
OpenAI BYOK (Starter/Professional). When it is unset, provider key and name are
both omitted so ramen-ai uses Enterprise managed-provider inference.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command
from ramen_ai import RamenClient

from ramen_foundry import JSONFileMemoryStore, RamenSteerNode, ToolInvocation
from ramen_foundry.core.memory import fingerprint_task
from ramen_foundry.core.steer_node import REFLECTION_HEADER
from ramen_foundry.templates.fintech import CREDIT_ADVERSE_ACTION_POLICY_ID

PLANNER_NODE = "planner"
STEER_NODE = "ramen_steer"
TOOL_NAME = "issue_credit_adverse_action"
TASK = (
    "Formulate a compliant adverse action notice for a denied commercial credit "
    "application based on model attribution."
)
MODEL_HASH = "sha256:1ea1336c6faca52fe43060bf122c129ffb449172855c7612399db95d1cc1e7b4"
SHAP_ATTRIBUTION_SUMMARY: dict[str, float] = {
    "checking_balance": -0.6602,
    "repayment_duration": -0.5456,
    "savings_balance": -0.2627,
    "employment_years": 0.3120,
}
GEOGRAPHIC_PROXY_REASONS = ["REGIONAL_ECONOMIC_VOLATILITY_ZIP_CODE"]
ATTRIBUTED_REASONS = ["INSUFFICIENT_LIQUIDITY", "EXCESSIVE_REPAYMENT_TERM"]


class SelfHealingState(TypedDict, total=False):
    """Graph state shared by the planner and RamenSteerNode."""

    task: str
    messages: Annotated[list[Any], add_messages]
    tool_invocation: ToolInvocation | Mapping[str, Any] | None
    governance_error: str | None
    repair_turns: int
    pending_correction: dict[str, Any] | None


def adverse_action_payload(reason_codes: list[str]) -> dict[str, Any]:
    """Return the adverse-action arguments for the demo application."""
    return {
        "application_id": "APP-CREDIT-99214",
        "decision": "DENIED",
        "reg_b_reason_codes": list(reason_codes),
        "model_hash": MODEL_HASH,
        "shap_attribution_summary": dict(SHAP_ATTRIBUTION_SUMMARY),
    }


def _reflection_field(reflection: str, label: str) -> str:
    prefix = f"{label}:"
    for line in reflection.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return ""


class ScriptedUnderwritingPlanner:
    """Deterministic planner that drafts a notice and repairs it from a reflection."""

    def __init__(self) -> None:
        self.turn = 0

    def __call__(self, state: Mapping[str, Any]) -> Command:
        messages = state.get("messages") or []
        last = messages[-1] if messages else None

        if isinstance(last, ToolMessage) and last.status != "error":
            return Command(
                update={"messages": [AIMessage(content=f"Notice issued: {last.content}")]},
                goto=END,
            )

        if isinstance(last, ToolMessage) and REFLECTION_HEADER in str(last.content):
            print("\n--- Planner scratchpad: reading reflection ---")
            print(last.content)
            directive = _reflection_field(str(last.content), "Steering Directive")
            print(f"\nPlanner: applying steering directive -> {directive}")
            print("Planner: citing reasons attributable to the model's negative SHAP factors.")
            arguments = adverse_action_payload(ATTRIBUTED_REASONS)
        else:
            arguments = adverse_action_payload(GEOGRAPHIC_PROXY_REASONS)

        self.turn += 1
        tool_call_id = f"{TOOL_NAME}-turn-{self.turn}"
        print(
            f"\n=== Turn {self.turn}: planner calls {TOOL_NAME}"
            f"(reg_b_reason_codes={arguments['reg_b_reason_codes']}) ==="
        )
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": TOOL_NAME, "args": arguments, "id": tool_call_id}],
                    )
                ],
                "tool_invocation": ToolInvocation(
                    name=TOOL_NAME,
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
    planner: ScriptedUnderwritingPlanner,
    dispatched: list[dict[str, Any]],
    provider_key: str | None,
    provider_name: str | None,
) -> Any:
    """Compile the planner <-> RamenSteerNode self-healing loop."""

    @tool
    def issue_credit_adverse_action(
        application_id: str,
        decision: str,
        reg_b_reason_codes: list[str],
        model_hash: str,
        shap_attribution_summary: dict[str, float],
    ) -> str:
        """Issue a governed adverse-action notice (simulated; nothing is sent)."""
        dispatched.append(
            {"application_id": application_id, "reg_b_reason_codes": reg_b_reason_codes}
        )
        return (
            f"[simulated] adverse-action notice queued for {application_id}: "
            f"{', '.join(reg_b_reason_codes)}"
        )

    steer = RamenSteerNode(
        client=client,
        tools={TOOL_NAME: issue_credit_adverse_action},
        planner_node=PLANNER_NODE,
        policy_ids=[CREDIT_ADVERSE_ACTION_POLICY_ID],
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
    print(f"Policy: {CREDIT_ADVERSE_ACTION_POLICY_ID}")
    print(f"Task: {TASK}")

    memory_store = JSONFileMemoryStore(Path(args.memory_path))
    dispatched: list[dict[str, Any]] = []

    with RamenClient(api_key) as client:
        graph = build_graph(
            client=client,
            memory_store=memory_store,
            planner=ScriptedUnderwritingPlanner(),
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
        print(f"Turn {index}: [{status}]")
    print(f"Dispatched notices: {dispatched}")

    if final_state.get("governance_error"):
        print(f"Loop halted: {final_state['governance_error']}")
        return 1
    if len(tool_messages) < 2 or tool_messages[0].status != "error":
        print("Expected the proxy-based notice to be blocked before the repair was allowed.")
        return 1
    if [notice["reg_b_reason_codes"] for notice in dispatched] != [ATTRIBUTED_REASONS]:
        print("Expected only the attributed notice to be dispatched.")
        return 1

    final_metadata = tool_messages[-1].response_metadata
    receipt = final_metadata.get("receipt") or {}
    print(f"Tool result: {tool_messages[-1].content}")
    print(f"Receipt verified (Ed25519): {final_metadata.get('receipt_verified')}")
    print(f"Receipt id: {receipt.get('id')}")
    print(f"Receipt schema_version: {receipt.get('schema_version')}")
    print(f"Receipt kid: {receipt.get('kid')}")
    print(f"Receipt signature: {receipt.get('signature')}")
    if final_metadata.get("memory_error"):
        print(f"Episodic memory error: {final_metadata['memory_error']}")
        return 1
    if not final_metadata.get("receipt_verified") or not final_metadata.get("exemplar_id"):
        print("Expected a verified receipt and a recorded exemplar for the repair.")
        return 1
    print(f"Exemplar recorded: {final_metadata['exemplar_id']} -> {memory_store.path}")

    exemplars = memory_store.retrieve_relevant_exemplars(fingerprint_task(TASK), TOOL_NAME)
    print(f"Zero-turn retrieval for this task: {len(exemplars)} exemplar(s)")
    if exemplars:
        latest = exemplars[0]
        print(f"  failed reasons:   {latest.failed_arguments['reg_b_reason_codes']}")
        print(f"  repaired reasons: {latest.repaired_arguments['reg_b_reason_codes']}")
        print(f"  anchor:   {latest.primary_statutory_anchor}")
        print(f"  steering: {latest.steering_directive}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
