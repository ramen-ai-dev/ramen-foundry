"""Two-agent collective learning relay through ramen forge.

Agent 1 (novice, blank memory) receives a credit adverse-action task:

* Turn 1: it cites a ZIP-code proxy reason and ``RamenSteerNode`` blocks the
  call before dispatch under ECOA / Regulation B.
* Turn 2: it follows the steering directive, cites reasons attributable to the
  model's negative SHAP factors, and is allowed with a verified Schema V5
  receipt. ``RamenSteerNode`` records the repair; when a ramen forge write
  token is configured, ``RemoteForgeMemoryStore`` contributes it to the
  Level 1 community commons.

Agent 2 (apprentice) is a fresh graph with an empty conversation history,
connected to ramen forge with ``RemoteForgeMemoryStore``:

* Turn 0: it queries ramen forge for exemplars matching the task fingerprint.
* Turn 1: it applies the recalled repair to its first draft and is allowed on
  the first attempt. ramen-ai still evaluates the call; recalled memory is
  treated as untrusted guidance, so only the reason-code list is taken from it.

Both planners are deterministic so the relay is reproducible; in production
they are LLM nodes. The notice tool is simulated and sends nothing.

Run from the repository root (makes live ramen-ai and ramen forge calls)::

    RAMEN_API_KEY=... python examples/demonstrate_collective_learning.py

Configuration:

``RAMEN_API_KEY`` (required)
    ramen-ai credential.
``OPENAI_API_KEY`` (optional)
    Starter/Professional BYOK. When unset, provider key and name are both
    omitted so ramen-ai uses Enterprise managed-provider inference.
``FORGE_WRITE_TOKEN`` (optional)
    ramen forge write token. Read from the environment first, then from
    ``ramen-ai-integrations/.env``. Without it, Agent 1's repair is kept in a
    local Level 0 store and not contributed, so Agent 2 can only inherit
    repairs that are already in the commons.
``--forge-url``
    ramen forge base URL (default: the production service). Contributions are
    publicly readable, including the task text and tool arguments.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command
from ramen_ai import RamenClient

from ramen_foundry import (
    BaseEpisodicMemoryStore,
    CorrectionExemplar,
    JSONFileMemoryStore,
    RamenSteerNode,
    RemoteForgeMemoryStore,
    ToolInvocation,
)
from ramen_foundry.core.memory import fingerprint_task
from ramen_foundry.core.steer_node import REFLECTION_HEADER
from ramen_foundry.templates.fintech import CREDIT_ADVERSE_ACTION_POLICY_ID

DEFAULT_FORGE_URL = "https://ramen-forge.ramenai.workers.dev"
INTEGRATIONS_ENV_PATH = Path(__file__).resolve().parents[2] / "ramen-ai-integrations" / ".env"
FORGE_DOMAIN = "fintech"
PLANNER_NODE = "planner"
STEER_NODE = "ramen_steer"
TOOL_NAME = "issue_credit_adverse_action"
TASK = "Formulate an adverse action notice for credit application APP-99214."
MODEL_HASH = "sha256:1ea1336c6faca52fe43060bf122c129ffb449172855c7612399db95d1cc1e7b4"
SHAP_ATTRIBUTION_SUMMARY: dict[str, float] = {
    "checking_balance": -0.6602,
    "repayment_duration": -0.5456,
    "savings_balance": -0.2627,
    "employment_years": 0.3120,
}
GEOGRAPHIC_PROXY_REASONS = ["REGIONAL_ECONOMIC_VOLATILITY_ZIP_CODE"]
ATTRIBUTED_REASONS = ["INSUFFICIENT_LIQUIDITY", "EXCESSIVE_REPAYMENT_TERM"]


class RelayState(TypedDict, total=False):
    """Graph state shared by each planner and its RamenSteerNode."""

    task: str
    messages: Annotated[list[Any], add_messages]
    tool_invocation: ToolInvocation | Mapping[str, Any] | None
    governance_error: str | None
    repair_turns: int
    pending_correction: dict[str, Any] | None


@dataclass
class AgentRun:
    """Outcome of one agent's run, used for the comparison summary."""

    name: str
    verdicts: list[str] = field(default_factory=list)
    dispatched: list[list[str]] = field(default_factory=list)
    recalled: list[CorrectionExemplar] = field(default_factory=list)
    final_metadata: dict[str, Any] = field(default_factory=dict)
    governance_error: str | None = None

    @property
    def turns(self) -> int:
        return len(self.verdicts)

    @property
    def failures(self) -> int:
        return self.verdicts.count("BLOCKED")


def read_dotenv_value(path: Path, key: str) -> str | None:
    """Return one ``KEY=value`` entry from a dotenv file without exporting the rest."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        name, separator, value = line.strip().partition("=")
        if separator and name.removeprefix("export ").strip() == key:
            value = value.strip().strip('"').strip("'")
            return value or None
    return None


def adverse_action_payload(reason_codes: list[str]) -> dict[str, Any]:
    """Return the adverse-action arguments for the demo application."""
    return {
        "application_id": "APP-99214",
        "decision": "DENIED",
        "reg_b_reason_codes": list(reason_codes),
        "model_hash": MODEL_HASH,
        "shap_attribution_summary": dict(SHAP_ATTRIBUTION_SUMMARY),
    }


def recalled_reason_codes(exemplars: list[CorrectionExemplar]) -> list[str] | None:
    """Take only a well-formed reason-code list from untrusted recalled memory."""
    for exemplar in exemplars:
        codes = exemplar.repaired_arguments.get("reg_b_reason_codes")
        if (
            isinstance(codes, list)
            and codes
            and all(isinstance(code, str) and code.isupper() and len(code) <= 64 for code in codes)
        ):
            return list(codes)
    return None


def _tool_call(turn: int, arguments: dict[str, Any], agent: str) -> Command:
    tool_call_id = f"{agent}-{TOOL_NAME}-turn-{turn}"
    print(f"  Turn {turn}: {TOOL_NAME}(reg_b_reason_codes={arguments['reg_b_reason_codes']})")
    return Command(
        update={
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"name": TOOL_NAME, "args": arguments, "id": tool_call_id}],
                )
            ],
            "tool_invocation": ToolInvocation(
                name=TOOL_NAME, arguments=arguments, tool_call_id=tool_call_id
            ),
        },
        goto=STEER_NODE,
    )


class UnderwritingPlanner:
    """Deterministic planner: drafts a notice, optionally from recalled memory."""

    def __init__(self, agent: str, recalled: list[CorrectionExemplar]) -> None:
        self._agent = agent
        self._recalled_codes = recalled_reason_codes(recalled)
        self.turn = 0

    def __call__(self, state: Mapping[str, Any]) -> Command:
        messages = state.get("messages") or []
        last = messages[-1] if messages else None
        if isinstance(last, ToolMessage) and last.status != "error":
            return Command(update={"messages": [AIMessage(content="Notice issued.")]}, goto=END)

        if isinstance(last, ToolMessage) and REFLECTION_HEADER in str(last.content):
            print("  --- reflection received ---")
            for line in str(last.content).splitlines():
                print(f"    {line}")
            reasons = ATTRIBUTED_REASONS
        elif self._recalled_codes is not None:
            print(f"  Planner: applying recalled repair -> {self._recalled_codes}")
            reasons = self._recalled_codes
        else:
            reasons = GEOGRAPHIC_PROXY_REASONS

        self.turn += 1
        return _tool_call(self.turn, adverse_action_payload(reasons), self._agent)


def run_agent(
    *,
    name: str,
    client: RamenClient,
    memory_store: BaseEpisodicMemoryStore,
    recalled: list[CorrectionExemplar],
    provider_key: str | None,
    provider_name: str | None,
) -> AgentRun:
    """Build a fresh planner <-> RamenSteerNode graph and run the task once."""
    run = AgentRun(name=name, recalled=recalled)

    @tool
    def issue_credit_adverse_action(
        application_id: str,
        decision: str,
        reg_b_reason_codes: list[str],
        model_hash: str,
        shap_attribution_summary: dict[str, float],
    ) -> str:
        """Issue a governed adverse-action notice (simulated; nothing is sent)."""
        run.dispatched.append(list(reg_b_reason_codes))
        return f"[simulated] notice queued for {application_id}: {', '.join(reg_b_reason_codes)}"

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
    workflow = StateGraph(RelayState)
    workflow.add_node(PLANNER_NODE, UnderwritingPlanner(name, recalled), destinations=(STEER_NODE, END))
    workflow.add_node(STEER_NODE, steer, destinations=(PLANNER_NODE, END))
    workflow.add_edge(START, PLANNER_NODE)
    final_state = workflow.compile().invoke(
        {"task": TASK, "messages": [HumanMessage(content=TASK)], "repair_turns": 0}
    )

    tool_messages = [m for m in final_state["messages"] if isinstance(m, ToolMessage)]
    run.verdicts = ["BLOCKED" if m.status == "error" else "ALLOWED" for m in tool_messages]
    for turn, verdict in enumerate(run.verdicts, start=1):
        print(f"  Turn {turn} verdict: [{verdict}]")
    if tool_messages:
        run.final_metadata = dict(tool_messages[-1].response_metadata)
    run.governance_error = final_state.get("governance_error")
    return run


def print_receipt(run: AgentRun) -> None:
    receipt = run.final_metadata.get("receipt") or {}
    print(f"  Receipt verified (Ed25519): {run.final_metadata.get('receipt_verified')}")
    print(f"  Receipt id: {receipt.get('id')} (schema {receipt.get('schema_version')}, kid {receipt.get('kid')})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--forge-url", default=DEFAULT_FORGE_URL, help="ramen forge base URL")
    args = parser.parse_args()

    api_key = os.environ.get("RAMEN_API_KEY")
    if not api_key:
        raise RuntimeError("RAMEN_API_KEY must be supplied through the environment")
    provider_key = os.environ.get("OPENAI_API_KEY") or None
    provider_name = "openai" if provider_key else None
    forge_token = os.environ.get("FORGE_WRITE_TOKEN") or read_dotenv_value(
        INTEGRATIONS_ENV_PATH, "FORGE_WRITE_TOKEN"
    )

    print(f"ramen-ai provider mode: {'BYOK (openai)' if provider_key else 'Enterprise managed-provider'}")
    print(f"ramen forge: {args.forge_url} (domain={FORGE_DOMAIN}, contribution {'enabled' if forge_token else 'disabled: no FORGE_WRITE_TOKEN'})")
    print(f"Task: {TASK}")
    fingerprint = fingerprint_task(TASK)

    with RamenClient(api_key) as client, tempfile.TemporaryDirectory() as scratch:
        print("\n=== Agent 1 (novice, blank memory) ===")
        novice_store: BaseEpisodicMemoryStore
        if forge_token:
            novice_store = RemoteForgeMemoryStore(
                base_url=args.forge_url, write_token=forge_token, domain=FORGE_DOMAIN
            )
        else:
            novice_store = JSONFileMemoryStore(Path(scratch) / "agent_memory.json")
        novice = run_agent(
            name="agent-1",
            client=client,
            memory_store=novice_store,
            recalled=[],
            provider_key=provider_key,
            provider_name=provider_name,
        )
        print_receipt(novice)
        contributed_id = novice.final_metadata.get("exemplar_id")
        if novice.final_metadata.get("memory_error"):
            print(f"  Contribution failed: {novice.final_metadata['memory_error']}")
            contributed_id = None
        elif contributed_id and forge_token:
            print(f"  Contributed exemplar {contributed_id} to ramen forge")
        elif contributed_id:
            print(f"  Recorded exemplar {contributed_id} locally only (Level 0)")

        print("\n=== Agent 2 (apprentice, connected to ramen forge) ===")
        apprentice_store = RemoteForgeMemoryStore(
            base_url=args.forge_url, write_token=forge_token, domain=FORGE_DOMAIN
        )
        recalled = apprentice_store.retrieve_relevant_exemplars(fingerprint, TOOL_NAME)
        print(f"  Turn 0: recalled {len(recalled)} exemplar(s) from ramen forge")
        for exemplar in recalled:
            origin = "Agent 1 (this run)" if exemplar.exemplar_id == contributed_id else "earlier contributor"
            print(f"    {exemplar.exemplar_id} [{origin}] anchor: {exemplar.primary_statutory_anchor}")
        # A read-only forge connection is used for recall only; any repair the
        # apprentice makes without a write token stays in a local Level 0 store.
        apprentice_memory: BaseEpisodicMemoryStore = (
            apprentice_store
            if forge_token
            else JSONFileMemoryStore(Path(scratch) / "apprentice_memory.json")
        )
        apprentice = run_agent(
            name="agent-2",
            client=client,
            memory_store=apprentice_memory,
            recalled=recalled,
            provider_key=provider_key,
            provider_name=provider_name,
        )
        print_receipt(apprentice)

    print("\n=== Collective learning summary ===")
    print(f"{'Agent':<28}{'Turns':>7}{'Blocked':>9}{'First-pass':>12}")
    for run in (novice, apprentice):
        first_pass = "yes" if run.verdicts[:1] == ["ALLOWED"] else "no"
        print(f"{run.name + (' (novice)' if run is novice else ' (apprentice)'):<28}{run.turns:>7}{run.failures:>9}{first_pass:>12}")

    relay_ok = (
        novice.verdicts == ["BLOCKED", "ALLOWED"]
        and novice.final_metadata.get("receipt_verified") is True
        and bool(recalled)
        and apprentice.verdicts == ["ALLOWED"]
        and apprentice.final_metadata.get("receipt_verified") is True
    )
    if relay_ok:
        saved = novice.turns - apprentice.turns
        print(f"\nRelay verified: Agent 2 passed on its first attempt, saving {saved} turn(s) and {novice.failures} blocked call(s).")
        return 0
    print("\nRelay not demonstrated:")
    if novice.governance_error or apprentice.governance_error:
        print(f"  governance_error: {apprentice.governance_error or novice.governance_error}")
    if not recalled:
        print("  Agent 2 recalled no exemplars. Configure FORGE_WRITE_TOKEN so Agent 1 can contribute its repair.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
