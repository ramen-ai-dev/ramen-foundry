"""Governed commercial lending and wire-disbursement tools."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, TypedDict

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command
from ramen_ai import RamenClient

from ramen_foundry.core import RamenToolNode, ToolInvocation

from ._security import validate_provider_configuration, validate_tool_registry

FINTECH_BANKING_INVARIANCE_BUNDLE_ID = "ramen__fintech_banking_invariance"
CREDIT_ADVERSE_ACTION_POLICY_ID = "796b7a87-d1f5-4ecc-91f2-a506a9b0d91e"
WIRE_DUAL_CONTROL_POLICY_ID = "b4c18ba1-26b7-4b7f-b44b-65e8de790572"

COMMERCIAL_LENDING_TOOL_NAMES = frozenset(
    {"issue_credit_adverse_action", "dispatch_wire"}
)


class CommercialLendingState(TypedDict, total=False):
    """State accepted and returned by the compiled lending workflow."""

    tool_invocation: ToolInvocation | Mapping[str, Any] | None
    governance_error: str | None
    messages: Annotated[list[Any], add_messages]


class CommercialLendingAgent:
    """Govern credit adverse actions and high-value wire disbursements.

    ``issue_credit_adverse_action`` binds the application decision, Regulation B
    reason codes, model identity, and SHAP attribution evidence for evaluation
    under CFPB Circular 2023-03, ECOA Regulation B, and FCRA section 615.

    ``dispatch_wire`` binds the account, beneficiary, amount, sanctions
    clearance, general-ledger offset, and Ed25519 dual-control evidence for
    evaluation under UCC section 4A-202 and the FinCEN Travel Rule.

    Host applications provide the actual LangChain tools and retain control of
    their banking infrastructure. Every invocation is evaluated against the
    immutable FinTech Banking Invariance bundle before tool execution.
    """

    def __init__(
        self,
        *,
        client: RamenClient,
        tools: Mapping[str, BaseTool],
        provider_key: str | None = None,
        provider_name: str | None = None,
    ) -> None:
        validate_provider_configuration(
            provider_key,
            provider_name,
            template_name=type(self).__name__,
        )
        registered = validate_tool_registry(
            tools,
            allowed_names=COMMERCIAL_LENDING_TOOL_NAMES,
            template_name=type(self).__name__,
        )
        self._tool_names = tuple(sorted(registered))
        self._guard = RamenToolNode(
            client=client,
            tools=registered,
            llm_node=END,
            bundle_ids=[FINTECH_BANKING_INVARIANCE_BUNDLE_ID],
            provider_key=provider_key,
            provider_name=provider_name,
        )
        self._graph = self._build_graph()

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Return the registered commercial-lending capability names."""

        return self._tool_names

    @property
    def graph(self) -> Any:
        """Return the compiled LangGraph workflow for host composition."""

        return self._graph

    def execute(
        self,
        tool_name: str,
        payload: Mapping[str, Any],
        *,
        tool_call_id: str | None = None,
    ) -> Command:
        """Evaluate and directly execute one resolved commercial-lending action."""

        invocation = ToolInvocation(
            name=tool_name,
            arguments=dict(payload),
            tool_call_id=tool_call_id or f"{tool_name}-direct",
        )
        return self._guard({"tool_invocation": invocation})

    def invoke(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Route lending state through the compiled governance workflow."""

        return self._graph.invoke(dict(state))

    def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Route lending state through the compiled governance workflow."""

        return self.invoke(state)

    def _build_graph(self) -> Any:
        workflow = StateGraph(CommercialLendingState)
        workflow.add_node("governed_commercial_lending_action", self._guard)
        workflow.add_edge(START, "governed_commercial_lending_action")
        workflow.add_edge("governed_commercial_lending_action", END)
        return workflow.compile()
