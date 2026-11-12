"""Shared base class for governed tool-execution templates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, ClassVar, TypedDict

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command
from ramen_ai import RamenClient

from ramen_foundry.core import RamenToolNode, ToolInvocation

from ._security import validate_provider_configuration, validate_tool_registry


class GovernedToolState(TypedDict, total=False):
    """State accepted and returned by a compiled governed-tool workflow."""

    tool_invocation: ToolInvocation | Mapping[str, Any] | None
    governance_error: str | None
    messages: Annotated[list[Any], add_messages]


class GovernedToolAgent:
    """Base class for templates that gate host tools behind one policy bundle.

    A subclass declares ``bundle_id`` and ``allowed_tool_names`` and may override
    :meth:`_preflight` to run deterministic local checks. The base class owns the
    parts that must not drift between templates:

    * the :class:`RamenToolNode` instantiation, bound to exactly one bundle;
    * the ``shadow_mode=False`` invariant. The constructor accepts no
      ``shadow_mode`` argument and ``RAMEN_SHADOW_MODE`` is ignored, so a
      template built on this class always enforces BLOCK verdicts;
    * the single-node LangGraph workflow, ``execute``, ``invoke`` and ``__call__``;
    * registry validation (only declared tool names, keys equal to tool names).

    :meth:`_preflight` runs before the backend is contacted. A returned string
    blocks the call locally: the tool is not run, no evaluation request is made,
    and the string becomes ``governance_error``. An exception raised by
    ``_preflight`` also blocks the call (fail closed).
    """

    bundle_id: ClassVar[str] = ""
    allowed_tool_names: ClassVar[frozenset[str]] = frozenset()
    node_name: ClassVar[str] = "governed_action"

    def __init__(
        self,
        *,
        client: RamenClient,
        tools: Mapping[str, BaseTool],
        provider_key: str | None = None,
        provider_name: str | None = None,
    ) -> None:
        template_name = type(self).__name__
        if not self.bundle_id or not self.allowed_tool_names:
            raise TypeError(
                f"{template_name} must define bundle_id and allowed_tool_names "
                "(GovernedToolAgent cannot be used directly)"
            )
        validate_provider_configuration(
            provider_key,
            provider_name,
            template_name=template_name,
        )
        registered = validate_tool_registry(
            tools,
            allowed_names=self.allowed_tool_names,
            template_name=template_name,
        )
        self._tool_names = tuple(sorted(registered))
        self._guard = RamenToolNode(
            client=client,
            tools=registered,
            llm_node=END,
            bundle_ids=[self.bundle_id],
            provider_key=provider_key,
            provider_name=provider_name,
            # High-assurance template: always enforce, ignoring RAMEN_SHADOW_MODE.
            shadow_mode=False,
        )
        self._graph = self._build_graph()

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Return the registered capability names."""

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
        """Evaluate and directly execute one resolved action."""

        invocation = ToolInvocation(
            name=tool_name,
            arguments=dict(payload),
            tool_call_id=tool_call_id or f"{tool_name}-direct",
        )
        return self._gate({"tool_invocation": invocation})

    def invoke(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Route state through the compiled governance workflow."""

        return self._graph.invoke(dict(state))

    def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Route state through the compiled governance workflow."""

        return self.invoke(state)

    def _preflight(self, invocation: ToolInvocation) -> str | None:
        """Return a reason to block ``invocation`` locally, or ``None`` to continue."""

        return None

    def _bind_invocation(self, invocation: ToolInvocation) -> ToolInvocation:
        """Return the invocation that is evaluated and executed after a clean pre-flight.

        Subclasses may add facts they derived and verified locally. The returned
        invocation is exactly what the backend evaluates, what the receipt binds,
        and what the tool receives.
        """

        return invocation

    def _gate(self, state: GovernedToolState) -> Command:
        invocation = ToolInvocation.model_validate(state["tool_invocation"])
        try:
            reason = self._preflight(invocation)
        except Exception as error:  # noqa: BLE001
            reason = f"Local pre-flight check failed; tool was not executed: {error}"
        if reason is None:
            try:
                invocation = self._bind_invocation(invocation)
            except Exception as error:  # noqa: BLE001
                reason = f"Local evidence binding failed; tool was not executed: {error}"
        if reason is not None:
            return self._guard._return_error(invocation, reason)
        return self._guard({"tool_invocation": invocation})

    def _build_graph(self) -> Any:
        workflow = StateGraph(GovernedToolState)
        workflow.add_node(self.node_name, self._gate)
        workflow.add_edge(START, self.node_name)
        workflow.add_edge(self.node_name, END)
        return workflow.compile()
