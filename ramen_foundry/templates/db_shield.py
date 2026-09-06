"""Governed database triage and inspection tools."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.tools import BaseTool
from langgraph.types import Command
from ramen_ai import RamenClient

from ramen_foundry.core import RamenToolNode, ToolInvocation

from ._security import (
    SHIELD_CORE_IT_BUNDLE_ID,
    validate_provider_configuration,
    validate_tool_registry,
)

DB_SHIELD_TOOL_NAMES = frozenset(
    {"explain_query", "inspect_deadlocks", "run_query", "terminate_backend"}
)


class DbShieldAgent:
    """Guard database diagnosis and intervention actions before execution.

    The host supplies database-specific tools and credentials. The template
    permits read-oriented query inspection, EXPLAIN, deadlock inspection, and
    controlled ``pg_terminate_backend`` adapters. Every invocation is evaluated
    against the Core IT bundle before execution so destructive DDL/DML such as
    DROP, TRUNCATE, and unsafe bulk mutations can fail closed.

    Query text, bind parameters, database/environment identity, mutation scope,
    index evidence, target backend PID, and dry-run state should be included in
    invocation arguments whenever applicable. Database tools should still use
    least-privilege roles, parameterized SQL, timeouts, and transactions.
    """

    def __init__(
        self,
        *,
        client: RamenClient,
        tools: Mapping[str, BaseTool],
        llm_node: str = "assistant",
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
            allowed_names=DB_SHIELD_TOOL_NAMES,
            template_name=type(self).__name__,
        )
        self._tool_names = tuple(sorted(registered))
        self._guard = RamenToolNode(
            client=client,
            tools=registered,
            llm_node=llm_node,
            bundle_ids=[SHIELD_CORE_IT_BUNDLE_ID],
            provider_key=provider_key,
            provider_name=provider_name,
        )

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Return the registered capability names."""

        return self._tool_names

    def execute(self, invocation: ToolInvocation | Mapping[str, Any]) -> Command:
        """Evaluate one resolved action and execute it only when verified."""

        return self._guard({"tool_invocation": invocation})

    def __call__(self, state: Mapping[str, Any]) -> Command:
        """Run as a LangGraph node using ``state['tool_invocation']``."""

        return self._guard(state)
