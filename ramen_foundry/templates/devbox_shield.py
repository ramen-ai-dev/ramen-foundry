"""Governed workstation inspection and cleanup tools."""

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

DEVBOX_SHIELD_TOOL_NAMES = frozenset(
    {"inspect_directory", "delete_path", "terminate_process"}
)


class DevboxShieldAgent:
    """Guard workstation inspection and cleanup actions before execution.

    Host applications provide the platform-specific tools. Only the declared
    devbox capabilities can be registered, and every invocation is evaluated
    against the Core IT bundle before the tool is selected or executed. The
    bundle covers destructive execution, system-path deletion, shell
    configuration damage, privilege abuse, and secret exfiltration.

    Safety-significant values such as absolute paths, recursive flags, process
    IDs, owners, and working directories must be present in invocation
    arguments so they are bound into the evaluated payload and signed receipt.
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
            allowed_names=DEVBOX_SHIELD_TOOL_NAMES,
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
