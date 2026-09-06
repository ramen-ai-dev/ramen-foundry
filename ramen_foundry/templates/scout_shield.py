"""Governed deep-research and untrusted-content tools."""

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

SCOUT_SHIELD_TOOL_NAMES = frozenset(
    {"extract_content", "fetch_url", "publish_research", "read_local_file"}
)


class ScoutShieldAgent:
    """Guard deep-research actions influenced by untrusted web content.

    Host applications provide fetch, extraction, local-file, and publication
    tools. Every resolved invocation is evaluated against the Core IT bundle,
    including its OWASP ASI06 Indirect Prompt Injection and Secret Exfiltration
    controls, before any capability can run. Attempts induced by scraped text to
    read ``.env`` files, collect credentials, or transmit secrets therefore fail
    closed when blocked, unverifiable, or unavailable.

    The invocation arguments must include the requested URL/path, destination,
    relevant untrusted instruction, and any proposed outbound payload. Hidden
    tool-side behavior cannot be evaluated and must not receive ambient secrets.
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
            allowed_names=SCOUT_SHIELD_TOOL_NAMES,
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
