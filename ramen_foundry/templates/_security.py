"""Shared constraints for ramen-foundry security templates."""

from __future__ import annotations

from collections.abc import Mapping

from langchain_core.tools import BaseTool

SHIELD_CORE_IT_BUNDLE_ID = "ramen__shield_core_it"


def validate_provider_configuration(
    provider_key: str | None,
    provider_name: str | None,
    *,
    template_name: str,
) -> None:
    """Require BYOK credentials and provider routing to be configured together."""

    if provider_key is None and provider_name is None:
        return
    if (
        not provider_key
        or not provider_key.strip()
        or not provider_name
        or not provider_name.strip()
    ):
        raise ValueError(
            f"{template_name} requires provider_key and provider_name together"
        )


def validate_tool_registry(
    tools: Mapping[str, BaseTool],
    *,
    allowed_names: frozenset[str],
    template_name: str,
) -> dict[str, BaseTool]:
    """Return a validated copy of a template's host-supplied tool registry."""

    registered = dict(tools)
    if not registered:
        raise ValueError(f"{template_name} requires at least one registered tool")

    unsupported = sorted(set(registered) - allowed_names)
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(f"{template_name} does not permit tool names: {names}")

    mismatched = sorted(
        registry_name
        for registry_name, tool in registered.items()
        if tool.name != registry_name
    )
    if mismatched:
        names = ", ".join(mismatched)
        raise ValueError(
            f"{template_name} registry keys must match BaseTool names: {names}"
        )

    return registered
