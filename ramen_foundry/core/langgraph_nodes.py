"""Raw LangGraph nodes that enforce ramen-ai governance boundaries."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command
from pydantic import BaseModel, Field
from ramen_ai import (
    GovernanceDeniedException,
    GovernedGenerationException,
    RamenClient,
)

from .receipts import signed_allow_problem


class ToolInvocation(BaseModel):
    """A resolved tool call supplied by an LLM node to :class:`RamenToolNode`."""

    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    tool_call_id: str = Field(min_length=1)


class RamenToolNode:
    """Evaluate a resolved tool call before execution and route back to the LLM.

    The node expects ``state["tool_invocation"]`` to contain a ``ToolInvocation``
    value or equivalent mapping. It always returns a ``Command`` to ``llm_node``:
    a blocked or unverifiable request becomes an error ``ToolMessage`` without
    invoking the tool; an allowed request executes the registered tool and
    returns its result as a ``ToolMessage``.
    """

    def __init__(
        self,
        *,
        client: RamenClient,
        tools: Mapping[str, BaseTool],
        llm_node: str,
        policy_ids: Sequence[str] | None = None,
        bundle_ids: Sequence[str] | None = None,
        provider_key: str | None = None,
        provider_name: str | None = None,
        public_keys: Mapping[str, str] | None = None,
    ) -> None:
        if not policy_ids and not bundle_ids:
            raise ValueError("provide at least one policy_id or bundle_id")
        self._client = client
        self._tools = dict(tools)
        self._llm_node = llm_node
        # Receipt verification keys; None means the SDK's production keys.
        # Override only in tests.
        self._public_keys = dict(public_keys) if public_keys is not None else None
        self._policy_ids = list(policy_ids) if policy_ids else None
        self._bundle_ids = list(bundle_ids) if bundle_ids else None
        self._provider_key = provider_key
        self._provider_name = provider_name

    def __call__(self, state: Mapping[str, Any]) -> Command:
        invocation = ToolInvocation.model_validate(state["tool_invocation"])

        try:
            verdict = self._evaluate(invocation)
        except Exception as error:  # noqa: BLE001
            return self._return_error(
                invocation,
                f"Governance evaluation unavailable; tool was not executed: {error}",
            )

        if not verdict.get("allowed", False):
            return self._return_error(
                invocation,
                verdict.get("steering") or "Tool execution was blocked by ramen-ai policy.",
            )
        unverified = self._unverified_allow(invocation, verdict)
        if unverified:
            return self._return_error(
                invocation,
                f"Tool execution was denied because the governance receipt could not be verified: {unverified}",
            )

        tool = self._tools.get(invocation.name)
        if tool is None:
            return self._return_error(
                invocation,
                f"Tool '{invocation.name}' is not registered and was not executed.",
            )

        try:
            result = tool.invoke(invocation.arguments)
        except Exception as error:  # noqa: BLE001
            return self._return_error(
                invocation,
                f"Tool '{invocation.name}' failed: {error}",
            )

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=_stringify(result),
                        tool_call_id=invocation.tool_call_id,
                        name=invocation.name,
                    )
                ],
                "governance_error": None,
                "tool_invocation": None,
            },
            goto=self._llm_node,
        )

    @staticmethod
    def _payload(invocation: ToolInvocation) -> str:
        """Return the exact string evaluated (and hash-bound by the receipt)."""
        return json.dumps(
            {"tool": invocation.name, "arguments": invocation.arguments},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    def _unverified_allow(self, invocation: ToolInvocation, verdict: Mapping[str, Any]) -> str | None:
        """Return why an ALLOW verdict may not release the tool, or None.

        Requires the client's own ``receipt_verified`` and, independently, a
        receipt whose signature, input binding, and signed verdict (1) verify
        locally. The second check means dispatch never rests on a flag the
        client sets: a replayed, validly signed BLOCK receipt is refused.
        """
        if not verdict.get("receipt_verified", False):
            return verdict.get("receipt_reason") or "No verifiable governance receipt was returned."
        return signed_allow_problem(verdict, self._payload(invocation), self._public_keys)

    def _evaluate(self, invocation: ToolInvocation) -> dict[str, Any]:
        """Evaluate one resolved tool call through the ramen-ai SDK."""
        payload = self._payload(invocation)
        return self._client.evaluate_compliance(
            payload,
            policy_ids=self._policy_ids,
            bundle_ids=self._bundle_ids,
            context={"tool_name": invocation.name},
            provider_key=self._provider_key,
            provider_name=self._provider_name,
        )

    def _return_error(self, invocation: ToolInvocation, reason: str) -> Command:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Governance blocked tool execution: {reason}",
                        tool_call_id=invocation.tool_call_id,
                        name=invocation.name,
                        status="error",
                    )
                ],
                "governance_error": reason,
                "tool_invocation": None,
            },
            goto=self._llm_node,
        )


class RamenGovernedNode:
    """Generate final state content through ramen-ai's self-correcting cascade.

    The node reads a prompt from ``state[prompt_key]`` and writes only content
    released by ``RamenClient.generate_governed`` to ``content_key``. A governed
    denial or transport failure is represented in ``governance_error`` rather
    than being emitted as generated content.
    """

    def __init__(
        self,
        *,
        client: RamenClient,
        policy_ids: Sequence[str] | None = None,
        bundle_ids: Sequence[str] | None = None,
        prompt_key: str = "governed_prompt",
        content_key: str = "governed_content",
        provider_key: str | None = None,
        provider_name: str | None = None,
    ) -> None:
        if not policy_ids and not bundle_ids:
            raise ValueError("provide at least one policy_id or bundle_id")
        self._client = client
        self._policy_ids = list(policy_ids) if policy_ids else None
        self._bundle_ids = list(bundle_ids) if bundle_ids else None
        self._prompt_key = prompt_key
        self._content_key = content_key
        self._provider_key = provider_key
        self._provider_name = provider_name

    def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        prompt = state.get(self._prompt_key)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"state['{self._prompt_key}'] must be a non-blank string")

        try:
            result = self._client.generate_governed(
                prompt,
                policy_ids=self._policy_ids,
                bundle_ids=self._bundle_ids,
                max_retries=1,
                provider_key=self._provider_key,
                provider_name=self._provider_name,
            )
        except GovernanceDeniedException as error:
            reason = _governed_denial_reason(error)
            return {
                self._content_key: None,
                "governance_error": reason,
                "messages": [AIMessage(content=f"Governed generation blocked: {reason}")],
            }
        except GovernedGenerationException as error:
            return {
                self._content_key: None,
                "governance_error": f"Governed generation unavailable ({error.code}): {error}",
                "messages": [AIMessage(content="Governed generation was unavailable.")],
            }
        except Exception as error:  # noqa: BLE001
            return {
                self._content_key: None,
                "governance_error": f"Governed generation failed: {error}",
                "messages": [AIMessage(content="Governed generation failed.")],
            }

        return {
            self._content_key: result.content,
            "governed_evaluation": result.evaluation,
            "governance_error": None,
            "messages": [AIMessage(content=result.content)],
        }


def _governed_denial_reason(error: GovernanceDeniedException) -> str:
    rationales = [
        rationale
        for attempt in error.data.attempt_metadata
        for rationale in attempt.steering_rationale or []
    ]
    return " | ".join(rationales) if rationales else str(error)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)
