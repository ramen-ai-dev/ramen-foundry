"""Raw LangGraph nodes that enforce ramen-ai governance boundaries."""

from __future__ import annotations

import json
import logging
import os
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

from .receipts import VERDICT_BLOCK, extract_receipt, signed_allow_problem, signed_verdict_problem

SHADOW_MODE_ENV = "RAMEN_SHADOW_MODE"
SHADOW_METADATA_KEY = "ramen_shadow"
SHADOW_LOG_HEADER = "[RAMEN SHADOW MODE: POLICY VIOLATION NOT ENFORCED]"
_SHADOW_TRUE = frozenset({"1", "true", "yes", "on"})
_SHADOW_FALSE = frozenset({"0", "false", "no", "off", ""})

DEFAULT_STATUTORY_ANCHOR = "General Operational Boundary"
_DEFAULT_RULE_ID = "UNSPECIFIED_RULE"
_DEFAULT_REASON = "Blocked by ramen-ai policy."
_DEFAULT_STEERING = "Revise the tool call so that it satisfies the governing policy."

shadow_logger = logging.getLogger("ramen_foundry.shadow")


def resolve_shadow_mode(
    shadow_mode: bool | None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Resolve the effective shadow-mode flag.

    An explicit ``True``/``False`` argument wins. ``None`` defers to the
    ``RAMEN_SHADOW_MODE`` environment variable, parsed strictly
    (case-insensitive, surrounding whitespace ignored): ``1/true/yes/on``
    enables, ``0/false/no/off`` or an empty value disables, and an unset
    variable means disabled. Any other value raises ``ValueError`` so a typo
    can never silently change enforcement.
    """
    if shadow_mode is not None:
        if not isinstance(shadow_mode, bool):
            raise ValueError("shadow_mode must be True, False, or None")
        return shadow_mode
    env = os.environ if environ is None else environ
    raw = env.get(SHADOW_MODE_ENV)
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in _SHADOW_TRUE:
        return True
    if value in _SHADOW_FALSE:
        return False
    raise ValueError(
        f"{SHADOW_MODE_ENV} must be one of 1/true/yes/on or 0/false/no/off (or empty); "
        f"got {raw!r}"
    )


class _ToolDispatchError(Exception):
    """The registered tool could not be dispatched; the message is the reason."""


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

    ``shadow_mode`` (disabled by default; ``None`` reads ``RAMEN_SHADOW_MODE``)
    turns a policy BLOCK into a logged, non-enforced observation: the violation,
    steering directive, and receipt are logged at WARNING on
    ``ramen_foundry.shadow`` and the tool is executed anyway. Evaluation
    failures, unverifiable ALLOW receipts, unregistered tools, and tool
    exceptions still fail closed.
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
        shadow_mode: bool | None = None,
    ) -> None:
        if not policy_ids and not bundle_ids:
            raise ValueError("provide at least one policy_id or bundle_id")
        self._shadow_mode = resolve_shadow_mode(shadow_mode)
        if self._shadow_mode:
            shadow_logger.warning(
                "ramen-foundry: SHADOW MODE ENABLED for %s; policy BLOCK verdicts "
                "will be logged but NOT enforced",
                type(self).__name__,
            )
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

    @property
    def shadow_mode(self) -> bool:
        """Return whether policy BLOCK verdicts are logged instead of enforced."""
        return self._shadow_mode

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
            if self._shadow_mode:
                return self._shadow_dispatch(invocation, verdict)
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

        return self._dispatch(invocation)

    def _run_tool(self, invocation: ToolInvocation) -> Any:
        """Invoke the registered tool; raise ``_ToolDispatchError`` if it cannot run."""
        tool = self._tools.get(invocation.name)
        if tool is None:
            raise _ToolDispatchError(
                f"Tool '{invocation.name}' is not registered and was not executed."
            )
        try:
            return tool.invoke(invocation.arguments)
        except Exception as error:  # noqa: BLE001
            raise _ToolDispatchError(f"Tool '{invocation.name}' failed: {error}") from error

    def _dispatch(
        self,
        invocation: ToolInvocation,
        metadata: Mapping[str, Any] | None = None,
    ) -> Command:
        """Execute the tool and return its result to ``llm_node``; fail closed on error."""
        try:
            result = self._run_tool(invocation)
        except _ToolDispatchError as error:
            return self._return_error(invocation, str(error))

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=_stringify(result),
                        tool_call_id=invocation.tool_call_id,
                        name=invocation.name,
                        response_metadata=dict(metadata or {}),
                    )
                ],
                "governance_error": None,
                "tool_invocation": None,
            },
            goto=self._llm_node,
        )

    def _shadow_dispatch(self, invocation: ToolInvocation, verdict: Mapping[str, Any]) -> Command:
        """Log a non-enforced BLOCK and execute the tool anyway (shadow mode only)."""
        return self._dispatch(invocation, self._shadow_metadata(invocation, verdict))

    def _shadow_metadata(
        self,
        invocation: ToolInvocation,
        verdict: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Log the shadow-mode violation and return the ``response_metadata`` for it.

        The receipt is reported as verified only if the client verified it and
        its signed bytes independently prove a BLOCK of this exact call. An
        unverified receipt is still logged and still passes through. Tool
        arguments are never logged.
        """
        receipt = extract_receipt(verdict)
        summary = _violation_summary(verdict)
        if not verdict.get("receipt_verified", False):
            problem: str | None = (
                _non_blank(verdict.get("receipt_reason"))
                or _non_blank(verdict.get("receipt_alert"))
                or "client did not verify the receipt"
            )
        else:
            problem = signed_verdict_problem(
                verdict, self._payload(invocation), VERDICT_BLOCK, self._public_keys
            )
        receipt_id = receipt.get("id") if receipt is not None else None
        receipt_kid = receipt.get("kid") if receipt is not None else None
        record: dict[str, Any] = {
            "shadow_mode": True,
            "enforced": False,
            "verdict": "BLOCK",
            "tool_name": invocation.name,
            "tool_call_id": invocation.tool_call_id,
            "rule_id": summary["rule_id"],
            "primary_statutory_anchor": summary["primary_statutory_anchor"],
            "reason": summary["reason"],
            "steering_directive": summary["steering"],
            "receipt_id": receipt_id if isinstance(receipt_id, str) and receipt_id else None,
            "receipt_kid": receipt_kid if isinstance(receipt_kid, str) and receipt_kid else None,
            "receipt_verified": problem is None,
            "receipt_problem": problem,
            "policy_ids": [str(policy_id) for policy_id in verdict.get("policy_ids") or []],
        }
        _log_shadow_violation(record, receipt)
        return {
            "receipt": receipt,
            "receipt_verified": record["receipt_verified"],
            "policy_ids": list(record["policy_ids"]),
            SHADOW_METADATA_KEY: record,
        }

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


def primary_statutory_anchor(anchors: Any) -> str:
    """Return the first non-blank statutory anchor, or the general boundary label."""
    if isinstance(anchors, Sequence) and not isinstance(anchors, str) and anchors:
        first = anchors[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
    return DEFAULT_STATUTORY_ANCHOR


def _primary_violation(data: Mapping[str, Any]) -> Mapping[str, Any]:
    violations = data.get("total_violations")
    if isinstance(violations, list):
        for violation in violations:
            if isinstance(violation, Mapping):
                return violation
    return {}


def _non_blank(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _violation_summary(verdict: Mapping[str, Any]) -> dict[str, str]:
    """Return the primary rule, anchor, reason, and steering directive of a BLOCK."""
    data = verdict.get("data")
    data = data if isinstance(data, Mapping) else {}
    violation = _primary_violation(data)
    return {
        "rule_id": _non_blank(violation.get("rule_id")) or _DEFAULT_RULE_ID,
        "reason": (
            _non_blank(violation.get("reasoning"))
            or _non_blank(violation.get("rule_name"))
            or _DEFAULT_REASON
        ),
        "steering": (
            _non_blank(verdict.get("steering"))
            or _non_blank(violation.get("recovery_instruction"))
            or _DEFAULT_STEERING
        ),
        "primary_statutory_anchor": primary_statutory_anchor(data.get("statutory_anchors")),
    }


def _log_shadow_violation(record: Mapping[str, Any], receipt: Mapping[str, Any] | None) -> None:
    """Emit one WARNING per non-enforced BLOCK, plus the full receipt at DEBUG."""
    signature = (
        "VERIFIED" if record["receipt_verified"] else f"UNVERIFIED: {record['receipt_problem']}"
    )
    shadow_logger.warning(
        "\n".join(
            (
                SHADOW_LOG_HEADER,
                "Receipt ID: %s  (signature: %s)",
                "Steering Directive: %s",
                "Rule Violated: %s (%s)",
                "Reason: %s",
                "Tool: %s  tool_call_id=%s  policy_ids=%s",
                "Action: tool executed anyway (shadow_mode=True)",
            )
        ),
        record["receipt_id"] or "<none>",
        signature,
        record["steering_directive"],
        record["rule_id"],
        record["primary_statutory_anchor"],
        record["reason"],
        record["tool_name"],
        record["tool_call_id"],
        ",".join(record["policy_ids"]) or "<none>",
        extra={SHADOW_METADATA_KEY: dict(record)},
    )
    if receipt is not None and shadow_logger.isEnabledFor(logging.DEBUG):
        shadow_logger.debug(
            "ramen-foundry shadow receipt %s: %s",
            record["receipt_id"] or "<none>",
            json.dumps(receipt, default=str, sort_keys=True),
        )


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)
