"""Closed-loop LangGraph steering node with episodic repair memory."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END
from langgraph.types import Command
from ramen_ai import RamenClient

from .langgraph_nodes import RamenToolNode, ToolInvocation, _stringify
from .memory import BaseEpisodicMemoryStore, CorrectionExemplar

logger = logging.getLogger(__name__)

REFLECTION_HEADER = "[ACTION BLOCKED BY RAMEN ACTION GATE]"
DEFAULT_STATUTORY_ANCHOR = "General Operational Boundary"
_REPLAN_INSTRUCTION = (
    "Replan and re-invoke the tool with parameters satisfying the steering directive."
)
_DEFAULT_RULE_ID = "UNSPECIFIED_RULE"
_DEFAULT_REASON = "Blocked by ramen-ai policy."
_DEFAULT_STEERING = "Revise the tool call so that it satisfies the governing policy."


class RamenSteerNode(RamenToolNode):
    """Gate tool calls pre-dispatch and steer blocked calls back to the planner.

    State contract (all keys are JSON-serialisable for LangGraph checkpointers):

    ``tool_invocation``
        The resolved :class:`ToolInvocation` (or equivalent mapping) to gate.
    ``messages``
        Receives one ``ToolMessage`` per gated call.
    ``repair_turns``
        Count of consecutive blocked attempts already steered back to the planner.
    ``pending_correction``
        Context of the most recent blocked attempt, consumed on a successful repair.
    ``governance_error``
        ``None`` on success, otherwise the reason the call was not dispatched.
    ``task`` (configurable via ``task_key``)
        Task text fingerprinted for episodic memory; required when a store is bound.

    Routing:

    * ALLOWED with a verified receipt: dispatch the tool, attach the full Schema
      V5 receipt to ``ToolMessage.response_metadata["receipt"]``, record a
      :class:`CorrectionExemplar` if this call repaired a prior block, and go to
      ``next_node``.
    * BLOCKED: do not dispatch; emit a structured reflection carrying only the
      primary statutory anchor and go to ``planner_node``. Once
      ``max_repair_turns`` reflections have been issued, go to ``halt_node``.
    * Evaluation failure, unverifiable receipt, unregistered tool, or tool
      exception: fail closed to ``halt_node`` without a repair retry.
    """

    def __init__(
        self,
        *,
        client: RamenClient,
        tools: Mapping[str, BaseTool],
        planner_node: str,
        next_node: str | None = None,
        halt_node: str = END,
        policy_ids: Sequence[str] | None = None,
        bundle_ids: Sequence[str] | None = None,
        provider_key: str | None = None,
        provider_name: str | None = None,
        memory_store: BaseEpisodicMemoryStore | None = None,
        max_repair_turns: int = 2,
        task_key: str = "task",
    ) -> None:
        super().__init__(
            client=client,
            tools=tools,
            llm_node=planner_node,
            policy_ids=policy_ids,
            bundle_ids=bundle_ids,
            provider_key=provider_key,
            provider_name=provider_name,
        )
        if (
            isinstance(max_repair_turns, bool)
            or not isinstance(max_repair_turns, int)
            or max_repair_turns < 0
        ):
            raise ValueError("max_repair_turns must be a non-negative integer")
        if memory_store is not None and not isinstance(memory_store, BaseEpisodicMemoryStore):
            raise TypeError("memory_store must implement BaseEpisodicMemoryStore")
        self._planner_node = planner_node
        self._next_node = next_node or planner_node
        self._halt_node = halt_node
        self._memory_store = memory_store
        self._max_repair_turns = max_repair_turns
        self._task_key = task_key

    @property
    def memory_store(self) -> BaseEpisodicMemoryStore | None:
        """Return the bound episodic memory store, if any."""
        return self._memory_store

    def __call__(self, state: Mapping[str, Any]) -> Command:
        invocation = ToolInvocation.model_validate(state["tool_invocation"])
        repair_turns = state.get("repair_turns") or 0
        if isinstance(repair_turns, bool) or not isinstance(repair_turns, int) or repair_turns < 0:
            raise ValueError("state['repair_turns'] must be a non-negative integer")
        task: str | None = None
        if self._memory_store is not None:
            task = state.get(self._task_key)
            if not isinstance(task, str) or not task.strip():
                raise ValueError(
                    f"state['{self._task_key}'] must be a non-blank string when a "
                    "memory store is bound"
                )

        try:
            verdict = self._evaluate(invocation)
        except Exception as error:  # noqa: BLE001
            return self._halt(
                invocation,
                f"Governance evaluation unavailable; tool was not executed: {error}",
                repair_turns,
            )

        if not verdict.get("allowed", False):
            return self._steer(invocation, verdict, repair_turns)

        if not verdict.get("receipt_verified", False):
            reason = verdict.get("receipt_reason") or "No verifiable governance receipt was returned."
            return self._halt(
                invocation,
                "Tool execution was denied because the governance receipt could not be "
                f"verified: {reason}",
                repair_turns,
            )

        tool = self._tools.get(invocation.name)
        if tool is None:
            return self._halt(
                invocation,
                f"Tool '{invocation.name}' is not registered and was not executed.",
                repair_turns,
            )

        try:
            result = tool.invoke(invocation.arguments)
        except Exception as error:  # noqa: BLE001
            return self._halt(
                invocation,
                f"Tool '{invocation.name}' failed: {error}",
                repair_turns,
            )

        receipt = _extract_receipt(verdict)
        metadata: dict[str, Any] = {
            "receipt": receipt,
            "receipt_verified": True,
            "policy_ids": list(verdict.get("policy_ids") or []),
        }
        pending = state.get("pending_correction")
        if (
            self._memory_store is not None
            and task is not None
            and isinstance(pending, Mapping)
            and pending.get("tool_name") == invocation.name
        ):
            self._record_repair(task, invocation, pending, receipt, metadata)

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=_stringify(result),
                        tool_call_id=invocation.tool_call_id,
                        name=invocation.name,
                        response_metadata=metadata,
                    )
                ],
                "governance_error": None,
                "tool_invocation": None,
                "repair_turns": 0,
                "pending_correction": None,
            },
            goto=self._next_node,
        )

    def _steer(
        self,
        invocation: ToolInvocation,
        verdict: Mapping[str, Any],
        repair_turns: int,
    ) -> Command:
        data = verdict.get("data")
        data = data if isinstance(data, Mapping) else {}
        violation = _primary_violation(data)
        rule_id = _non_blank(violation.get("rule_id")) or _DEFAULT_RULE_ID
        reason = (
            _non_blank(violation.get("reasoning"))
            or _non_blank(violation.get("rule_name"))
            or _DEFAULT_REASON
        )
        steering = (
            _non_blank(verdict.get("steering"))
            or _non_blank(violation.get("recovery_instruction"))
            or _DEFAULT_STEERING
        )
        anchor = primary_statutory_anchor(data.get("statutory_anchors"))
        exhausted = repair_turns >= self._max_repair_turns
        instruction = (
            f"Repair budget exhausted after {repair_turns} of {self._max_repair_turns} "
            "attempts. Do not retry; escalate to a human operator."
            if exhausted
            else _REPLAN_INSTRUCTION
        )
        reflection = format_reflection(
            rule_id=rule_id,
            primary_anchor=anchor,
            reason=reason,
            steering=steering,
            instruction=instruction,
        )
        next_turn = repair_turns if exhausted else repair_turns + 1
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=reflection,
                        tool_call_id=invocation.tool_call_id,
                        name=invocation.name,
                        status="error",
                        response_metadata={
                            "rule_id": rule_id,
                            "primary_statutory_anchor": anchor,
                            "repair_turn": next_turn,
                            "max_repair_turns": self._max_repair_turns,
                            "receipt": _extract_receipt(verdict),
                            "receipt_verified": bool(verdict.get("receipt_verified", False)),
                        },
                    )
                ],
                "governance_error": f"{rule_id} ({anchor}): {reason}",
                "tool_invocation": None,
                "repair_turns": next_turn,
                "pending_correction": {
                    "tool_name": invocation.name,
                    "failed_arguments": _json_safe(invocation.arguments),
                    "rule_id": rule_id,
                    "violation_reason": reason,
                    "primary_statutory_anchor": anchor,
                    "steering_directive": steering,
                },
            },
            goto=self._halt_node if exhausted else self._planner_node,
        )

    def _halt(self, invocation: ToolInvocation, reason: str, repair_turns: int) -> Command:
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
                "repair_turns": repair_turns,
            },
            goto=self._halt_node,
        )

    def _record_repair(
        self,
        task: str,
        invocation: ToolInvocation,
        pending: Mapping[str, Any],
        receipt: Mapping[str, Any] | None,
        metadata: dict[str, Any],
    ) -> None:
        store = self._memory_store
        if store is None:
            return
        receipt_id = receipt.get("id") if isinstance(receipt, Mapping) else None
        try:
            exemplar = CorrectionExemplar.create(
                task=task,
                tool_name=invocation.name,
                failed_arguments=dict(pending.get("failed_arguments") or {}),
                violation_reason=str(pending.get("violation_reason") or _DEFAULT_REASON),
                primary_statutory_anchor=str(
                    pending.get("primary_statutory_anchor") or DEFAULT_STATUTORY_ANCHOR
                ),
                steering_directive=str(pending.get("steering_directive") or _DEFAULT_STEERING),
                repaired_arguments=_json_safe(invocation.arguments),
                receipt_id=receipt_id if isinstance(receipt_id, str) and receipt_id else None,
            )
            store.record_correction(exemplar)
        except Exception as error:  # noqa: BLE001
            # The tool has already been dispatched; surface the failure rather than
            # raising and losing the tool result.
            logger.error("ramen-foundry: failed to record correction exemplar: %s", error)
            metadata["memory_error"] = f"Failed to record correction exemplar: {error}"
            return
        metadata["exemplar_id"] = exemplar.exemplar_id


def primary_statutory_anchor(anchors: Any) -> str:
    """Return the first non-blank statutory anchor, or the general boundary label."""
    if isinstance(anchors, Sequence) and not isinstance(anchors, str) and anchors:
        first = anchors[0]
        if isinstance(first, str) and first.strip():
            return first.strip()
    return DEFAULT_STATUTORY_ANCHOR


def format_reflection(
    *,
    rule_id: str,
    primary_anchor: str,
    reason: str,
    steering: str,
    instruction: str = _REPLAN_INSTRUCTION,
) -> str:
    """Render the structured reflection message returned to the planner."""
    return "\n".join(
        (
            REFLECTION_HEADER,
            f"Rule Violated: {rule_id} ({primary_anchor})",
            f"Reason: {reason}",
            f"Steering Directive: {steering}",
            f"Instruction: {instruction}",
        )
    )


def _primary_violation(data: Mapping[str, Any]) -> Mapping[str, Any]:
    violations = data.get("total_violations")
    if isinstance(violations, list):
        for violation in violations:
            if isinstance(violation, Mapping):
                return violation
    return {}


def _extract_receipt(verdict: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the Schema V5 receipt from an SDK verdict.

    ``RamenClient.evaluate_compliance`` exposes the receipt inside the full
    ``data`` payload; a top-level ``receipt`` key takes precedence if present.
    """
    receipt = verdict.get("receipt")
    if not isinstance(receipt, Mapping):
        data = verdict.get("data")
        receipt = data.get("receipt") if isinstance(data, Mapping) else None
    return dict(receipt) if isinstance(receipt, Mapping) else None


def _non_blank(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _json_safe(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(arguments), default=str, sort_keys=True))
