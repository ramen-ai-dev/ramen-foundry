"""Governed contractual and eligibility decisions with local deterministic exclusions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ramen_ai import RamenClient

from ramen_foundry.core import ToolInvocation

from ._base import GovernedToolAgent

ELIGIBILITY_GATE_BUNDLE_ID = "ramen__contractual_eligibility_invariance"
ELIGIBILITY_GATE_TOOL_NAMES = frozenset({"confirm_eligibility", "bind_contract_terms"})
DEFAULT_FACTS_KEY = "eligibility_facts"

ComparisonOp = Literal["eq", "ne", "lt", "lte", "gt", "gte", "in", "not_in"]
_ORDERING_OPS = frozenset({"lt", "lte", "gt", "gte"})
_MEMBERSHIP_OPS = frozenset({"in", "not_in"})

_NonBlank = Annotated[str, Field(pattern=r"\S")]


def _kind(value: Any) -> str | None:
    """Classify a scalar fact. ``None`` means the value cannot be compared safely."""

    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number" if math.isfinite(value) else None
    if isinstance(value, str):
        return "string"
    return None


class _Indeterminate(Exception):
    """A condition could not be evaluated; the caller must fail closed."""


class EligibilityCondition(BaseModel):
    """One comparison of a named fact against a literal.

    ``eq``/``ne`` compare scalars of the same kind (string, number or boolean).
    ``lt``/``lte``/``gt``/``gte`` compare finite numbers only. ``in``/``not_in``
    take a non-empty list of scalars that all share one kind.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    field: _NonBlank
    op: ComparisonOp
    value: Any

    @model_validator(mode="after")
    def _check_value(self) -> EligibilityCondition:
        if self.op in _MEMBERSHIP_OPS:
            if not isinstance(self.value, (list, tuple)) or not self.value:
                raise ValueError(f"{self.op} requires a non-empty list value")
            kinds = {_kind(item) for item in self.value}
            if len(kinds) != 1 or None in kinds:
                raise ValueError(f"{self.op} requires finite scalars of a single kind")
        elif self.op in _ORDERING_OPS:
            if _kind(self.value) != "number":
                raise ValueError(f"{self.op} requires a finite numeric value")
        elif _kind(self.value) is None:
            raise ValueError(f"{self.op} requires a finite scalar value")
        return self

    def holds(self, facts: Mapping[str, Any]) -> bool:
        """Evaluate against ``facts``; raise ``_Indeterminate`` if that is not possible."""

        fact = facts.get(self.field)
        fact_kind = _kind(fact)
        if fact_kind is None:
            raise _Indeterminate(f"fact {self.field!r} is missing or not a comparable scalar")
        if self.op in _MEMBERSHIP_OPS:
            if fact_kind != _kind(self.value[0]):
                raise _Indeterminate(f"fact {self.field!r} has the wrong type for {self.op}")
            member = any(fact == item for item in self.value)
            return member if self.op == "in" else not member
        if fact_kind != _kind(self.value):
            raise _Indeterminate(f"fact {self.field!r} has the wrong type for {self.op}")
        if self.op == "eq":
            return bool(fact == self.value)
        if self.op == "ne":
            return bool(fact != self.value)
        if self.op == "lt":
            return bool(fact < self.value)
        if self.op == "lte":
            return bool(fact <= self.value)
        if self.op == "gt":
            return bool(fact > self.value)
        return bool(fact >= self.value)


class ExclusionRule(BaseModel):
    """Excludes a subject when every ``all_of`` condition holds and, if ``any_of``
    is present, at least one ``any_of`` condition also holds.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: _NonBlank
    description: _NonBlank
    all_of: tuple[EligibilityCondition, ...] = ()
    any_of: tuple[EligibilityCondition, ...] = ()

    @model_validator(mode="after")
    def _require_conditions(self) -> ExclusionRule:
        if not self.all_of and not self.any_of:
            raise ValueError("a rule needs at least one condition")
        return self

    @property
    def fields(self) -> frozenset[str]:
        return frozenset(c.field for c in (*self.all_of, *self.any_of))

    def excludes(self, facts: Mapping[str, Any]) -> bool:
        # Every condition is evaluated (no short-circuit) so one unresolvable
        # condition cannot be masked by another condition that happens to fail.
        all_results = [condition.holds(facts) for condition in self.all_of]
        any_results = [condition.holds(facts) for condition in self.any_of]
        return all(all_results) and (not any_results or any(any_results))


class ExclusionHit(BaseModel):
    """A rule that excluded the subject."""

    model_config = ConfigDict(frozen=True)

    rule_id: str
    description: str


class EligibilityDecision(BaseModel):
    """Result of the local evaluation. ``eligible`` is true only with no hit and no problem."""

    model_config = ConfigDict(frozen=True)

    eligible: bool
    hits: tuple[ExclusionHit, ...] = ()
    problems: tuple[str, ...] = ()


class ExclusionRuleSet(BaseModel):
    """A set of deterministic exclusion rules plus facts that must always be present."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rules: tuple[ExclusionRule, ...] = Field(min_length=1)
    required_facts: tuple[_NonBlank, ...] = ()

    @model_validator(mode="after")
    def _unique_rule_ids(self) -> ExclusionRuleSet:
        ids = [rule.rule_id for rule in self.rules]
        if len(set(ids)) != len(ids):
            raise ValueError("rule_id values must be unique")
        return self

    @property
    def referenced_facts(self) -> frozenset[str]:
        referenced = set(self.required_facts)
        for rule in self.rules:
            referenced |= rule.fields
        return frozenset(referenced)

    def evaluate(self, facts: Any) -> EligibilityDecision:
        """Evaluate every rule. Any fact that is absent or unusable fails closed.

        A fact that is missing, ``None``, non-scalar, non-finite, or of the wrong
        type for a comparison makes the subject ineligible. The result never
        contains fact values.
        """

        if not isinstance(facts, Mapping):
            return EligibilityDecision(
                eligible=False, problems=("eligibility facts must be an object",)
            )
        unusable = sorted(name for name in self.referenced_facts if _kind(facts.get(name)) is None)
        if unusable:
            return EligibilityDecision(
                eligible=False,
                problems=(f"required eligibility facts are missing or unusable: {', '.join(unusable)}",),
            )
        hits: list[ExclusionHit] = []
        problems: list[str] = []
        for rule in self.rules:
            try:
                if rule.excludes(facts):
                    hits.append(ExclusionHit(rule_id=rule.rule_id, description=rule.description))
            except _Indeterminate as error:
                problems.append(f"rule {rule.rule_id!r} could not be evaluated: {error}")
        return EligibilityDecision(
            eligible=not hits and not problems,
            hits=tuple(hits),
            problems=tuple(problems),
        )


class EligibilityGateAgent(GovernedToolAgent):
    """Gate contract and eligibility actions behind local deterministic exclusions.

    ``confirm_eligibility`` and ``bind_contract_terms`` must carry a mapping of
    resolved facts under ``eligibility_facts`` (configurable with ``facts_key``).
    Before the ramen-ai backend is contacted, the facts are evaluated against the
    host-supplied :class:`ExclusionRuleSet`. The call is blocked locally, with no
    evaluation request and no tool execution, when any rule excludes the subject,
    when a referenced fact is missing or unusable, or when a rule cannot be
    evaluated. A subject that clears every rule is still subject to the bundle
    verdict and its Ed25519 receipt.

    The facts are part of the evaluated payload, so the signed receipt binds the
    exact facts the local rules saw. Block reasons name rules and fact fields but
    never echo fact values. Upstream data quality, protected-attribute controls
    and rule authorship remain with the host.
    """

    bundle_id = ELIGIBILITY_GATE_BUNDLE_ID
    allowed_tool_names = ELIGIBILITY_GATE_TOOL_NAMES
    node_name = "governed_eligibility_action"

    def __init__(
        self,
        *,
        client: RamenClient,
        tools: Mapping[str, BaseTool],
        exclusion_rules: ExclusionRuleSet,
        facts_key: str = DEFAULT_FACTS_KEY,
        provider_key: str | None = None,
        provider_name: str | None = None,
    ) -> None:
        if not isinstance(exclusion_rules, ExclusionRuleSet):
            raise TypeError("exclusion_rules must be an ExclusionRuleSet")
        if not isinstance(facts_key, str) or not facts_key.strip():
            raise ValueError("facts_key must be a non-blank string")
        self._rules = exclusion_rules
        self._facts_key = facts_key
        super().__init__(
            client=client,
            tools=tools,
            provider_key=provider_key,
            provider_name=provider_name,
        )

    @property
    def exclusion_rules(self) -> ExclusionRuleSet:
        """Return the rule set evaluated before every backend call."""

        return self._rules

    def check_eligibility(self, facts: Mapping[str, Any]) -> EligibilityDecision:
        """Run only the local evaluation, without contacting the backend."""

        return self._rules.evaluate(facts)

    def _preflight(self, invocation: ToolInvocation) -> str | None:
        decision = self._rules.evaluate(invocation.arguments.get(self._facts_key))
        if decision.eligible:
            return None
        if decision.hits:
            rules = "; ".join(f"{hit.rule_id} ({hit.description})" for hit in decision.hits)
            return f"Excluded by local eligibility rule(s): {rules}. The action was not evaluated."
        return (
            "Eligibility could not be determined locally (fail closed): "
            f"{' | '.join(decision.problems)}"
        )
