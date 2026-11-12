"""Governed contractual and eligibility decisions with local deterministic exclusions."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import date
from typing import Annotated, Any, Literal

from langchain_core.tools import BaseTool
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)
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
_Money = Annotated[float, Field(ge=0, allow_inf_nan=False)]
_PositiveMoney = Annotated[float, Field(gt=0, allow_inf_nan=False)]
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _check_iso_date(value: str) -> str:
    if not _ISO_DATE.fullmatch(value):
        raise ValueError("must be an ISO date (YYYY-MM-DD)")
    date.fromisoformat(value)
    return value


_IsoDate = Annotated[str, AfterValidator(_check_iso_date)]


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


class _Context(BaseModel):
    """Claim-evidence object. Unknown keys are kept and sent to the bundle unchanged."""

    model_config = ConfigDict(extra="allow", strict=True)


class PolicySystemRecord(_Context):
    """Term, status and loss date as confirmed by the policy system of record."""

    effective: _IsoDate
    expiry: _IsoDate
    status: _NonBlank
    date_of_loss: _IsoDate
    confirmed_by: _NonBlank
    written_endorsement: _NonBlank | None = None


class InspectionEvidence(_Context):
    """Inspection token issued out of band and verified by the platform."""

    token: _NonBlank
    issued_out_of_band_by: _NonBlank
    verified_by_platform: bool
    inspection_date: _IsoDate
    scope: _NonBlank
    assessed_usd: _PositiveMoney


class PlatformSchedule(_Context):
    """Schedule, sub-limit and aggregate delivered by the policy platform."""

    source: _NonBlank
    unit_block_limit_usd: _PositiveMoney
    remaining_aggregate_usd: _Money
    deductible_usd: _Money
    drawn_usd: _Money


class ClaimEvidence(_Context):
    """The platform-confirmed context that must sit inside ``eligibility_facts``."""

    policy_system_record: PolicySystemRecord
    inspection_evidence: InspectionEvidence
    platform_schedule: PlatformSchedule
    verified_file_findings: _NonBlank


class ClaimRequest(_Context):
    """The action-level arguments of ``confirm_eligibility`` and ``bind_contract_terms``."""

    subject_id: _NonBlank
    approved_amount_usd: _PositiveMoney


def _claim_evidence_problem(request: ClaimRequest, evidence: ClaimEvidence) -> str | None:
    record, inspection, schedule = (
        evidence.policy_system_record,
        evidence.inspection_evidence,
        evidence.platform_schedule,
    )
    effective, expiry = date.fromisoformat(record.effective), date.fromisoformat(record.expiry)
    loss = date.fromisoformat(record.date_of_loss)
    if expiry < effective:
        return "Policy term expires before it takes effect."
    if record.status.strip().upper() != "ACTIVE":
        return "The policy system does not show the policy as active."
    if not effective <= loss <= expiry and record.written_endorsement is None:
        return "The date of loss is outside the confirmed policy term and no written endorsement covers it."
    if inspection.verified_by_platform is not True:
        return "The inspection token has not been verified by the platform."
    if date.fromisoformat(inspection.inspection_date) < loss:
        return "The inspection predates the loss, so it cannot support this claim."
    amount = request.approved_amount_usd
    if amount > inspection.assessed_usd:
        return "The approved amount exceeds the amount assessed by the verified inspection."
    if amount > schedule.unit_block_limit_usd or amount > schedule.remaining_aggregate_usd:
        return "The approved amount exceeds the scheduled limit or the remaining aggregate."
    return None


class EligibilityGateAgent(GovernedToolAgent):
    """Gate contract and eligibility actions behind local deterministic checks.

    ``confirm_eligibility`` and ``bind_contract_terms`` carry ``subject_id``,
    ``approved_amount_usd`` and a mapping of resolved facts under
    ``eligibility_facts`` (configurable with ``facts_key``). The facts hold the
    flat values your :class:`ExclusionRuleSet` reads plus the claim evidence in
    :class:`ClaimEvidence`: ``policy_system_record``, ``inspection_evidence``,
    ``platform_schedule`` and ``verified_file_findings``.

    Before the ramen-ai backend is contacted, the call is blocked locally, with no
    evaluation request and no tool execution, when:

    * any host exclusion rule excludes the subject, a referenced fact is missing
      or unusable, or a rule cannot be evaluated;
    * the claim evidence is missing or malformed;
    * the policy is not active, the loss is outside the confirmed term with no
      written endorsement, the inspection is unverified or predates the loss, or
      the approved amount exceeds the assessed amount, the scheduled limit or the
      remaining aggregate.

    A subject that clears every local check is still subject to the bundle verdict
    and its Ed25519 receipt. The facts are part of the evaluated payload, so the
    signed receipt binds exactly what the local checks saw. Block reasons name
    rules and fields but never echo fact values. Upstream data quality,
    protected-attribute controls and rule authorship remain with the host.
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
        facts = invocation.arguments.get(self._facts_key)
        decision = self._rules.evaluate(facts)
        if decision.hits:
            rules = "; ".join(f"{hit.rule_id} ({hit.description})" for hit in decision.hits)
            return f"Excluded by local eligibility rule(s): {rules}. The action was not evaluated."
        if not decision.eligible:
            return (
                "Eligibility could not be determined locally (fail closed): "
                f"{' | '.join(decision.problems)}"
            )
        try:
            request = ClaimRequest.model_validate(invocation.arguments)
            evidence = ClaimEvidence.model_validate(facts)
        except ValidationError as error:
            fields = sorted({".".join(str(part) for part in item["loc"]) for item in error.errors()})
            return (
                "Claim evidence is incomplete or malformed "
                f"(fields: {', '.join(fields)}); supply the full platform-confirmed claim context."
            )
        return _claim_evidence_problem(request, evidence)
