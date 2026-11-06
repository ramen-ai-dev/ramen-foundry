"""Tests for EligibilityGateAgent and its local deterministic exclusion evaluator."""

from __future__ import annotations

import unittest
from typing import Any

from langchain_core.tools import tool
from pydantic import ValidationError

from ramen_foundry import (
    ELIGIBILITY_GATE_BUNDLE_ID,
    EligibilityCondition,
    EligibilityGateAgent,
    ExclusionRule,
    ExclusionRuleSet,
)
from tests.gate_fixtures import FakeClient


def _rules() -> ExclusionRuleSet:
    return ExclusionRuleSet.model_validate(
        {
            "required_facts": ["applicant_id"],
            "rules": [
                {
                    "rule_id": "R-MINOR",
                    "description": "Applicant under the minimum age",
                    "all_of": [{"field": "age", "op": "lt", "value": 18}],
                },
                {
                    "rule_id": "R-RESTRICTED-REGION",
                    "description": "Residence in a restricted region",
                    "all_of": [{"field": "region", "op": "in", "value": ["XX", "YY"]}],
                },
                {
                    # Multi-variable: low income AND no guarantor AND (poor band OR prior default).
                    "rule_id": "R-HIGH-RISK-UNSECURED",
                    "description": "Low income, no guarantor, and adverse credit history",
                    "all_of": [
                        {"field": "income_usd", "op": "lt", "value": 30000},
                        {"field": "has_guarantor", "op": "eq", "value": False},
                    ],
                    "any_of": [
                        {"field": "credit_band", "op": "in", "value": ["D", "E"]},
                        {"field": "prior_default", "op": "eq", "value": True},
                    ],
                },
            ],
        }
    )


def _facts(**overrides: Any) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "applicant_id": "APP-1",
        "age": 40,
        "region": "ZZ",
        "income_usd": 85_000,
        "has_guarantor": False,
        "credit_band": "B",
        "prior_default": False,
    }
    facts.update(overrides)
    return facts


class ExclusionRuleSetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = _rules()

    def _hit_ids(self, **overrides: Any) -> list[str]:
        return [hit.rule_id for hit in self.rules.evaluate(_facts(**overrides)).hits]

    def test_clean_subject_is_eligible(self) -> None:
        decision = self.rules.evaluate(_facts())
        self.assertTrue(decision.eligible)
        self.assertEqual((decision.hits, decision.problems), ((), ()))

    def test_single_variable_rules(self) -> None:
        self.assertEqual(self._hit_ids(age=17), ["R-MINOR"])
        self.assertEqual(self._hit_ids(age=18), [])
        self.assertEqual(self._hit_ids(region="YY"), ["R-RESTRICTED-REGION"])

    def test_multi_variable_rule_requires_all_of_and_one_any_of(self) -> None:
        risky = {"income_usd": 20_000, "has_guarantor": False}
        self.assertEqual(self._hit_ids(**risky, credit_band="E"), ["R-HIGH-RISK-UNSECURED"])
        self.assertEqual(self._hit_ids(**risky, prior_default=True), ["R-HIGH-RISK-UNSECURED"])
        self.assertEqual(self._hit_ids(**risky), [], "no any_of condition holds")
        self.assertEqual(self._hit_ids(income_usd=20_000, has_guarantor=True, credit_band="E"), [])
        self.assertEqual(self._hit_ids(income_usd=30_000, has_guarantor=False, credit_band="E"), [])

    def test_all_matching_rules_are_reported(self) -> None:
        hits = self._hit_ids(age=10, region="XX")
        self.assertEqual(hits, ["R-MINOR", "R-RESTRICTED-REGION"])

    def test_unresolvable_facts_fail_closed(self) -> None:
        missing = {k: v for k, v in _facts().items() if k != "age"}
        cases: dict[str, Any] = {
            "missing referenced fact": missing,
            "missing required fact": {k: v for k, v in _facts().items() if k != "applicant_id"},
            "none fact": _facts(age=None),
            "nan fact": _facts(income_usd=float("nan")),
            "inf fact": _facts(income_usd=float("inf")),
            "list fact": _facts(region=["XX"]),
            "string for number": _facts(age="40"),
            "number for string": _facts(region=7),
            "bool for number": _facts(age=True),
            "string for bool": _facts(has_guarantor="no"),
            "int for bool": _facts(has_guarantor=0),
            "not a mapping": ["age", 40],
            "none": None,
        }
        for name, facts in cases.items():
            with self.subTest(name):
                decision = self.rules.evaluate(facts)
                self.assertFalse(decision.eligible)
                self.assertTrue(decision.problems)

    def test_an_unresolvable_condition_cannot_be_masked_by_a_failing_one(self) -> None:
        # income is high, so the first all_of condition is False; the rule must
        # still report that credit_band is the wrong type instead of short-circuiting.
        decision = self.rules.evaluate(_facts(income_usd=90_000, credit_band=3))
        self.assertFalse(decision.eligible)
        self.assertIn("R-HIGH-RISK-UNSECURED", decision.problems[0])

    def test_decisions_never_contain_fact_values(self) -> None:
        decision = self.rules.evaluate(_facts(age=17, applicant_id="APP-SECRET-123"))
        self.assertNotIn("APP-SECRET-123", decision.model_dump_json())
        bad = self.rules.evaluate(_facts(age="SECRET-AGE"))
        self.assertNotIn("SECRET-AGE", bad.model_dump_json())

    def test_rule_models_validate_their_inputs(self) -> None:
        bad_conditions = [
            {"field": "age", "op": "lt", "value": "18"},
            {"field": "age", "op": "lt", "value": True},
            {"field": "age", "op": "lt", "value": float("nan")},
            {"field": "region", "op": "in", "value": []},
            {"field": "region", "op": "in", "value": "XX"},
            {"field": "region", "op": "in", "value": ["XX", 1]},
            {"field": "region", "op": "eq", "value": None},
            {"field": "region", "op": "eq", "value": ["XX"]},
            {"field": "region", "op": "matches", "value": "X"},
            {"field": " ", "op": "eq", "value": "X"},
            {"field": "region", "op": "eq"},
        ]
        for condition in bad_conditions:
            with self.subTest(condition=condition), self.assertRaises(ValidationError):
                EligibilityCondition.model_validate(condition)
        with self.assertRaises(ValidationError):
            ExclusionRule.model_validate({"rule_id": "R", "description": "no conditions"})
        with self.assertRaises(ValidationError):
            ExclusionRuleSet.model_validate({"rules": []})
        duplicate = {"rule_id": "R", "description": "d", "all_of": [{"field": "a", "op": "eq", "value": 1}]}
        with self.assertRaises(ValidationError):
            ExclusionRuleSet.model_validate({"rules": [duplicate, duplicate]})
        with self.assertRaises(ValidationError):
            EligibilityCondition.model_validate({"field": "a", "op": "eq", "value": 1, "extra": 1})


class EligibilityGateAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executions: list[dict[str, Any]] = []

        @tool
        def confirm_eligibility(subject_id: str, eligibility_facts: dict[str, Any]) -> str:
            """Confirm eligibility."""

            self.executions.append({"tool": "confirm_eligibility", "subject_id": subject_id})
            return "confirmed"

        @tool
        def bind_contract_terms(subject_id: str, eligibility_facts: dict[str, Any]) -> str:
            """Bind contract terms."""

            self.executions.append({"tool": "bind_contract_terms", "subject_id": subject_id})
            return "bound"

        self.client = FakeClient()
        self.tools = {"confirm_eligibility": confirm_eligibility, "bind_contract_terms": bind_contract_terms}
        self.agent = EligibilityGateAgent(
            client=self.client,  # type: ignore[arg-type]
            tools=self.tools,
            exclusion_rules=_rules(),
        )

    def _call(self, tool_name: str, facts: Any) -> Any:
        return self.agent.execute(tool_name, {"subject_id": "APP-1", "eligibility_facts": facts})

    def test_eligible_subject_reaches_the_bundle_and_runs(self) -> None:
        command = self._call("confirm_eligibility", _facts())

        self.assertIsNone(command.update["governance_error"])
        self.assertEqual(self.executions, [{"tool": "confirm_eligibility", "subject_id": "APP-1"}])
        self.assertEqual(len(self.client.calls), 1)
        _, options = self.client.calls[0]
        self.assertEqual(options["bundle_ids"], [ELIGIBILITY_GATE_BUNDLE_ID])

    def test_excluded_subject_is_blocked_locally_even_though_the_backend_would_allow(self) -> None:
        self.assertTrue(self.client.allowed)

        command = self._call("bind_contract_terms", _facts(age=16, region="XX"))

        reason = command.update["governance_error"]
        self.assertIn("R-MINOR", reason)
        self.assertIn("R-RESTRICTED-REGION", reason)
        self.assertEqual(self.client.calls, [], "exclusions are decided before the backend is contacted")
        self.assertEqual(self.executions, [])

    def test_multi_variable_exclusion_blocks_locally(self) -> None:
        command = self._call(
            "confirm_eligibility",
            _facts(income_usd=12_000, has_guarantor=False, credit_band="D"),
        )

        self.assertIn("R-HIGH-RISK-UNSECURED", command.update["governance_error"])
        self.assertEqual(self.client.calls, [])
        self.assertEqual(self.executions, [])

    def test_missing_malformed_or_absent_facts_fail_closed_before_the_backend(self) -> None:
        arguments: dict[str, Any]
        for name, arguments in {
            "no facts key": {"subject_id": "APP-1"},
            "facts not an object": {"subject_id": "APP-1", "eligibility_facts": "all good"},
            "missing fact": {"subject_id": "APP-1", "eligibility_facts": {"applicant_id": "APP-1"}},
            "wrong type": {"subject_id": "APP-1", "eligibility_facts": _facts(age="forty")},
        }.items():
            with self.subTest(name):
                command = self.agent.execute("confirm_eligibility", arguments)
                self.assertIn("could not be determined locally", command.update["governance_error"])
                self.assertEqual(self.client.calls, [])
                self.assertEqual(self.executions, [])

    def test_block_reason_never_echoes_fact_values(self) -> None:
        command = self._call("confirm_eligibility", _facts(age=17, applicant_id="APP-SECRET-123"))
        self.assertNotIn("APP-SECRET-123", command.update["governance_error"])
        self.assertNotIn("17", command.update["governance_error"])

    def test_clearing_local_rules_does_not_bypass_the_backend_verdict(self) -> None:
        self.client.allowed = False

        command = self._call("confirm_eligibility", _facts())

        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.executions, [])
        self.assertIsNotNone(command.update["governance_error"])

    def test_the_evaluated_payload_carries_the_facts_the_rules_saw(self) -> None:
        self._call("confirm_eligibility", _facts())
        payload, _ = self.client.calls[0]
        self.assertIn('"eligibility_facts"', payload)
        self.assertIn('"credit_band":"B"', payload)

    def test_compiled_graph_applies_the_same_exclusions(self) -> None:
        state = self.agent.invoke(
            {
                "messages": [],
                "tool_invocation": {
                    "name": "confirm_eligibility",
                    "arguments": {"subject_id": "APP-1", "eligibility_facts": _facts(age=5)},
                    "tool_call_id": "g1",
                },
            }
        )
        self.assertIn("R-MINOR", state["governance_error"])
        self.assertEqual(self.client.calls, [])

    def test_check_eligibility_runs_without_the_backend(self) -> None:
        self.assertFalse(self.agent.check_eligibility(_facts(age=1)).eligible)
        self.assertTrue(self.agent.check_eligibility(_facts()).eligible)
        self.assertEqual(self.client.calls, [])

    def test_custom_facts_key(self) -> None:
        agent = EligibilityGateAgent(
            client=self.client,  # type: ignore[arg-type]
            tools=self.tools,
            exclusion_rules=_rules(),
            facts_key="facts",
        )
        command = agent.execute("confirm_eligibility", {"subject_id": "APP-1", "facts": _facts(age=3)})
        self.assertIn("R-MINOR", command.update["governance_error"])

    def test_constructor_validation(self) -> None:
        with self.assertRaises(TypeError):
            EligibilityGateAgent(client=self.client, tools=self.tools, exclusion_rules={"rules": []})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            EligibilityGateAgent(
                client=self.client, tools=self.tools, exclusion_rules=_rules(), facts_key=" "  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            EligibilityGateAgent(
                client=self.client,  # type: ignore[arg-type]
                tools={"dispatch_wire": self.tools["confirm_eligibility"]},
                exclusion_rules=_rules(),
            )


if __name__ == "__main__":
    unittest.main()
