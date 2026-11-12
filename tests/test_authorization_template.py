"""Tests for AuthorizationGateAgent: context-complete evidence and local pre-checks."""

from __future__ import annotations

import copy
import json
import unittest
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ramen_foundry import (
    AUTHORIZATION_GATE_BUNDLE_ID,
    AuthorizationGateAgent,
    Ed25519TokenVerifier,
)
from tests.gate_fixtures import FakeClient, mint_token, public_key_hex, stub_tool

NOW = 1_800_000_000.0


def _evidence(**overrides: Any) -> dict[str, Any]:
    """A context-complete, compliant request: $9k, no dual control needed."""

    evidence: dict[str, Any] = {
        "request_id": "REQ-100",
        "requested_by": "emp-1",
        "amount_usd": 9_000.0,
        "payee": "Tenbury Scientific",
        "payee_account": "ACCT-TEN-4401",
        "cost_center": "Lab Operations",
        "fiscal_period": "FY2026-Q4",
        "spend_category": "supplies",
        "platform_limits": {
            "source": "finance platform",
            "per_order_usd": 30_000,
            "per_vendor_cumulative_usd": 90_000,
            "dual_approval_threshold_usd": 50_000,
        },
        "erp_clearance": {
            "id": "BC-7701",
            "status": "OPEN",
            "cost_center": "Lab Operations",
            "fiscal_period": "FY2026-Q4",
            "spend_category": "supplies",
            "balance_usd": 14_000,
            "confirmed_by": "ERP system of record",
        },
        "vendor_exposure": {
            "source": "finance platform",
            "vendor_group": "Tenbury Scientific",
            "open_pending_paid_usd": 21_000,
        },
        "approvals": [],
    }
    evidence.update(overrides)
    return evidence


def _approval(approver: str, amount: float, **overrides: Any) -> dict[str, Any]:
    approval: dict[str, Any] = {
        "approver_id": approver,
        "role": "Finance Controller",
        "approval_token": f"token-{approver}",
        "approved_out_of_band": True,
        "bound_to": {"amount_usd": amount, "payee": "Tenbury Scientific", "payee_account": "ACCT-TEN-4401"},
    }
    approval.update(overrides)
    return approval


def _dual(amount: float = 40_000.0, *approvers: str, threshold: float = 25_000, **extra: Any) -> dict[str, Any]:
    """A request large enough to need dual control, with ample limits and balance."""

    approvers = approvers or ("a1", "a2")
    evidence = _evidence(
        amount_usd=amount,
        approvals=[_approval(a, amount) for a in approvers],
    )
    evidence["platform_limits"] = {**evidence["platform_limits"], "per_order_usd": 100_000,
                                   "per_vendor_cumulative_usd": 500_000, "dual_approval_threshold_usd": threshold}
    evidence["erp_clearance"] = {**evidence["erp_clearance"], "balance_usd": 80_000}
    evidence.update(extra)
    return evidence


class AuthorizationGateAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executions: list[dict[str, Any]] = []
        self.tools = {n: stub_tool(n, self.executions) for n in ("approve_expenditure", "release_payment")}
        self.client = FakeClient()

    def _agent(self, **kwargs: Any) -> AuthorizationGateAgent:
        return AuthorizationGateAgent(client=self.client, tools=self.tools, **kwargs)  # type: ignore[arg-type]

    def _blocked_locally(self, command: Any) -> str:
        self.assertEqual(self.client.calls, [], "the backend must not be contacted")
        self.assertEqual(self.executions, [])
        reason = command.update["governance_error"]
        self.assertIsNotNone(reason)
        return str(reason)

    def test_context_complete_request_reaches_the_bundle_and_runs(self) -> None:
        command = self._agent().execute("approve_expenditure", _evidence())

        self.assertIsNone(command.update["governance_error"])
        self.assertEqual(len(self.executions), 1)
        _, options = self.client.calls[0]
        self.assertEqual(options["bundle_ids"], [AUTHORIZATION_GATE_BUNDLE_ID])

    def test_extra_keys_reach_the_bundle_unchanged(self) -> None:
        payload = _evidence(business_approval="Lab manager approved")
        payload["erp_clearance"]["note"] = "extra nested key"
        self._agent().execute("approve_expenditure", payload)

        evaluated = json.loads(self.client.calls[0][0])["arguments"]
        self.assertEqual(evaluated["business_approval"], "Lab manager approved")
        self.assertEqual(evaluated["erp_clearance"]["note"], "extra nested key")

    def test_starved_payloads_are_blocked_locally_without_echoing_values(self) -> None:
        secret = "SECRET-APPROVAL-TOKEN"
        full = _evidence()
        cases: dict[str, dict[str, Any]] = {
            "old native shape": {
                "request_id": "REQ-1", "requested_by": "e", "amount_usd": 5.0, "budget_id": "B",
                "budget_remaining_usd": 10.0, "budget_clearance_token": secret, "approvals": [],
            },
            "no platform limits": {k: v for k, v in full.items() if k != "platform_limits"},
            "no erp clearance": {k: v for k, v in full.items() if k != "erp_clearance"},
            "no vendor exposure": {k: v for k, v in full.items() if k != "vendor_exposure"},
            "no payee": {k: v for k, v in full.items() if k != "payee"},
            "no payee account": {k: v for k, v in full.items() if k != "payee_account"},
            "no request cost center": {k: v for k, v in full.items() if k != "cost_center"},
            "no request fiscal period": {k: v for k, v in full.items() if k != "fiscal_period"},
            "no request spend category": {k: v for k, v in full.items() if k != "spend_category"},
            "bool amount": _evidence(amount_usd=True),
            "nan amount": _evidence(amount_usd=float("nan")),
            "negative exposure": _evidence(vendor_exposure={**full["vendor_exposure"], "open_pending_paid_usd": -1}),
            "approval without binding": _evidence(approvals=[
                {"approver_id": "a1", "role": "CFO", "approval_token": secret, "approved_out_of_band": True}]),
        }
        agent = self._agent()
        for name, payload in cases.items():
            with self.subTest(name):
                reason = self._blocked_locally(agent.execute("release_payment", payload))
                self.assertIn("incomplete or malformed", reason)
                self.assertNotIn(secret, reason)

    def test_limit_balance_status_and_exposure_checks(self) -> None:
        agent = self._agent()
        full = _evidence()
        cases = {
            "per-order limit": (_evidence(amount_usd=30_000.01, **{}) | {
                "erp_clearance": {**full["erp_clearance"], "balance_usd": 99_000}}, "per-order limit"),
            "clearance not open": (_evidence(erp_clearance={**full["erp_clearance"], "status": "CLOSED"}), "not open"),
            "balance": (_evidence(amount_usd=14_000.01), "clearance balance"),
            "vendor cap": (_evidence(vendor_exposure={**full["vendor_exposure"], "open_pending_paid_usd": 85_000}), "cumulative vendor exposure"),
        }
        for name, (payload, fragment) in cases.items():
            with self.subTest(name):
                self.assertIn(fragment, self._blocked_locally(agent.execute("release_payment", payload)))

    def test_clearance_must_match_the_requests_cost_center_period_and_category(self) -> None:
        agent = self._agent()
        for field, other in (("cost_center", "Marketing"), ("fiscal_period", "FY2027-Q1"), ("spend_category", "travel and lodging")):
            with self.subTest(field):
                reason = self._blocked_locally(agent.execute("approve_expenditure", _evidence(**{field: other})))
                self.assertIn("does not match the request's cost center", reason)

    def test_dimension_comparison_ignores_case_and_surrounding_space(self) -> None:
        payload = _evidence(cost_center="  lab operations ", spend_category="SUPPLIES")
        self.assertIsNone(self._agent().execute("approve_expenditure", payload).update["governance_error"])

    def test_exact_limit_and_balance_are_allowed(self) -> None:
        full = _evidence()
        payload = _evidence(
            amount_usd=14_000.0,
            vendor_exposure={**full["vendor_exposure"], "open_pending_paid_usd": 76_000},
        )
        self.assertIsNone(self._agent().execute("release_payment", payload).update["governance_error"])

    def test_threshold_defaults_to_the_platform_limit_not_a_foundry_constant(self) -> None:
        """Regression: a $12k spend under a $50k platform threshold needs no dual control."""

        payload = _evidence(amount_usd=12_000.0)
        command = self._agent().execute("approve_expenditure", payload)
        self.assertIsNone(command.update["governance_error"])
        self.assertEqual(len(self.client.calls), 1)

    def test_dual_control_at_the_platform_threshold(self) -> None:
        agent = self._agent()
        one = agent.execute("release_payment", _dual(25_000.0, "a1", threshold=25_000))
        self.assertIn("requires 2 distinct approvers", self._blocked_locally(one))

    def test_dual_control_with_two_independent_bound_approvers_proceeds(self) -> None:
        command = self._agent().execute("release_payment", _dual())
        self.assertIsNone(command.update["governance_error"])
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(len(self.executions), 1)

    def test_duplicate_approver_and_self_approval_are_blocked(self) -> None:
        agent = self._agent()
        duplicate = agent.execute("release_payment", _dual(40_000.0, "a1", "a1"))
        self.assertIn("distinct approvers", self._blocked_locally(duplicate))
        self_approved = agent.execute("release_payment", _dual(40_000.0, "emp-1", "a2"))
        self.assertIn("independent of the requester", self._blocked_locally(self_approved))

    def test_approvals_must_be_out_of_band_and_bound_to_the_payment(self) -> None:
        agent = self._agent()
        base = _dual()
        cases = {
            "in band": ({**base, "approvals": [_approval("a1", 40_000.0, approved_out_of_band=False), _approval("a2", 40_000.0)]}, "out of band"),
            "other amount": ({**base, "approvals": [_approval("a1", 39_000.0), _approval("a2", 40_000.0)]}, "not bound"),
            "other payee": ({**base, "approvals": [_approval("a1", 40_000.0, bound_to={"amount_usd": 40_000.0, "payee": "Someone Else", "payee_account": "ACCT-TEN-4401"}), _approval("a2", 40_000.0)]}, "not bound"),
            "other account": ({**base, "approvals": [_approval("a1", 40_000.0, bound_to={"amount_usd": 40_000.0, "payee": "Tenbury Scientific", "payee_account": "ACCT-OTHER"}), _approval("a2", 40_000.0)]}, "not bound"),
        }
        for name, (payload, fragment) in cases.items():
            with self.subTest(name):
                self.assertIn(fragment, self._blocked_locally(agent.execute("release_payment", payload)))

    def test_misbound_approval_is_rejected_even_below_the_threshold(self) -> None:
        payload = _evidence(approvals=[_approval("a1", 1.0)])
        self.assertIn("not bound", self._blocked_locally(self._agent().execute("release_payment", payload)))

    def test_local_settings_can_only_tighten_the_platform_values(self) -> None:
        stricter = self._agent(dual_control_threshold_usd=5_000.0)
        self.assertIn("requires 2 distinct approvers", self._blocked_locally(
            stricter.execute("release_payment", _evidence(amount_usd=9_000.0))))

        weaker = self._agent(dual_control_threshold_usd=1e9)
        self.assertIn("requires 2 distinct approvers", self._blocked_locally(
            weaker.execute("release_payment", _dual(40_000.0, "a1"))))

        three = self._agent(required_approvers=3)
        self.assertIn("requires 3 distinct approvers", self._blocked_locally(
            three.execute("release_payment", _dual())))

        platform_three = _dual()
        platform_three["platform_limits"]["required_approvers"] = 3
        self.assertIn("requires 3 distinct approvers", self._blocked_locally(
            self._agent(required_approvers=1).execute("release_payment", platform_three)))

    def test_backend_block_still_prevents_execution(self) -> None:
        self.client.allowed = False
        command = self._agent().execute("approve_expenditure", _evidence())

        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.executions, [])
        self.assertIsNotNone(command.update["governance_error"])

    def test_tokens_are_verified_locally_when_verifiers_are_configured(self) -> None:
        budget_key, approver_key = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
        agent = self._agent(
            budget_verifier=Ed25519TokenVerifier({"k1": public_key_hex(budget_key)}, clock=lambda: NOW),
            approval_verifier=Ed25519TokenVerifier({"k1": public_key_hex(approver_key)}, clock=lambda: NOW),
        )

        def payload(**clearance_overrides: Any) -> dict[str, Any]:
            evidence = _dual()
            claims = {"request_id": "REQ-100", "amount_usd": 40_000.0, "exp": NOW + 60}
            evidence["erp_clearance"]["confirmation_token"] = mint_token(
                budget_key, {**claims, "clearance_id": "BC-7701", **clearance_overrides}
            )
            for approval in evidence["approvals"]:
                approval["approval_token"] = mint_token(
                    approver_key, {**claims, "approver_id": approval["approver_id"]}
                )
            return evidence

        good = agent.execute("release_payment", payload())
        self.assertIsNone(good.update["governance_error"])
        self.assertEqual(len(self.client.calls), 1)

        self.client.calls.clear()
        self.executions.clear()
        wrong_amount = agent.execute("release_payment", payload(amount_usd=1.0))
        self.assertIn("Budget clearance token rejected", self._blocked_locally(wrong_amount))

        forged = payload()
        forged["approvals"][0]["approval_token"] = mint_token(
            budget_key, {"request_id": "REQ-100", "amount_usd": 40_000.0, "approver_id": "a1", "exp": NOW + 60}
        )
        self.assertIn("Approval token for approver 'a1' rejected", self._blocked_locally(
            agent.execute("release_payment", forged)))

        missing = _dual()
        self.assertIn("Budget clearance token rejected", self._blocked_locally(
            agent.execute("release_payment", missing)))

    def test_compiled_graph_applies_the_same_local_checks(self) -> None:
        state = self._agent().invoke(
            {
                "messages": [],
                "tool_invocation": {
                    "name": "release_payment",
                    "arguments": _evidence(amount_usd=99_999.0),
                    "tool_call_id": "g1",
                },
            }
        )
        self.assertIn("per-order limit", state["governance_error"])
        self.assertEqual(self.client.calls, [])

    def test_caller_arguments_are_not_mutated(self) -> None:
        payload = _dual()
        snapshot = copy.deepcopy(payload)
        self._agent().execute("release_payment", payload)
        self.assertEqual(payload, snapshot)

    def test_constructor_validation(self) -> None:
        for kwargs in (
            {"dual_control_threshold_usd": -1},
            {"dual_control_threshold_usd": float("inf")},
            {"dual_control_threshold_usd": True},
            {"required_approvers": 0},
            {"required_approvers": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self._agent(**kwargs)
        with self.assertRaises(TypeError):
            self._agent(budget_verifier=object())

    def test_only_declared_tools_may_be_registered(self) -> None:
        with self.assertRaises(ValueError):
            AuthorizationGateAgent(client=self.client, tools={"dispatch_wire": self.tools["release_payment"]})  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
