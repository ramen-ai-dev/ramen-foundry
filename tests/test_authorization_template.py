"""Tests for AuthorizationGateAgent: budget clearance and dual control."""

from __future__ import annotations

import unittest
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from langchain_core.tools import tool

from ramen_foundry import (
    AUTHORIZATION_GATE_BUNDLE_ID,
    AuthorizationGateAgent,
    Ed25519TokenVerifier,
)
from tests.gate_fixtures import FakeClient, mint_token, public_key_hex

NOW = 1_800_000_000.0


def _evidence(**overrides: Any) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "request_id": "REQ-100",
        "requested_by": "emp-1",
        "amount_usd": 2_500.0,
        "budget_id": "BUD-7",
        "budget_remaining_usd": 100_000.0,
        "budget_clearance_token": "opaque-clearance",
        "approvals": [],
    }
    evidence.update(overrides)
    return evidence


def _approvals(*ids: str) -> list[dict[str, str]]:
    return [{"approver_id": approver, "approval_token": f"token-{approver}"} for approver in ids]


class AuthorizationGateAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executions: list[tuple[str, dict[str, Any]]] = []

        def record(name: str, **arguments: Any) -> str:
            self.executions.append((name, arguments))
            return "ok"

        @tool
        def approve_expenditure(
            request_id: str,
            requested_by: str,
            amount_usd: float,
            budget_id: str,
            budget_remaining_usd: float,
            budget_clearance_token: str,
            approvals: list[dict[str, str]],
        ) -> str:
            """Commit a budgeted expenditure."""

            return record("approve_expenditure", request_id=request_id, amount_usd=amount_usd)

        @tool
        def release_payment(
            request_id: str,
            requested_by: str,
            amount_usd: float,
            budget_id: str,
            budget_remaining_usd: float,
            budget_clearance_token: str,
            approvals: list[dict[str, str]],
        ) -> str:
            """Release a payment."""

            return record("release_payment", request_id=request_id, amount_usd=amount_usd)

        self.tools = {"approve_expenditure": approve_expenditure, "release_payment": release_payment}
        self.client = FakeClient()

    def _agent(self, **kwargs: Any) -> AuthorizationGateAgent:
        return AuthorizationGateAgent(client=self.client, tools=self.tools, **kwargs)  # type: ignore[arg-type]

    def _blocked_locally(self, command: Any) -> str:
        self.assertEqual(self.client.calls, [], "the backend must not be contacted")
        self.assertEqual(self.executions, [])
        reason = command.update["governance_error"]
        self.assertIsNotNone(reason)
        return str(reason)

    def test_in_budget_request_below_threshold_reaches_the_bundle_and_runs(self) -> None:
        command = self._agent().execute("approve_expenditure", _evidence())

        self.assertIsNone(command.update["governance_error"])
        self.assertEqual(self.executions, [("approve_expenditure", {"request_id": "REQ-100", "amount_usd": 2500.0})])
        _, options = self.client.calls[0]
        self.assertEqual(options["bundle_ids"], [AUTHORIZATION_GATE_BUNDLE_ID])

    def test_amount_over_remaining_budget_is_blocked_locally(self) -> None:
        command = self._agent().execute(
            "release_payment", _evidence(amount_usd=10_000.01, budget_remaining_usd=10_000.0)
        )
        self.assertIn("exceeds the remaining cleared budget", self._blocked_locally(command))

    def test_amount_equal_to_remaining_budget_is_not_over_budget(self) -> None:
        agent = self._agent(dual_control_threshold_usd=1e9)
        command = agent.execute("release_payment", _evidence(amount_usd=500.0, budget_remaining_usd=500.0))
        self.assertIsNone(command.update["governance_error"])

    def test_dual_control_applies_at_the_threshold(self) -> None:
        agent = self._agent()
        one = agent.execute("release_payment", _evidence(amount_usd=10_000.0, approvals=_approvals("a1")))
        self.assertIn("requires 2 distinct approvers", self._blocked_locally(one))

    def test_dual_control_with_two_distinct_independent_approvers_proceeds(self) -> None:
        command = self._agent().execute(
            "release_payment", _evidence(amount_usd=50_000.0, approvals=_approvals("a1", "a2"))
        )
        self.assertIsNone(command.update["governance_error"])
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(len(self.executions), 1)

    def test_duplicate_approver_and_self_approval_are_blocked(self) -> None:
        agent = self._agent()
        duplicate = agent.execute(
            "release_payment", _evidence(amount_usd=50_000.0, approvals=_approvals("a1", "a1"))
        )
        self.assertIn("distinct approvers", self._blocked_locally(duplicate))
        self_approved = agent.execute(
            "release_payment", _evidence(amount_usd=50_000.0, approvals=_approvals("emp-1", "a2"))
        )
        self.assertIn("independent of the requester", self._blocked_locally(self_approved))

    def test_required_approvers_is_configurable(self) -> None:
        agent = self._agent(required_approvers=3)
        command = agent.execute(
            "release_payment", _evidence(amount_usd=50_000.0, approvals=_approvals("a1", "a2"))
        )
        self.assertIn("requires 3 distinct approvers", self._blocked_locally(command))

    def test_missing_or_malformed_evidence_is_blocked_without_echoing_values(self) -> None:
        secret = "super-secret-clearance"
        cases = {
            "missing token": {k: v for k, v in _evidence().items() if k != "budget_clearance_token"},
            "bool amount": _evidence(amount_usd=True, budget_clearance_token=secret),
            "string amount": _evidence(amount_usd="100", budget_clearance_token=secret),
            "nan amount": _evidence(amount_usd=float("nan"), budget_clearance_token=secret),
            "zero amount": _evidence(amount_usd=0, budget_clearance_token=secret),
            "blank request": _evidence(request_id="  ", budget_clearance_token=secret),
            "bad approval": _evidence(approvals=[{"approver_id": "a1"}], budget_clearance_token=secret),
        }
        agent = self._agent()
        for name, payload in cases.items():
            with self.subTest(name):
                reason = self._blocked_locally(agent.execute("approve_expenditure", payload))
                self.assertIn("incomplete or malformed", reason)
                self.assertNotIn(secret, reason)

    def test_backend_block_still_prevents_execution(self) -> None:
        self.client.allowed = False
        command = self._agent().execute("approve_expenditure", _evidence())

        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.executions, [])
        self.assertIsNotNone(command.update["governance_error"])

    def test_budget_and_approval_tokens_are_verified_locally_when_verifiers_are_set(self) -> None:
        budget_key, approver_key = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
        agent = self._agent(
            budget_verifier=Ed25519TokenVerifier({"k1": public_key_hex(budget_key)}, clock=lambda: NOW),
            approval_verifier=Ed25519TokenVerifier({"k1": public_key_hex(approver_key)}, clock=lambda: NOW),
        )
        base = {"request_id": "REQ-100", "amount_usd": 50_000.0}

        def payload(**overrides: Any) -> dict[str, Any]:
            approvals = [
                {
                    "approver_id": approver,
                    "approval_token": mint_token(
                        approver_key, {**base, "approver_id": approver, "exp": NOW + 60}
                    ),
                }
                for approver in ("a1", "a2")
            ]
            budget = mint_token(budget_key, {**base, "budget_id": "BUD-7", "exp": NOW + 60})
            return _evidence(
                amount_usd=50_000.0,
                budget_remaining_usd=60_000.0,
                budget_clearance_token=budget,
                approvals=approvals,
            ) | overrides

        good = agent.execute("release_payment", payload())
        self.assertIsNone(good.update["governance_error"])
        self.assertEqual(len(self.client.calls), 1)

        self.client.calls.clear()
        self.executions.clear()
        wrong_amount = mint_token(
            budget_key, {"request_id": "REQ-100", "amount_usd": 1.0, "budget_id": "BUD-7", "exp": NOW + 60}
        )
        reason = self._blocked_locally(
            agent.execute("release_payment", payload(budget_clearance_token=wrong_amount))
        )
        self.assertIn("Budget clearance token rejected", reason)

        forged = [
            {"approver_id": "a1", "approval_token": mint_token(
                budget_key, {**base, "approver_id": "a1", "exp": NOW + 60})},
            payload()["approvals"][1],
        ]
        reason = self._blocked_locally(agent.execute("release_payment", payload(approvals=forged)))
        self.assertIn("Approval token for approver 'a1' rejected", reason)

    def test_compiled_graph_applies_the_same_local_checks(self) -> None:
        state = self._agent().invoke(
            {
                "messages": [],
                "tool_invocation": {
                    "name": "release_payment",
                    "arguments": _evidence(amount_usd=99_999.0, budget_remaining_usd=1.0),
                    "tool_call_id": "g1",
                },
            }
        )
        self.assertIn("exceeds the remaining cleared budget", state["governance_error"])
        self.assertEqual(self.client.calls, [])

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
