"""Offline unit tests for the empirical German Credit benchmark."""

from __future__ import annotations

import json
import unittest
from typing import Any

from langchain_core.tools import tool

from examples.benchmark_credit_data import (
    EXCLUDED_MODEL_FEATURES,
    assert_verified_live_outcome,
    build_adverse_action_payload,
    build_real_credit_evidence,
    map_negative_shap_to_reg_b,
)
from ramen_foundry import CommercialLendingAgent
from ramen_foundry.templates.fintech import FINTECH_BANKING_INVARIANCE_BUNDLE_ID


class BenchmarkPolicyFakeClient:
    """Apply the benchmark's attribution invariant without network access."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def evaluate_compliance(self, input_text: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((input_text, kwargs))
        arguments = json.loads(input_text)["arguments"]
        negative_reasons = {
            reason
            for reason, value in arguments["shap_attribution_summary"].items()
            if value < 0
        }
        allowed = (
            bool(arguments.get("model_hash"))
            and set(arguments["reg_b_reason_codes"]) == negative_reasons
        )
        steering = None if allowed else "Use principal reasons grounded in model attribution."
        canonical_payload = json.dumps({"verdict": 1 if allowed else 0})
        return {
            "allowed": allowed,
            "receipt_verified": True,
            "receipt_valid": True,
            "receipt_reason": None,
            "steering": steering,
            "policy_ids": ["credit-policy-id"],
            "data": {
                "statutory_anchors": ["12 C.F.R. § 1002.9"],
                "receipt": {
                    "kid": "test-kid",
                    "signature": "test-signature",
                    "canonical_payload": canonical_payload,
                },
            },
        }


class CreditBenchmarkTests(unittest.TestCase):
    def test_maps_only_grounded_adverse_features_to_reg_b_codes(self) -> None:
        mapped = map_negative_shap_to_reg_b(
            {
                "age": 1.4,
                "installment_commitment": 0.72,
                "credit_history": 0.51,
                "employment": -0.20,
                "savings_status": 0.33,
                "unmapped_free_text": 4.0,
            },
            top_n=3,
        )

        self.assertEqual(
            mapped,
            {
                "HIGH_DEBT_TO_INCOME_RATIO": -0.72,
                "DELINQUENT_CREDIT_HISTORY": -0.51,
                "INSUFFICIENT_CASH_RESERVES": -0.33,
            },
        )
        self.assertNotIn("age", mapped)

    def test_builds_typed_payload_and_rejects_non_adverse_values(self) -> None:
        digest = "sha256:" + ("a" * 64)
        payload = build_adverse_action_payload(
            application_id="credit-g-row-42",
            model_hash=digest,
            reason_attributions={"DELINQUENT_CREDIT_HISTORY": -0.61},
        )

        self.assertEqual(payload["decision"], "declined")
        self.assertEqual(payload["reg_b_reason_codes"], ["DELINQUENT_CREDIT_HISTORY"])
        self.assertEqual(payload["model_hash"], digest)
        with self.assertRaisesRegex(ValueError, "finite negative"):
            build_adverse_action_payload(
                application_id="credit-g-row-42",
                model_hash=digest,
                reason_attributions={"DELINQUENT_CREDIT_HISTORY": 0.61},
            )

    def test_verified_outcome_requires_receipt_and_block_steering(self) -> None:
        fake = BenchmarkPolicyFakeClient()
        allowed = fake.evaluate_compliance(
            json.dumps(
                {
                    "arguments": {
                        "model_hash": "sha256:" + ("a" * 64),
                        "reg_b_reason_codes": ["DELINQUENT_CREDIT_HISTORY"],
                        "shap_attribution_summary": {
                            "DELINQUENT_CREDIT_HISTORY": -0.5
                        },
                    }
                }
            )
        )
        metadata = assert_verified_live_outcome(allowed, expected_allowed=True)
        self.assertEqual(metadata["kid"], "test-kid")

        blocked = dict(allowed)
        blocked.update(
            {
                "allowed": False,
                "steering": "Use grounded reasons.",
                "data": {
                    **allowed["data"],
                    "receipt": {
                        **allowed["data"]["receipt"],
                        "canonical_payload": json.dumps({"verdict": 0}),
                    },
                },
            }
        )
        assert_verified_live_outcome(
            blocked,
            expected_allowed=False,
            require_steering=True,
        )
        blocked["receipt_verified"] = False
        with self.assertRaisesRegex(AssertionError, "did not verify"):
            assert_verified_live_outcome(blocked, expected_allowed=False)

    def test_complete_model_path_is_deterministic_and_excludes_proxy_fields(self) -> None:
        from types import SimpleNamespace

        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(42)
        row_count = 160
        checking_status = rng.choice(["<0", "0<=X<200", ">=200"], row_count)
        duration = rng.integers(6, 60, row_count)
        credit_history = rng.choice(
            ["critical/other existing credit", "existing paid", "no credits/all paid"],
            row_count,
        )
        installment_commitment = rng.integers(1, 5, row_count)
        frame = pd.DataFrame(
            {
                "checking_status": checking_status,
                "duration": duration,
                "credit_history": credit_history,
                "credit_amount": rng.integers(500, 15000, row_count),
                "savings_status": rng.choice(["<100", "100<=X<500", ">=1000"], row_count),
                "employment": rng.choice(["unemployed", "1<=X<4", ">=7"], row_count),
                "installment_commitment": installment_commitment,
                "age": rng.integers(18, 80, row_count),
                "foreign_worker": rng.choice(["yes", "no"], row_count),
                "personal_status": rng.choice(["single", "married"], row_count),
                "own_telephone": rng.choice(["none", "yes"], row_count),
            }
        )
        risk_score = (
            (checking_status == "<0").astype(int) * 3
            + (duration > 36).astype(int) * 2
            + (credit_history == "critical/other existing credit").astype(int) * 2
            + (installment_commitment >= 4).astype(int)
            + rng.normal(0, 0.4, row_count)
        )
        target = pd.Series(np.where(risk_score >= 4, "bad", "good"))
        calls: list[tuple[str, dict[str, Any]]] = []

        def offline_loader(name: str, **kwargs: Any) -> Any:
            calls.append((name, kwargs))
            return SimpleNamespace(data=frame, target=target)

        first = build_real_credit_evidence(dataset_loader=offline_loader)
        second = build_real_credit_evidence(dataset_loader=offline_loader)

        self.assertEqual(calls[0][0], "credit-g")
        self.assertEqual(calls[0][1]["version"], 1)
        self.assertTrue(EXCLUDED_MODEL_FEATURES.isdisjoint(first.model_features))
        self.assertTrue(EXCLUDED_MODEL_FEATURES.isdisjoint(first.source_risk_shap))
        self.assertEqual(first.dataset_fingerprint, second.dataset_fingerprint)
        self.assertEqual(first.payload["model_hash"], second.payload["model_hash"])
        self.assertEqual(
            first.payload["shap_attribution_summary"],
            second.payload["shap_attribution_summary"],
        )
        self.assertGreaterEqual(first.accuracy, 0.70)
        self.assertGreaterEqual(first.roc_auc, 0.70)

    def test_agent_allows_grounded_payload_and_blocks_hallucinated_reason(self) -> None:
        executions: list[str] = []

        @tool
        def issue_credit_adverse_action(
            application_id: str,
            decision: str,
            reg_b_reason_codes: list[str],
            model_hash: str,
            shap_attribution_summary: dict[str, float],
        ) -> str:
            """Record an offline benchmark adverse action."""

            executions.append(application_id)
            return "issued"

        client = BenchmarkPolicyFakeClient()
        agent = CommercialLendingAgent(
            client=client,  # type: ignore[arg-type]
            tools={"issue_credit_adverse_action": issue_credit_adverse_action},
        )
        payload = build_adverse_action_payload(
            application_id="credit-g-row-77",
            model_hash="sha256:" + ("b" * 64),
            reason_attributions={
                "HIGH_DEBT_TO_INCOME_RATIO": -0.72,
                "DELINQUENT_CREDIT_HISTORY": -0.51,
            },
        )

        allowed_command = agent.execute("issue_credit_adverse_action", payload)
        noisy_payload = dict(payload)
        noisy_payload["reg_b_reason_codes"] = ["GEOGRAPHIC_LOCATION_PROXY"]
        blocked_command = agent.execute("issue_credit_adverse_action", noisy_payload)

        self.assertIsNone(allowed_command.update["governance_error"])
        self.assertIsNotNone(blocked_command.update["governance_error"])
        self.assertEqual(executions, ["credit-g-row-77"])
        for _, options in client.calls:
            self.assertEqual(
                options["bundle_ids"], [FINTECH_BANKING_INVARIANCE_BUNDLE_ID]
            )
            self.assertIsNone(options["policy_ids"])


if __name__ == "__main__":
    unittest.main()
