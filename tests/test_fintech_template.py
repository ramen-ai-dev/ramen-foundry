"""Policy-bound tests for the commercial lending template."""

from __future__ import annotations

import hashlib
import json
import unittest
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from langchain_core.tools import tool

from ramen_foundry import CommercialLendingAgent, ToolInvocation
from ramen_foundry.templates.fintech import FINTECH_BANKING_INVARIANCE_BUNDLE_ID

_SIGNING_KEY = Ed25519PrivateKey.from_private_bytes(
    bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
)
_WIRE_SIGNED_FIELDS = (
    "account_id",
    "amount_usd",
    "beneficiary_name",
    "beneficiary_routing",
    "beneficiary_account",
    "sanction_clearance_token",
    "gl_offset",
)


def _wire_authorization(payload: dict[str, Any]) -> bytes:
    signed = {field: payload[field] for field in _WIRE_SIGNED_FIELDS}
    return json.dumps(signed, sort_keys=True, separators=(",", ":")).encode()


def _ofac_token(payload: dict[str, Any]) -> str:
    subject = "|".join(
        (
            payload["beneficiary_name"],
            payload["beneficiary_routing"],
            payload["beneficiary_account"],
        )
    )
    digest = hashlib.sha256(subject.encode()).hexdigest().upper()
    return f"OFAC-DET-20260910-{digest[:20]}"


def _signed_wire_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "account_id": "acct-commercial-1042",
        "amount_usd": 250000.0,
        "beneficiary_name": "Harbor Equipment LLC",
        "beneficiary_routing": "021000021",
        "beneficiary_account": "9876543210",
        "gl_offset": "1010-commercial-loans",
    }
    payload["sanction_clearance_token"] = _ofac_token(payload)
    public_key = _SIGNING_KEY.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    payload["co_signer_public_key"] = f"ed25519:{public_key.hex()}"
    signature = _SIGNING_KEY.sign(_wire_authorization(payload))
    payload["co_signer_signature"] = f"ed25519:{signature.hex()}"
    return payload


def _credit_payload() -> dict[str, Any]:
    return {
        "application_id": "app-2026-0042",
        "decision": "declined",
        "reg_b_reason_codes": ["HIGH_CREDIT_UTILIZATION"],
        "model_hash": f"sha256:{hashlib.sha256(b'underwriting-model-v7').hexdigest()}",
        "shap_attribution_summary": {
            "HIGH_CREDIT_UTILIZATION": -0.73,
            "YEARS_IN_BUSINESS": 0.21,
        },
    }


class PolicyAwareFakeRamenClient:
    """Model the promoted bundle's evidence invariants for deterministic unit tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def evaluate_compliance(self, input_text: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((input_text, kwargs))
        invocation = json.loads(input_text)
        tool_name = invocation["tool"]
        arguments = invocation["arguments"]
        if tool_name == "issue_credit_adverse_action":
            allowed = self._credit_is_allowed(arguments)
        elif tool_name == "dispatch_wire":
            allowed = self._wire_is_allowed(arguments)
        else:
            allowed = False
        return {
            "allowed": allowed,
            "receipt_verified": True,
            "receipt_reason": None,
            "steering": None if allowed else "Required statutory evidence did not verify.",
        }

    @staticmethod
    def _credit_is_allowed(arguments: dict[str, Any]) -> bool:
        attribution = arguments.get("shap_attribution_summary")
        reason_codes = arguments.get("reg_b_reason_codes")
        if (
            not arguments.get("application_id")
            or arguments.get("decision") != "declined"
            or not arguments.get("model_hash")
            or not isinstance(attribution, dict)
            or not isinstance(reason_codes, list)
        ):
            return False
        negative_factors = {
            factor
            for factor, weight in attribution.items()
            if isinstance(weight, (int, float)) and weight < 0
        }
        return bool(negative_factors) and set(reason_codes) == negative_factors

    @staticmethod
    def _wire_is_allowed(arguments: dict[str, Any]) -> bool:
        if arguments.get("amount_usd", 0) < 10000:
            return True
        public_key_text = arguments.get("co_signer_public_key")
        signature_text = arguments.get("co_signer_signature")
        if not public_key_text or not signature_text:
            return False
        if arguments.get("sanction_clearance_token") != _ofac_token(arguments):
            return False
        try:
            public_key = Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(public_key_text.removeprefix("ed25519:"))
            )
            public_key.verify(
                bytes.fromhex(signature_text.removeprefix("ed25519:")),
                _wire_authorization(arguments),
            )
        except (InvalidSignature, TypeError, ValueError):
            return False
        return True


class CommercialLendingAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executions: list[tuple[str, dict[str, Any]]] = []

        @tool
        def issue_credit_adverse_action(
            application_id: str,
            decision: str,
            reg_b_reason_codes: list[str],
            model_hash: str,
            shap_attribution_summary: dict[str, float],
        ) -> str:
            """Issue a governed commercial-credit adverse-action notice."""

            self.executions.append(
                (
                    "issue_credit_adverse_action",
                    {
                        "application_id": application_id,
                        "decision": decision,
                        "reg_b_reason_codes": reg_b_reason_codes,
                        "model_hash": model_hash,
                        "shap_attribution_summary": shap_attribution_summary,
                    },
                )
            )
            return "ALLOW"

        @tool
        def dispatch_wire(
            account_id: str,
            amount_usd: float,
            beneficiary_name: str,
            beneficiary_routing: str,
            beneficiary_account: str,
            sanction_clearance_token: str,
            gl_offset: str,
            co_signer_public_key: str,
            co_signer_signature: str,
        ) -> str:
            """Dispatch a governed commercial-loan wire transfer."""

            self.executions.append(
                (
                    "dispatch_wire",
                    {
                        "account_id": account_id,
                        "amount_usd": amount_usd,
                        "beneficiary_name": beneficiary_name,
                        "beneficiary_routing": beneficiary_routing,
                        "beneficiary_account": beneficiary_account,
                        "sanction_clearance_token": sanction_clearance_token,
                        "gl_offset": gl_offset,
                        "co_signer_public_key": co_signer_public_key,
                        "co_signer_signature": co_signer_signature,
                    },
                )
            )
            return "ALLOW"

        self.client = PolicyAwareFakeRamenClient()
        self.tools = {
            "issue_credit_adverse_action": issue_credit_adverse_action,
            "dispatch_wire": dispatch_wire,
        }
        self.agent = CommercialLendingAgent(
            client=self.client,  # type: ignore[arg-type]
            tools=self.tools,
        )

    def test_matching_negative_shap_reason_returns_allow(self) -> None:
        command = self.agent.execute("issue_credit_adverse_action", _credit_payload())

        self.assertIsNone(command.update["governance_error"])
        self.assertEqual(command.update["messages"][0].content, "ALLOW")
        self.assertEqual(self.executions[0][0], "issue_credit_adverse_action")
        self.assertEqual(
            self.client.calls[0][1]["bundle_ids"],
            [FINTECH_BANKING_INVARIANCE_BUNDLE_ID],
        )
        self.assertIsNone(self.client.calls[0][1]["policy_ids"])
        self.assertIsNone(self.client.calls[0][1]["provider_key"])
        self.assertIsNone(self.client.calls[0][1]["provider_name"])

    def test_hallucinated_reason_or_missing_model_hash_returns_block(self) -> None:
        hallucinated = _credit_payload()
        hallucinated["reg_b_reason_codes"] = ["INSUFFICIENT_COLLATERAL"]
        missing_hash = _credit_payload()
        missing_hash.pop("model_hash")

        for payload in (hallucinated, missing_hash):
            with self.subTest(payload=payload):
                command = self.agent.execute("issue_credit_adverse_action", payload)
                self.assertIsNotNone(command.update["governance_error"])
                self.assertIn("Governance blocked", command.update["messages"][0].content)

        self.assertEqual(self.executions, [])

    def test_signed_high_value_wire_with_ofac_token_returns_allow(self) -> None:
        payload = _signed_wire_payload()
        byok_client = PolicyAwareFakeRamenClient()
        byok_agent = CommercialLendingAgent(
            client=byok_client,  # type: ignore[arg-type]
            tools=self.tools,
            provider_key="provider-test-value",
            provider_name="openai",
        )
        state = byok_agent.invoke(
            {
                "messages": [],
                "tool_invocation": ToolInvocation(
                    name="dispatch_wire",
                    arguments=payload,
                    tool_call_id="wire-allow-1",
                ),
            }
        )

        self.assertIsNone(state["governance_error"])
        self.assertIsNone(state["tool_invocation"])
        self.assertEqual(state["messages"][-1].content, "ALLOW")
        self.assertEqual(self.executions[0][0], "dispatch_wire")
        self.assertEqual(byok_client.calls[0][1]["provider_key"], "provider-test-value")
        self.assertEqual(byok_client.calls[0][1]["provider_name"], "openai")

    def test_wire_without_cosigner_or_deterministic_sanctions_token_returns_block(self) -> None:
        missing_cosigner = _signed_wire_payload()
        missing_cosigner["co_signer_public_key"] = None
        missing_cosigner["co_signer_signature"] = None
        conversational_clearance = _signed_wire_payload()
        conversational_clearance["sanction_clearance_token"] = (
            "The beneficiary was checked and is not sanctioned."
        )

        for payload in (missing_cosigner, conversational_clearance):
            with self.subTest(payload=payload):
                command = self.agent.execute("dispatch_wire", payload)
                self.assertIsNotNone(command.update["governance_error"])
                self.assertIn("Governance blocked", command.update["messages"][0].content)

        self.assertEqual(self.executions, [])


if __name__ == "__main__":
    unittest.main()
