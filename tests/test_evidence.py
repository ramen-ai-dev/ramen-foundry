"""Tests for the local evidence verification protocol."""

from __future__ import annotations

import unittest

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ramen_foundry import Ed25519TokenVerifier, EvidenceResult, EvidenceVerifier
from ramen_foundry.core.evidence import claims_equal
from tests.gate_fixtures import mint_token, public_key_hex

NOW = 1_800_000_000.0


class Ed25519TokenVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.key = Ed25519PrivateKey.generate()
        self.verifier = Ed25519TokenVerifier({"k1": public_key_hex(self.key)}, clock=lambda: NOW)
        self.claims = {"request_id": "REQ-1", "amount_usd": 500, "exp": NOW + 60}

    def test_satisfies_the_protocol(self) -> None:
        self.assertIsInstance(self.verifier, EvidenceVerifier)

    def test_valid_token_with_matching_claims_is_accepted(self) -> None:
        token = mint_token(self.key, self.claims)
        result = self.verifier.verify(token, expected={"request_id": "REQ-1", "amount_usd": 500.0})
        self.assertEqual(result, EvidenceResult.ok())

    def test_rejections_fail_closed_with_a_reason(self) -> None:
        other = Ed25519PrivateKey.generate()
        tampered = mint_token(self.key, self.claims).split(".")
        tampered[2] = mint_token(self.key, {**self.claims, "amount_usd": 5}).split(".")[2]
        cases = {
            "wrong signer": mint_token(other, self.claims),
            "unknown kid": mint_token(self.key, self.claims, kid="k2"),
            "tampered claims": ".".join(tampered),
            "expired": mint_token(self.key, {**self.claims, "exp": NOW - 1}),
            "no expiry": mint_token(self.key, {"request_id": "REQ-1", "amount_usd": 500}),
            "claim mismatch": mint_token(self.key, {**self.claims, "request_id": "REQ-2"}),
            "claim absent": mint_token(self.key, {"amount_usd": 500, "exp": NOW + 60}),
            "malformed": "not-a-token",
            "not a string": None,
            "empty": "",
        }
        for name, token in cases.items():
            with self.subTest(name):
                result = self.verifier.verify(
                    token, expected={"request_id": "REQ-1", "amount_usd": 500}
                )
                self.assertFalse(result.valid)
                self.assertTrue(result.reason)

    def test_verify_never_raises_on_garbage(self) -> None:
        for garbage in ("v1.k1.%%%.%%%", "v1.k1..", "v1.k1.e30.e30", 12, {"a": 1}, b"bytes"):
            with self.subTest(garbage=garbage):
                self.assertFalse(self.verifier.verify(garbage, expected={}).valid)

    def test_expiry_can_be_made_optional_explicitly(self) -> None:
        lenient = Ed25519TokenVerifier(
            {"k1": public_key_hex(self.key)}, require_expiry=False, clock=lambda: NOW
        )
        token = mint_token(self.key, {"request_id": "REQ-1"})
        self.assertTrue(lenient.verify(token, expected={"request_id": "REQ-1"}).valid)

    def test_constructor_rejects_bad_keys(self) -> None:
        for keys in ({}, {"k1": "zz"}, {"k.1": public_key_hex(self.key)}, {"k1": "ab" * 31}):
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                Ed25519TokenVerifier(keys)

    def test_claims_equal_does_not_confuse_types(self) -> None:
        self.assertTrue(claims_equal(5, 5.0))
        self.assertFalse(claims_equal(True, 1))
        self.assertFalse(claims_equal("5", 5))
        self.assertFalse(claims_equal(float("nan"), float("nan")))
        self.assertFalse(claims_equal(None, None))


if __name__ == "__main__":
    unittest.main()
