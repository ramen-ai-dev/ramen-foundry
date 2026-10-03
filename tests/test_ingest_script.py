"""Backfill and verification logic of scripts/ingest_verified_production_exemplars.py (offline)."""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import httpx
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ramen_foundry import RemoteForgeMemoryStore

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ingest_verified_production_exemplars.py"
_spec = importlib.util.spec_from_file_location("ingest_script", _SCRIPT)
ingest = importlib.util.module_from_spec(_spec)
sys.modules["ingest_script"] = ingest
_spec.loader.exec_module(ingest)  # type: ignore[union-attr]

FORGE = "https://forge.example.test"
RECEIPT_ID = "8d1f2c4e-5a6b-4c7d-8e9f-0a1b2c3d4e5f"
SCENARIO = next(s for s in ingest.SCENARIOS if s.tool_name == "dispatch_wire")
STORED_ARGS = dict(SCENARIO.arguments) | {"sanction_clearance_token": "OFAC-DET-20200101-STORED"}


# A test key stands in for ramen_pk_v1, whose private half is not available here.
TEST_KEY = Ed25519PrivateKey.generate()
TEST_KEY_RAW_HEX = TEST_KEY.public_key().public_bytes(
    serialization.Encoding.Raw, serialization.PublicFormat.Raw
).hex()


def _signed(receipt_id: str = RECEIPT_ID, verdict: int = 1) -> tuple[dict[str, Any], Ed25519PrivateKey]:
    key = TEST_KEY
    canonical = json.dumps({"id": receipt_id, "schema_version": "5.0", "kid": "ramen_pk_v1", "verdict": verdict})
    signature = base64.urlsafe_b64encode(key.sign(canonical.encode())).decode().rstrip("=")
    return {"signature": signature, "canonical_payload": canonical, "receipt_id": receipt_id}, key


def _verdict(receipt_id: str = RECEIPT_ID, *, allowed: bool = True, verified: bool = True) -> dict[str, Any]:
    canonical = json.dumps(
        {"id": receipt_id, "schema_version": "5.0", "kid": "ramen_pk_v1", "verdict": 1 if allowed else 0}
    )
    return {
        "allowed": allowed,
        "receipt_verified": verified,
        "receipt_reason": None if verified else "bad signature",
        "data": {
            "receipt": {
                "id": receipt_id, "schema_version": "5.0", "kid": "ramen_pk_v1",
                "signature": "c2ln", "canonical_payload": canonical, "statutory_anchors": [], "attestation": None,
            },
            "total_violations": [] if allowed else [{"reasoning": "Blocked for test."}],
        },
    }


def _stored(**overrides: Any) -> dict[str, Any]:
    record = {
        "exemplar_id": "11111111-1111-4111-8111-111111111111",
        "task_description": SCENARIO.task_description,
        "violation_reason": "No violation: compliant reference action, allowed by ramen-ai on its first attempt.",
        "primary_statutory_anchor": "UCC Article 4A, Section 4A-202",
        "steering_directive": "Stored guidance text.",
        "failed_arguments": {},
        "repaired_arguments": STORED_ARGS,
        "receipt_id": "old-receipt-id",
        "signature": None,
        "canonical_payload": None,
    }
    return record | overrides


class FakeClient:
    def __init__(self, verdict: dict[str, Any]) -> None:
        self.verdict = verdict
        self.evaluated: list[dict[str, Any]] = []

    def evaluate_compliance(self, text: str, **kwargs: Any) -> dict[str, Any]:
        self.evaluated.append(json.loads(text)["arguments"])
        return self.verdict


class IngestTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch.object(ingest, "RAMEN_PK_V1_RAW_HEX", TEST_KEY_RAW_HEX)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_ingest(
        self,
        client: FakeClient,
        reads: list[dict[str, Any] | None],
        *,
        dry_run: bool = False,
        post_status: int = 200,
    ) -> tuple[Any, mock.MagicMock]:
        responses = iter(reads)

        def fake_get(url: str, params: dict[str, str] | None = None, **kwargs: Any) -> httpx.Response:
            record = next(responses)
            body = {"exemplars": [record] if record else []}
            return httpx.Response(200, json=body, request=httpx.Request("GET", url))

        post_response = httpx.Response(post_status, json={"success": True, "refreshed": post_status == 200},
                                       request=httpx.Request("POST", FORGE))
        store = RemoteForgeMemoryStore(base_url=FORGE, write_token="forge-test-token", domain=SCENARIO.domain)
        with mock.patch.object(ingest.httpx, "get", side_effect=fake_get), \
             mock.patch("ramen_foundry.core.memory.httpx.post", return_value=post_response) as post:
            result = ingest.ingest(client, SCENARIO, {}, store, dry_run)
        return result, post

    def test_backfill_evaluates_stored_arguments_and_reposts_stored_text(self) -> None:
        proof, _ = _signed()
        signed_record = _stored(**proof)
        client = FakeClient(_verdict())
        result, post = self.run_ingest(client, [_stored(), signed_record])

        self.assertEqual(result.status, "BACKFILLED")
        self.assertEqual(client.evaluated, [STORED_ARGS])  # the receipt covers what is stored
        sent = post.call_args.kwargs["json"]
        self.assertEqual(sent["violation_reason"], _stored()["violation_reason"])  # upsert key
        self.assertEqual(sent["steering_directive"], "Stored guidance text.")
        self.assertEqual(sent["repaired_arguments"], STORED_ARGS)
        self.assertEqual(sent["receipt"]["id"], RECEIPT_ID)

    def test_already_signed_record_is_skipped_without_evaluation(self) -> None:
        proof, _ = _signed()
        client = FakeClient(_verdict())
        result, post = self.run_ingest(client, [_stored(**proof)])
        self.assertEqual(result.status, "ALREADY SIGNED")
        self.assertEqual(client.evaluated, [])
        post.assert_not_called()

    def test_blocked_backfill_is_skipped_and_nothing_is_posted(self) -> None:
        result, post = self.run_ingest(FakeClient(_verdict(allowed=False)), [_stored()])
        self.assertEqual(result.status, "SKIPPED")
        self.assertIn("BLOCKED live", result.detail)
        post.assert_not_called()

    def test_unverified_receipt_is_skipped(self) -> None:
        result, post = self.run_ingest(FakeClient(_verdict(verified=False)), [_stored()])
        self.assertEqual(result.status, "SKIPPED")
        post.assert_not_called()

    def test_dry_run_posts_nothing(self) -> None:
        result, post = self.run_ingest(FakeClient(_verdict()), [_stored()], dry_run=True)
        self.assertEqual(result.status, "VERIFIED (dry run)")
        post.assert_not_called()

    def test_new_lesson_is_created_when_absent(self) -> None:
        proof, _ = _signed()
        result, post = self.run_ingest(
            FakeClient(_verdict()), [None, _stored(**proof, repaired_arguments=SCENARIO.arguments)], post_status=201
        )
        self.assertEqual(result.status, "CREATED")
        self.assertEqual(post.call_args.kwargs["json"]["repaired_arguments"], SCENARIO.arguments)

    def test_readback_without_signature_is_reported_as_an_error(self) -> None:
        result, _ = self.run_ingest(FakeClient(_verdict()), [_stored(), _stored(receipt_id=RECEIPT_ID)])
        self.assertEqual(result.status, "ERROR")
        self.assertIn("readback failed", result.detail)


class SignatureProblemTests(unittest.TestCase):
    def test_signature_from_the_pinned_key_verifies(self) -> None:
        proof, _ = _signed()
        with mock.patch.object(ingest, "RAMEN_PK_V1_RAW_HEX", TEST_KEY_RAW_HEX):
            self.assertIsNone(ingest.signature_problem(_stored(**proof), RECEIPT_ID))

    def test_signature_from_any_other_key_is_rejected(self) -> None:
        proof, _ = _signed()  # signed by TEST_KEY; the real pinned key differs
        self.assertIn("does not verify", ingest.signature_problem(_stored(**proof), RECEIPT_ID))

    def test_tampered_payload_is_rejected(self) -> None:
        proof, _ = _signed()
        proof["canonical_payload"] = proof["canonical_payload"].replace('"verdict": 1', '"verdict": 0')
        with mock.patch.object(ingest, "RAMEN_PK_V1_RAW_HEX", TEST_KEY_RAW_HEX):
            self.assertIn("does not verify", ingest.signature_problem(_stored(**proof), RECEIPT_ID))

    def test_blocked_signed_verdict_is_rejected(self) -> None:
        proof, _ = _signed(verdict=0)
        with mock.patch.object(ingest, "RAMEN_PK_V1_RAW_HEX", TEST_KEY_RAW_HEX):
            self.assertIn("does not describe this ALLOW", ingest.signature_problem(_stored(**proof), RECEIPT_ID))

    def test_missing_or_mismatched_proof(self) -> None:
        self.assertIn("missing", ingest.signature_problem(_stored(), RECEIPT_ID))
        proof, _ = _signed()
        self.assertIn("stored receipt_id", ingest.signature_problem(_stored(**proof), "other-id"))

    def test_sdk_key_matches_the_key_forge_pins(self) -> None:
        ingest.check_trust_root()


if __name__ == "__main__":
    unittest.main()
