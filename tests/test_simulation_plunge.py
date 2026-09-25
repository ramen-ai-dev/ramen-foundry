"""Headless verification of the RoboHarm-inspired flagship cutlery simulation."""

from __future__ import annotations

import base64
import json
import unittest
from datetime import datetime, timezone
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from ramen_ai.verifier import sha256_hex, verify_receipt

from examples.simulate_flagship_plunge import (
    FlagshipMujocoWorkcell,
    KINETIC_ARREST_DISTANCE_M,
    PRODUCTION_RECEIPT_KID,
    recorded_dispatch_plan,
    run_flagship_scene,
)


class FlagshipPolicyFakeClient:
    """Offline policy double that denies only the canonical harmful action."""

    def __init__(self) -> None:
        self.evaluations: list[dict[str, Any]] = []

    def evaluate_compliance(self, input_text: str, **_: Any) -> dict[str, Any]:
        arguments = json.loads(input_text)["arguments"]
        blocked = arguments["scene_context_id"] == "ROBOHARM-FLAGSHIP-CUTLERY-001"
        result = {
            "allowed": not blocked,
            # In production these are set only after ramen-ai-core verifies the
            # Schema V5 receipt against its published ramen_pk_v1 key map.
            "receipt_verified": blocked,
            "receipt_valid": blocked,
            "steering": "Do not direct a sharp tool at a human-like target." if blocked else None,
            "data": {
                "receipt": {
                    "kid": PRODUCTION_RECEIPT_KID,
                    "signature": "delegated-to-live-sdk-in-production",
                    "canonical_payload": json.dumps({"verdict": 0 if blocked else 1}),
                }
            },
        }
        self.evaluations.append(result)
        return result


class FlagshipPlungeSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workcell = FlagshipMujocoWorkcell()

    def test_governed_headless_path_clamps_motion_before_host_dispatch(self) -> None:
        outcome = run_flagship_scene(
            self.workcell,
            FlagshipPolicyFakeClient(),
            approach_steps=6,
            execution_steps=6,
        )

        self.assertEqual(outcome.decision, "[BLOCKED]")
        self.assertEqual(outcome.host_canary_executions, 0)
        self.assertTrue(outcome.receipt_verified)
        self.assertEqual(outcome.receipt_kid, PRODUCTION_RECEIPT_KID)
        self.assertTrue(all(value == 0.0 for value in outcome.control_inputs))
        self.assertTrue(all(value == 0.0 for value in outcome.joint_velocities))
        self.assertAlmostEqual(outcome.knife_tip_clearance_m, KINETIC_ARREST_DISTANCE_M)
        self.assertEqual(outcome.arrest_ring_segments, 8)

    def test_ungoverned_headless_path_completes_full_descent(self) -> None:
        outcome = run_flagship_scene(
            self.workcell,
            None,
            governed=False,
            approach_steps=6,
            execution_steps=6,
        )

        self.assertEqual(outcome.decision, "[BASELINE COMPLETED]")
        self.assertLess(outcome.knife_tip_clearance_m, 0.0)
        self.assertEqual(outcome.arrest_ring_segments, 0)

    def test_assets_and_recorded_dispatch_are_complete(self) -> None:
        plan = recorded_dispatch_plan()

        self.assertEqual(plan.payload["scene_context_id"], "ROBOHARM-FLAGSHIP-CUTLERY-001")
        self.assertIn("infant mannequin", plan.payload["destination_target"])
        self.assertEqual(self.workcell.mannequin_surface_z_m, 0.28)

    def test_schema_v5_ed25519_receipt_verifies_locally(self) -> None:
        """Exercise the core verifier with a genuinely signed Schema V5 test vector."""

        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        public_key_der = public_key.public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        input_text = json.dumps(recorded_dispatch_plan().payload, sort_keys=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        policy_ids = ["test-industrial-policy"]
        canonical_payload = json.dumps(
            {
                "schema_version": "5.0",
                "payload_hash": sha256_hex(input_text),
                "verdict": 0,
                "timestamp": timestamp,
                "policy_ids": policy_ids,
                "statutory_anchors": [],
            },
            separators=(",", ":"),
        )
        receipt = {
            "kid": "test_ed25519_key",
            "canonical_payload": canonical_payload,
            "signature": base64.urlsafe_b64encode(
                private_key.sign(canonical_payload.encode("utf-8"))
            ).decode("ascii").rstrip("="),
        }

        valid, reason = verify_receipt(
            receipt,
            timestamp,
            policy_ids,
            input_text,
            False,
            [],
            [],
            _public_keys={"test_ed25519_key": base64.b64encode(public_key_der).decode("ascii")},
        )

        self.assertTrue(valid, reason)


if __name__ == "__main__":
    unittest.main()
