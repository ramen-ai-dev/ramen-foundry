"""Genuine headless MuJoCo tests for the Panda trajectory execution boundary."""

from __future__ import annotations

import json
import unittest
from typing import Any

from examples.simulate_robotics_3d import (
    PandaMujocoWorkcell,
    build_simulation_scenarios,
    run_governed_scene,
)


class SimulationPolicyFakeClient:
    """Receipt-bearing offline policy client for real MuJoCo execution tests."""

    def __init__(self) -> None:
        self.evaluations: list[dict[str, Any]] = []

    def evaluate_compliance(self, input_text: str, **_: Any) -> dict[str, Any]:
        arguments = json.loads(input_text)["arguments"]
        allowed = arguments["scene_context_id"] == "MUJOCO-ISO15066-ALLOW-002"
        result = {
            "allowed": allowed,
            "receipt_verified": True,
            "receipt_valid": True,
            "steering": None if allowed else "De-energize and isolate before tool insertion.",
            "data": {
                "receipt": {
                    "kid": "test-ed25519-kid",
                    "signature": "test-signature",
                    "canonical_payload": json.dumps({"verdict": 1 if allowed else 0}),
                }
            },
        }
        self.evaluations.append(result)
        return result


class RoboticsSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workcell = PandaMujocoWorkcell()
        self.client = SimulationPolicyFakeClient()
        self.blocked, self.allowed = build_simulation_scenarios()

    def test_headless_blocked_trajectory_clamps_controls_and_velocity_to_zero(self) -> None:
        outcome = run_governed_scene(
            self.workcell,
            self.client,
            self.blocked,
            approach_steps=8,
            execution_steps=8,
        )

        self.assertEqual(outcome.decision, "[BLOCKED]")
        self.assertEqual(outcome.host_executions, 0)
        self.assertTrue(all(value == 0.0 for value in outcome.control_inputs))
        self.assertTrue(all(value == 0.0 for value in outcome.joint_velocities))
        self.assertAlmostEqual(
            float(self.workcell.model.geom_rgba[self.workcell.hazard_wireframe_id, 3]),
            0.30,
        )

    def test_headless_allowed_trajectory_completes_without_unhandled_exception(self) -> None:
        outcome = run_governed_scene(
            self.workcell,
            self.client,
            self.allowed,
            approach_steps=8,
            execution_steps=8,
        )

        self.assertEqual(outcome.decision, "[ALLOWED]")
        self.assertEqual(outcome.host_executions, 1)
        self.assertEqual(len(outcome.control_inputs), 7)
        self.assertEqual(len(outcome.joint_velocities), 7)

    def test_model_exposes_native_seven_dof_panda_form_actuation(self) -> None:
        self.assertEqual(self.workcell.model.nu, 9)
        self.assertEqual(self.workcell.model.nq, 9)
        self.assertEqual(len(self.workcell.arm_actuators), 7)


if __name__ == "__main__":
    unittest.main()
