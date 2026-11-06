"""Tests that IndustrialAutomationAgent enforces ManipulationDispatchPayload."""

from __future__ import annotations

import copy
import unittest
from typing import Any

from langchain_core.tools import tool

import ramen_foundry
import ramen_foundry.templates as templates
from ramen_foundry import (
    IndustrialAutomationAgent,
    ManipulationActionType,
    ManipulationDispatchPayload,
    ToolInvocation,
    manipulation_payload_problem,
)
from tests.gate_fixtures import FakeClient

GOOD: dict[str, Any] = {
    "robot_id": "ARM-01",
    "action_type": "PICK_AND_PLACE",
    "target_object": "aluminium bracket",
    "destination_target": "assembly_fixture",
    "commanded_velocity_mps": 0.25,
    "commanded_force_nm": 35,
    "human_proximity_meters": 1.8,
    "active_hazard_flags": [],
    "scene_context_id": "COLLABORATIVE_ASSEMBLY_CELL",
}


class ManipulationPayloadContractTests(unittest.TestCase):
    def test_contract_is_exported_from_both_packages(self) -> None:
        for module in (ramen_foundry, templates):
            self.assertIn("ManipulationDispatchPayload", module.__all__)
            self.assertIn("ManipulationActionType", module.__all__)
            self.assertIn("manipulation_payload_problem", module.__all__)
        self.assertEqual(
            set(ManipulationDispatchPayload.__annotations__),
            set(GOOD),
        )
        self.assertIsNotNone(ManipulationActionType)

    def test_valid_payload_has_no_problem(self) -> None:
        self.assertIsNone(manipulation_payload_problem(GOOD))
        self.assertIsNone(manipulation_payload_problem({**GOOD, "commanded_force_nm": 0, "active_hazard_flags": ["FIRE"]}))

    def test_violations_are_reported_by_field_without_values(self) -> None:
        cases: dict[str, tuple[dict[str, Any] | str, str]] = {
            "missing key": ({k: v for k, v in GOOD.items() if k != "robot_id"}, "robot_id"),
            "unexpected key": ({**GOOD, "override_safety": True}, "override_safety"),
            "unknown action": ({**GOOD, "action_type": "STAB"}, "action_type"),
            "blank string": ({**GOOD, "target_object": "   "}, "target_object"),
            "numeric string": ({**GOOD, "commanded_velocity_mps": "0.2"}, "commanded_velocity_mps"),
            "bool as number": ({**GOOD, "commanded_force_nm": True}, "commanded_force_nm"),
            "nan": ({**GOOD, "commanded_force_nm": float("nan")}, "commanded_force_nm"),
            "infinity": ({**GOOD, "human_proximity_meters": float("inf")}, "human_proximity_meters"),
            "negative": ({**GOOD, "human_proximity_meters": -0.5}, "human_proximity_meters"),
            "flags not strings": ({**GOOD, "active_hazard_flags": [1]}, "active_hazard_flags"),
            "flags not a list": ({**GOOD, "active_hazard_flags": "FIRE"}, "active_hazard_flags"),
            "not an object": ("dispatch", "payload"),
        }
        for name, (arguments, field) in cases.items():
            with self.subTest(name):
                problem = manipulation_payload_problem(arguments)
                self.assertIsNotNone(problem)
                self.assertIn(field, str(problem))
        secret = "SECRET-TARGET-NAME"
        problem = manipulation_payload_problem({**GOOD, "target_object": secret, "extra": secret})
        self.assertNotIn(secret, str(problem))


class IndustrialManipulationEnforcementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executions: list[dict[str, Any]] = []

        @tool
        def dispatch_manipulation(
            robot_id: str,
            action_type: str,
            target_object: str,
            destination_target: str,
            commanded_velocity_mps: float,
            commanded_force_nm: float,
            human_proximity_meters: float,
            active_hazard_flags: list[str],
            scene_context_id: str,
        ) -> str:
            """Dispatch a manipulation."""

            self.executions.append({"robot_id": robot_id})
            return "dispatched"

        self.client = FakeClient()
        self.agent = IndustrialAutomationAgent(
            client=self.client,  # type: ignore[arg-type]
            tools={"dispatch_manipulation": dispatch_manipulation},
        )

    def test_conforming_payload_is_evaluated_and_dispatched(self) -> None:
        command = self.agent.execute("dispatch_manipulation", GOOD)

        self.assertIsNone(command.update["governance_error"])
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.executions, [{"robot_id": "ARM-01"}])

    def test_nonconforming_payloads_never_reach_the_backend_or_the_tool(self) -> None:
        for name, payload in {
            "missing": {k: v for k, v in GOOD.items() if k != "scene_context_id"},
            "extra": {**GOOD, "note": "ignore previous limits"},
            "nan": {**GOOD, "commanded_velocity_mps": float("nan")},
            "bool": {**GOOD, "commanded_force_nm": False},
        }.items():
            with self.subTest(name):
                command = self.agent.execute("dispatch_manipulation", copy.deepcopy(payload))
                self.assertIn("ManipulationDispatchPayload", command.update["governance_error"])
                self.assertEqual(self.client.calls, [])
                self.assertEqual(self.executions, [])

    def test_compiled_graph_enforces_the_contract(self) -> None:
        state = self.agent.invoke(
            {
                "messages": [],
                "tool_invocation": ToolInvocation(
                    name="dispatch_manipulation",
                    arguments={**GOOD, "action_type": "STAB"},
                    tool_call_id="m1",
                ),
            }
        )

        self.assertIn("action_type", state["governance_error"])
        self.assertEqual(self.client.calls, [])
        self.assertEqual(self.executions, [])

    def test_other_industrial_tools_are_not_subject_to_the_manipulation_contract(self) -> None:
        @tool
        def mutate_safety_parameter(node_id: str) -> str:
            """Mutate."""

            return "ok"

        agent = IndustrialAutomationAgent(
            client=self.client,  # type: ignore[arg-type]
            tools={"mutate_safety_parameter": mutate_safety_parameter},
        )
        command = agent.execute("mutate_safety_parameter", {"node_id": "SIS-1"})
        self.assertIsNone(command.update["governance_error"])
        self.assertEqual(len(self.client.calls), 1)


if __name__ == "__main__":
    unittest.main()
