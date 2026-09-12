"""Policy-bound tests for the industrial automation template."""

from __future__ import annotations

import json
import unittest
from typing import Any

from langchain_core.tools import tool

from ramen_foundry import IndustrialAutomationAgent, ToolInvocation
from ramen_foundry.templates import INDUSTRIAL_IOT_ACTUATION_INVARIANCE_BUNDLE_ID

_OPERATING_ENVELOPES = {
    "ENV-CRACKER-Z3-STEADY": {
        "lower": 700.0,
        "upper": 800.0,
        "max_slew_per_sec": 0.05,
        "engineering_units": "CELSIUS",
    },
    "ENV-FURNACE-TUBE-SAFE": {
        "lower": 650.0,
        "upper": 780.0,
        "max_slew_per_sec": 0.02,
        "engineering_units": "CELSIUS",
    },
}


class IndustrialPolicyFakeClient:
    """Model the promoted bundle's decisive actuation invariants."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def evaluate_compliance(self, input_text: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((input_text, kwargs))
        invocation = json.loads(input_text)
        arguments = invocation["arguments"]
        if invocation["tool"] == "adjust_plc_setpoint":
            allowed = self._setpoint_is_allowed(arguments)
        elif invocation["tool"] == "mutate_safety_parameter":
            allowed = self._safety_mutation_is_allowed(arguments)
        else:
            allowed = False
        return {
            "allowed": allowed,
            "receipt_verified": True,
            "receipt_reason": None,
            "steering": None if allowed else "Industrial safety invariant blocked actuation.",
        }

    @staticmethod
    def _setpoint_is_allowed(arguments: dict[str, Any]) -> bool:
        envelope = _OPERATING_ENVELOPES.get(arguments.get("operating_envelope_id"))
        if envelope is None:
            return False
        if arguments.get("engineering_units") != envelope["engineering_units"]:
            return False
        try:
            target = float(arguments["target_val"])
            current = float(arguments["current_val"])
            ramp_seconds = float(arguments["ramp_rate_sec"])
        except (KeyError, TypeError, ValueError):
            return False
        if not envelope["lower"] <= target <= envelope["upper"]:
            return False
        delta = abs(target - current)
        if delta == 0:
            return ramp_seconds >= 0
        if ramp_seconds <= 0:
            return False
        return delta / ramp_seconds <= envelope["max_slew_per_sec"]

    @staticmethod
    def _safety_mutation_is_allowed(arguments: dict[str, Any]) -> bool:
        try:
            duration = float(arguments["duration_sec"])
        except (KeyError, TypeError, ValueError):
            return False
        return (
            arguments.get("physical_key_interlock_verified") is True
            and bool(arguments.get("management_of_change_id"))
            and 0 < duration <= 7200
            and arguments.get("mutation_type") == "SET_DIAGNOSTIC_BYPASS"
        )


class IndustrialAutomationAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executions: list[tuple[str, dict[str, Any]]] = []

        @tool
        def adjust_plc_setpoint(
            node_id: str,
            tag: str,
            target_val: float,
            current_val: float,
            ramp_rate_sec: float,
            engineering_units: str,
            asset_criticality: str,
            operating_envelope_id: str,
        ) -> str:
            """Dispatch a governed PLC setpoint trajectory."""

            self.executions.append(
                (
                    "adjust_plc_setpoint",
                    {
                        "node_id": node_id,
                        "tag": tag,
                        "target_val": target_val,
                        "current_val": current_val,
                        "ramp_rate_sec": ramp_rate_sec,
                        "engineering_units": engineering_units,
                        "asset_criticality": asset_criticality,
                        "operating_envelope_id": operating_envelope_id,
                    },
                )
            )
            return "ALLOW"

        @tool
        def mutate_safety_parameter(
            node_id: str,
            tag: str,
            mutation_type: str,
            requested_value: str | float,
            duration_sec: float,
            physical_key_interlock_verified: bool,
            management_of_change_id: str,
        ) -> str:
            """Dispatch a governed operator-controlled SIS maintenance mutation."""

            self.executions.append(
                (
                    "mutate_safety_parameter",
                    {
                        "node_id": node_id,
                        "tag": tag,
                        "mutation_type": mutation_type,
                        "requested_value": requested_value,
                        "duration_sec": duration_sec,
                        "physical_key_interlock_verified": physical_key_interlock_verified,
                        "management_of_change_id": management_of_change_id,
                    },
                )
            )
            return "ALLOW"

        self.client = IndustrialPolicyFakeClient()
        self.tools = {
            "adjust_plc_setpoint": adjust_plc_setpoint,
            "mutate_safety_parameter": mutate_safety_parameter,
        }
        self.agent = IndustrialAutomationAgent(
            client=self.client,  # type: ignore[arg-type]
            tools=self.tools,
        )

    def test_gradual_setpoint_adjustment_with_safe_slew_returns_allow(self) -> None:
        payload = {
            "node_id": "PLC-ETH-CRK-03",
            "tag": "TIC_3012_SP",
            "target_val": 765.0,
            "current_val": 740.0,
            "ramp_rate_sec": 1200.0,
            "engineering_units": "CELSIUS",
            "asset_criticality": "HIGH_HIGH_CONSEQUENCE",
            "operating_envelope_id": "ENV-CRACKER-Z3-STEADY",
        }

        command = self.agent.execute("adjust_plc_setpoint", payload)

        self.assertIsNone(command.update["governance_error"])
        self.assertEqual(command.update["messages"][0].content, "ALLOW")
        self.assertEqual(self.executions, [("adjust_plc_setpoint", payload)])
        _, options = self.client.calls[0]
        self.assertEqual(
            options["bundle_ids"],
            [INDUSTRIAL_IOT_ACTUATION_INVARIANCE_BUNDLE_ID],
        )
        self.assertIsNone(options["policy_ids"])
        self.assertIsNone(options["provider_key"])
        self.assertIsNone(options["provider_name"])

    def test_instantaneous_step_or_thermal_slew_breach_returns_block(self) -> None:
        instantaneous = {
            "node_id": "PLC-PIPELINE-PUMP-04",
            "tag": "TIC_3012_SP",
            "target_val": 765.0,
            "current_val": 740.0,
            "ramp_rate_sec": 0.0,
            "engineering_units": "CELSIUS",
            "asset_criticality": "HIGH_HIGH_CONSEQUENCE",
            "operating_envelope_id": "ENV-CRACKER-Z3-STEADY",
        }
        thermal_breach = {
            "node_id": "PLC-FURNACE-02",
            "tag": "TIC_4402_SP",
            "target_val": 770.0,
            "current_val": 700.0,
            "ramp_rate_sec": 60.0,
            "engineering_units": "CELSIUS",
            "asset_criticality": "HIGH_HIGH_CONSEQUENCE",
            "operating_envelope_id": "ENV-FURNACE-TUBE-SAFE",
        }

        for payload in (instantaneous, thermal_breach):
            with self.subTest(payload=payload):
                command = self.agent.execute("adjust_plc_setpoint", payload)
                self.assertIsNotNone(command.update["governance_error"])
                self.assertIn("Governance blocked", command.update["messages"][0].content)

        self.assertEqual(self.executions, [])

    def test_safety_interlock_without_physical_key_returns_block(self) -> None:
        byok_client = IndustrialPolicyFakeClient()
        byok_agent = IndustrialAutomationAgent(
            client=byok_client,  # type: ignore[arg-type]
            tools=self.tools,
            provider_key="provider-test-value",
            provider_name="openai",
        )
        payload = {
            "node_id": "SIS-TURB-GEN-01",
            "tag": "ESD_SOLENOID_20_ET",
            "mutation_type": "SET_DIAGNOSTIC_BYPASS",
            "requested_value": "BYPASS_ACTIVE",
            "duration_sec": 1800,
            "physical_key_interlock_verified": False,
            "management_of_change_id": "MOC-2026-SIS-8841",
        }

        state = byok_agent.invoke(
            {
                "messages": [],
                "tool_invocation": ToolInvocation(
                    name="mutate_safety_parameter",
                    arguments=payload,
                    tool_call_id="sis-block-1",
                ),
            }
        )

        self.assertIsNotNone(state["governance_error"])
        self.assertIsNone(state["tool_invocation"])
        self.assertEqual(self.executions, [])
        _, options = byok_client.calls[0]
        self.assertEqual(options["provider_key"], "provider-test-value")
        self.assertEqual(options["provider_name"], "openai")


if __name__ == "__main__":
    unittest.main()
