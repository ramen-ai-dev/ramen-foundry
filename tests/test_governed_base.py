"""Tests for the GovernedToolAgent base class."""

from __future__ import annotations

import inspect
import os
import unittest
from collections.abc import Mapping
from typing import Any
from unittest import mock

from langchain_core.tools import tool

from ramen_foundry import GovernedToolAgent, ToolInvocation
from tests.gate_fixtures import FakeClient

BUNDLE = "ramen__test_bundle"


class _Agent(GovernedToolAgent):
    bundle_id = BUNDLE
    allowed_tool_names = frozenset({"do_thing"})
    node_name = "governed_test_action"

    def __init__(self, *, preflight: Any = None, **kwargs: Any) -> None:
        self._preflight_impl = preflight
        super().__init__(**kwargs)

    def _preflight(self, invocation: ToolInvocation) -> str | None:
        if self._preflight_impl is None:
            return None
        result: str | None = self._preflight_impl(invocation)
        return result


class GovernedToolAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.executions: list[dict[str, Any]] = []

        @tool
        def do_thing(value: str) -> str:
            """Do a governed thing."""

            self.executions.append({"value": value})
            return "done"

        self.tool = do_thing

    def _agent(self, client: FakeClient, **kwargs: Any) -> _Agent:
        return _Agent(client=client, tools={"do_thing": self.tool}, **kwargs)  # type: ignore[arg-type]

    def test_base_class_cannot_be_used_directly(self) -> None:
        with self.assertRaises(TypeError):
            GovernedToolAgent(client=FakeClient(), tools={"do_thing": self.tool})  # type: ignore[arg-type]

    def test_binds_exactly_one_bundle_and_no_policy_ids(self) -> None:
        client = FakeClient()
        command = self._agent(client).execute("do_thing", {"value": "x"})

        self.assertIsNone(command.update["governance_error"])
        self.assertEqual(self.executions, [{"value": "x"}])
        _, options = client.calls[0]
        self.assertEqual(options["bundle_ids"], [BUNDLE])
        self.assertIsNone(options["policy_ids"])

    def test_shadow_mode_cannot_be_enabled(self) -> None:
        client = FakeClient(allowed=False)
        with mock.patch.dict(os.environ, {"RAMEN_SHADOW_MODE": "true"}):
            agent = self._agent(client)
            command = agent.execute("do_thing", {"value": "x"})

        self.assertFalse(agent._guard.shadow_mode)
        self.assertEqual(self.executions, [])
        self.assertIsNotNone(command.update["governance_error"])
        self.assertNotIn("shadow_mode", inspect.signature(GovernedToolAgent.__init__).parameters)
        with self.assertRaises(TypeError):
            self._agent(client, shadow_mode=True)

    def test_registry_is_restricted_to_declared_tools(self) -> None:
        @tool
        def other_tool(value: str) -> str:
            """Not allowed."""

            return value

        with self.assertRaises(ValueError):
            _Agent(client=FakeClient(), tools={"other_tool": other_tool})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            _Agent(client=FakeClient(), tools={})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            _Agent(client=FakeClient(), tools={"do_thing": other_tool})  # type: ignore[arg-type]

    def test_provider_key_and_name_must_be_given_together(self) -> None:
        with self.assertRaises(ValueError):
            self._agent(FakeClient(), provider_key="k")

    def test_preflight_reason_blocks_without_contacting_the_backend(self) -> None:
        client = FakeClient()
        agent = self._agent(client, preflight=lambda invocation: "local rule says no")

        command = agent.execute("do_thing", {"value": "x"})

        self.assertEqual(command.update["governance_error"], "local rule says no")
        self.assertEqual(client.calls, [])
        self.assertEqual(self.executions, [])
        self.assertIn("Governance blocked", command.update["messages"][0].content)

    def test_preflight_exception_fails_closed(self) -> None:
        def boom(invocation: ToolInvocation) -> str | None:
            raise RuntimeError("rule store offline")

        client = FakeClient()
        command = self._agent(client, preflight=boom).execute("do_thing", {"value": "x"})

        self.assertIn("rule store offline", command.update["governance_error"])
        self.assertEqual(client.calls, [])
        self.assertEqual(self.executions, [])

    def test_compiled_graph_runs_preflight_too(self) -> None:
        client = FakeClient()
        agent = self._agent(client, preflight=lambda invocation: "blocked in graph")
        state = agent.invoke(
            {
                "messages": [],
                "tool_invocation": ToolInvocation(
                    name="do_thing", arguments={"value": "x"}, tool_call_id="c1"
                ),
            }
        )

        self.assertEqual(state["governance_error"], "blocked in graph")
        self.assertEqual(client.calls, [])
        self.assertEqual(self.executions, [])

    def test_clean_preflight_still_requires_a_verified_allow(self) -> None:
        client = FakeClient(allowed=False)
        command = self._agent(client, preflight=lambda invocation: None).execute(
            "do_thing", {"value": "x"}
        )

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(self.executions, [])
        self.assertIsNotNone(command.update["governance_error"])

    def test_tool_names_and_call_protocol(self) -> None:
        agent = self._agent(FakeClient())
        self.assertEqual(agent.tool_names, ("do_thing",))
        state: Mapping[str, Any] = agent(
            {
                "messages": [],
                "tool_invocation": {"name": "do_thing", "arguments": {"value": "y"}, "tool_call_id": "c2"},
            }
        )
        self.assertIsNone(state["governance_error"])
        self.assertEqual(self.executions, [{"value": "y"}])


if __name__ == "__main__":
    unittest.main()
