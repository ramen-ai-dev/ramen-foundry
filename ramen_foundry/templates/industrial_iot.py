"""Governed industrial control and safety-system actuation tools."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command
from ramen_ai import RamenClient

from ramen_foundry.core import RamenToolNode, ToolInvocation

from ._security import validate_provider_configuration, validate_tool_registry

INDUSTRIAL_IOT_ACTUATION_INVARIANCE_BUNDLE_ID = (
    "ramen__industrial_iot_actuation_invariance"
)
ManipulationActionType = Literal[
    "PICK_AND_PLACE", "INSERT_TOOL", "APPLY_FORCE", "POUR_LIQUID", "WIPE_SURFACE"
]


class ManipulationDispatchPayload(TypedDict):
    """Resolved host-tool contract for one robotic manipulation dispatch."""

    robot_id: str
    action_type: ManipulationActionType
    target_object: str
    destination_target: str
    commanded_velocity_mps: float
    commanded_force_nm: float
    human_proximity_meters: float
    active_hazard_flags: list[str]
    scene_context_id: str


INDUSTRIAL_AUTOMATION_TOOL_NAMES = frozenset(
    {
        "adjust_plc_setpoint",
        "mutate_safety_parameter",
        "dispatch_manipulation",
    }
)


class IndustrialAutomationState(TypedDict, total=False):
    """State accepted and returned by the compiled industrial workflow."""

    tool_invocation: ToolInvocation | Mapping[str, Any] | None
    governance_error: str | None
    messages: Annotated[list[Any], add_messages]


class IndustrialAutomationAgent:
    """Govern PLC setpoints and represented SIS maintenance mutations.

    ``adjust_plc_setpoint`` binds the target, current telemetry, ramp duration,
    engineering units, asset criticality, and operating-envelope identity for
    safety-envelope, slew-rate, and telemetry-degradation evaluation.

    ``mutate_safety_parameter`` binds the requested mutation, finite duration,
    target-local physical-key evidence, and management-of-change identity for
    SIS isolation and zero-autonomous-software-override evaluation.

    ``dispatch_manipulation`` binds the robot identity, action type, target and
    destination, commanded velocity and force, human proximity, active hazards,
    and scene context for embodied physical-safety evaluation.

    Host applications provide the actual LangChain tools and retain physical
    control of PLC/SIS infrastructure. Every invocation is evaluated against
    the immutable Industrial IoT Actuation Invariance bundle before execution.
    """

    def __init__(
        self,
        *,
        client: RamenClient,
        tools: Mapping[str, BaseTool],
        provider_key: str | None = None,
        provider_name: str | None = None,
    ) -> None:
        validate_provider_configuration(
            provider_key,
            provider_name,
            template_name=type(self).__name__,
        )
        registered = validate_tool_registry(
            tools,
            allowed_names=INDUSTRIAL_AUTOMATION_TOOL_NAMES,
            template_name=type(self).__name__,
        )
        self._tool_names = tuple(sorted(registered))
        self._guard = RamenToolNode(
            client=client,
            tools=registered,
            llm_node=END,
            bundle_ids=[INDUSTRIAL_IOT_ACTUATION_INVARIANCE_BUNDLE_ID],
            provider_key=provider_key,
            provider_name=provider_name,
        )
        self._graph = self._build_graph()

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Return the registered industrial capability names."""

        return self._tool_names

    @property
    def graph(self) -> Any:
        """Return the compiled LangGraph workflow for host composition."""

        return self._graph

    def execute(
        self,
        tool_name: str,
        payload: Mapping[str, Any],
        *,
        tool_call_id: str | None = None,
    ) -> Command:
        """Evaluate and directly execute one resolved industrial action."""

        invocation = ToolInvocation(
            name=tool_name,
            arguments=dict(payload),
            tool_call_id=tool_call_id or f"{tool_name}-direct",
        )
        return self._guard({"tool_invocation": invocation})

    def invoke(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Route industrial state through the compiled governance workflow."""

        return self._graph.invoke(dict(state))

    def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Route industrial state through the compiled governance workflow."""

        return self.invoke(state)

    def _build_graph(self) -> Any:
        workflow = StateGraph(IndustrialAutomationState)
        workflow.add_node("governed_industrial_automation_action", self._guard)
        workflow.add_edge(START, "governed_industrial_automation_action")
        workflow.add_edge("governed_industrial_automation_action", END)
        return workflow.compile()
