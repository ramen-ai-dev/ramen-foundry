"""Human-review resume workflow template built with LangGraph and ramen-ai."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph, add_messages
from pydantic import BaseModel, Field
from ramen_ai import RamenClient

from ramen_foundry.core import RamenGovernedNode

EU_AI_ACT_PROXY_BIAS_POLICY_ID = "0d5ed2af-5e98-4a8c-92c3-dea26c07bf9a"


class ResumeScreeningRequest(BaseModel):
    """Inputs for an assistive, human-reviewed resume assessment."""

    resume_text: str = Field(min_length=1)
    job_description: str = Field(min_length=1)


class ResumeScreeningResult(BaseModel):
    """Governed review output that must be reviewed by a human decision-maker."""

    report: str | None
    governance_error: str | None
    requires_human_review: bool = True
    policy_id: str = EU_AI_ACT_PROXY_BIAS_POLICY_ID


class ResumeScreeningState(TypedDict, total=False):
    """Private state for the resume-review graph."""

    resume_text: str
    job_description: str
    reviewer_plan: str
    governed_prompt: str
    governed_content: str | None
    governance_error: str | None
    governed_evaluation: Any
    messages: Annotated[list[Any], add_messages]


class ResumeScreeningAgent:
    """Create a governed, evidence-focused resume review for human assessment.

    The template never returns a hiring decision, candidate ranking, or automated
    eligibility outcome. It uses the supplied LangChain chat model only to draft
    a neutral review plan, then uses ``RamenGovernedNode`` to release the final
    report under the fixed EU AI Act Proxy Bias Policy.
    """

    def __init__(
        self,
        *,
        llm: BaseChatModel,
        client: RamenClient,
        provider_key: str | None = None,
        provider_name: str | None = None,
    ) -> None:
        self._llm = llm
        self._graph = self._build_graph(
            client=client,
            provider_key=provider_key,
            provider_name=provider_name,
        )

    def screen(self, request: ResumeScreeningRequest) -> ResumeScreeningResult:
        """Return a governed review for a human reviewer, never a hiring decision."""

        state = self._graph.invoke(
            {
                "resume_text": request.resume_text,
                "job_description": request.job_description,
            }
        )
        error = state.get("governance_error")
        if error:
            return ResumeScreeningResult(report=None, governance_error=str(error))

        report = state.get("governed_content")
        if not isinstance(report, str):
            return ResumeScreeningResult(
                report=None,
                governance_error="Governed review completed without released content.",
            )
        return ResumeScreeningResult(report=report, governance_error=None)

    def _build_graph(
        self,
        *,
        client: RamenClient,
        provider_key: str | None,
        provider_name: str | None,
    ) -> Any:
        workflow = StateGraph(ResumeScreeningState)
        workflow.add_node("draft_review_prompt", self._draft_review_prompt)
        workflow.add_node(
            "governed_resume_review",
            RamenGovernedNode(
                client=client,
                policy_ids=[EU_AI_ACT_PROXY_BIAS_POLICY_ID],
                provider_key=provider_key,
                provider_name=provider_name,
            ),
        )
        workflow.add_edge(START, "draft_review_prompt")
        workflow.add_edge("draft_review_prompt", "governed_resume_review")
        workflow.add_edge("governed_resume_review", END)
        return workflow.compile()

    def _draft_review_prompt(self, state: ResumeScreeningState) -> dict[str, str]:
        response = self._llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Create a neutral evidence-collection plan for a human resume reviewer. "
                        "Do not rank candidates, recommend hiring or rejection, infer protected traits, "
                        "or make an employment decision."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Job description:\n{state['job_description']}\n\n"
                        f"Resume:\n{state['resume_text']}"
                    )
                ),
            ]
        )
        plan = str(response.content)
        prompt = (
            "Produce an evidence-focused resume review for a human decision-maker. "
            "Do not make or recommend a hiring, rejection, ranking, or eligibility decision. "
            "Do not infer or discuss protected characteristics. Identify job-relevant evidence, "
            "missing information, and questions for the human reviewer.\n\n"
            f"Job description:\n{state['job_description']}\n\n"
            f"Resume:\n{state['resume_text']}\n\n"
            f"Neutral review plan:\n{plan}"
        )
        return {"reviewer_plan": plan, "governed_prompt": prompt}
