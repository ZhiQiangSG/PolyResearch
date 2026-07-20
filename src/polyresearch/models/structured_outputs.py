"""Pydantic contracts used for model structured output and tool calls."""

from pydantic import BaseModel, Field


class ConductResearch(BaseModel):
    """Request one bounded, evidence-producing research unit."""

    task: "EvidenceTask" = Field(
        description=(
            "A typed evidence task selected from the persisted multilingual plan; "
            "never an open-ended prose-summary request."
        ),
    )


class EvidenceTask(BaseModel):
    """The bounded evidence target assigned to one parallel research unit."""

    subquestion: str = Field(min_length=1)
    language: str = Field(min_length=1)
    target_source_type: str = Field(min_length=1)
    evidence_goal: str = Field(min_length=1)
    query_rationale: str = Field(min_length=1)


class ResearchComplete(BaseModel):
    """Signal that a research phase is complete."""


class ClarifyWithUser(BaseModel):
    """Structured result for deciding whether research scope needs clarification."""

    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question that clarifies the report scope.",
    )
    verification: str = Field(
        description=(
            "A message confirming that research will begin after the user provides "
            "the necessary information."
        ),
    )


class ResearchQuestion(BaseModel):
    """Structured research brief generated from the user's request."""

    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )
