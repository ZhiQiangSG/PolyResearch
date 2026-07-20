"""Pydantic contracts used for model structured output and tool calls."""

from pydantic import BaseModel, Field


class ConductResearch(BaseModel):
    """Request focused research on one clearly scoped topic."""

    research_topic: str = Field(
        description=(
            "The topic to research. It must be a single topic described in high "
            "detail (at least a paragraph)."
        ),
    )


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
