"""LangGraph state schemas and reducers.

These schemas remain ``TypedDict`` contracts because LangGraph applies reducers to
individual state channels. Pydantic contracts for model input and output live in
``structured_outputs.py``.
"""

import operator
from typing import Annotated, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from polyresearch.models.evidence import (
    Claim,
    EvidencePassage,
    SourceRecord,
    VerificationResult,
)


def override_reducer(current_value, new_value):
    """Append state values unless an explicit override update is provided."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    return operator.add(current_value, new_value)


def merge_evidence_by_id(current_value, new_value):
    """Merge evidence collections without counting duplicate artifacts twice."""
    current_items = current_value or []
    new_items = new_value or []
    artifact_id = lambda item: item.id if hasattr(item, "id") else item["id"]
    merged = {str(artifact_id(item)): item for item in current_items}
    merged.update({str(artifact_id(item)): item for item in new_items})
    return list(merged.values())


class AgentInputState(MessagesState):
    """State accepted at the public graph boundary."""


class AgentState(MessagesState):
    """Main graph state containing messages and research data."""

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    sources: Annotated[list[SourceRecord], merge_evidence_by_id]
    passages: Annotated[list[EvidencePassage], merge_evidence_by_id]
    claims: Annotated[list[Claim], merge_evidence_by_id]
    verification_results: Annotated[list[VerificationResult], merge_evidence_by_id]
    final_report: str


class SupervisorState(TypedDict):
    """State used by the supervisor subgraph."""

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    research_iterations: int = 0
    sources: Annotated[list[SourceRecord], merge_evidence_by_id]
    passages: Annotated[list[EvidencePassage], merge_evidence_by_id]
    claims: Annotated[list[Claim], merge_evidence_by_id]
    verification_results: Annotated[list[VerificationResult], merge_evidence_by_id]


class ResearcherState(TypedDict):
    """State used by an individual researcher subgraph."""

    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    sources: Annotated[list[SourceRecord], merge_evidence_by_id]
    passages: Annotated[list[EvidencePassage], merge_evidence_by_id]
    claims: Annotated[list[Claim], merge_evidence_by_id]
    verification_results: Annotated[list[VerificationResult], merge_evidence_by_id]


class ResearcherOutputState(BaseModel):
    """Typed output emitted by an individual researcher subgraph."""

    sources: list[SourceRecord] = Field(default_factory=list)
    passages: list[EvidencePassage] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    verification_results: list[VerificationResult] = Field(default_factory=list)
