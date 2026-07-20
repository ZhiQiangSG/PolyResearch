"""LangGraph state schemas and reducers.

These schemas remain ``TypedDict`` contracts because LangGraph applies reducers to
individual state channels. Pydantic contracts for model input and output live in
``structured_outputs.py``.
"""

import operator
from typing import Annotated, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel
from typing_extensions import TypedDict


def override_reducer(current_value, new_value):
    """Append state values unless an explicit override update is provided."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    return operator.add(current_value, new_value)


class AgentInputState(MessagesState):
    """State accepted at the public graph boundary."""


class AgentState(MessagesState):
    """Main graph state containing messages and research data."""

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str


class SupervisorState(TypedDict):
    """State used by the supervisor subgraph."""

    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []


class ResearcherState(TypedDict):
    """State used by an individual researcher subgraph."""

    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []


class ResearcherOutputState(BaseModel):
    """Typed output emitted by an individual researcher subgraph."""

    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
