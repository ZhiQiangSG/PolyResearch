"""Typed metadata for a durable research run."""

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ResearchRun(BaseModel):
    """The durable identity and lifecycle metadata of a research run."""

    id: UUID = Field(default_factory=uuid4)
    question: str = Field(min_length=1)
    output_language: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: Literal["active", "completed", "failed"] = "active"


class ResearchPlan(BaseModel):
    """Reproducible research-plan decision recorded for a run."""

    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    subquestions: list[str] = Field(default_factory=list)
    language_rationale: dict[str, str] = Field(default_factory=dict)
    query_variants: dict[str, list[str]] = Field(default_factory=dict)
    target_source_types: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_id: str | None = None
    prompt_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
