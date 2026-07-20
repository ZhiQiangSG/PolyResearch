"""Typed metadata for a durable research run."""

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ResearchRun(BaseModel):
    """The durable identity and lifecycle metadata of a research run."""

    id: UUID = Field(default_factory=uuid4)
    question: str = Field(min_length=1)
    output_language: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: Literal["active", "completed", "failed"] = "active"


class ResearchEntity(BaseModel):
    """An entity and its language-specific names used during discovery."""

    model_config = ConfigDict(extra="forbid")

    canonical_name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    transliterations: list[str] = Field(default_factory=list)
    native_script_variants: list[str] = Field(default_factory=list)


class ResearchLanguage(BaseModel):
    """A ranked research language with an explicit retrieval allocation."""

    model_config = ConfigDict(extra="forbid")

    language: str = Field(min_length=1)
    priority: int = Field(ge=1)
    query_budget: int = Field(ge=1)
    expected_unique_value: str = Field(min_length=1)
    selection_rationale: str = Field(min_length=1)
    expected_source_types: list[str] = Field(default_factory=list)
    preferred_domains: list[str] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    """Reproducible research-plan decision recorded for a run."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    subquestions: list[str] = Field(default_factory=list)
    entities: list[ResearchEntity] = Field(default_factory=list)
    ranked_languages: list[ResearchLanguage] = Field(default_factory=list)
    language_rationale: dict[str, str] = Field(default_factory=dict)
    query_variants: dict[str, list[str]] = Field(default_factory=dict)
    target_source_types: list[str] = Field(default_factory=list)
    target_domains: list[str] = Field(default_factory=list)
    anticipated_conflict_dimensions: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_id: str | None = None
    prompt_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
