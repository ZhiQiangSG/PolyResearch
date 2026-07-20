"""Typed metadata for a durable research run."""

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class AtomicSubquestion(BaseModel):
    """One independently answerable unit of a research plan."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1)
    answer_scope: str = Field(
        min_length=1,
        description="The precise fact, comparison, or causal relationship to establish.",
    )


class LanguageSelectionAssessment(BaseModel):
    """Recorded basis for choosing one research language over another."""

    model_config = ConfigDict(extra="forbid")

    place_and_institutional_jurisdiction: str = Field(min_length=1)
    primary_actors_and_official_records: str = Field(min_length=1)
    scholarly_technical_and_media_ecosystems: str = Field(min_length=1)
    diasporic_or_regional_coverage: str = Field(min_length=1)
    primary_source_availability: str = Field(min_length=1)
    marginal_information_gain: str = Field(min_length=1)


class ResearchLanguage(BaseModel):
    """A ranked research language with an explicit retrieval allocation."""

    model_config = ConfigDict(extra="forbid")

    language: str = Field(min_length=1)
    priority: int = Field(ge=1)
    query_budget: int = Field(ge=1)
    expected_unique_value: str = Field(min_length=1)
    selection_rationale: str = Field(min_length=1)
    selection_assessment: LanguageSelectionAssessment
    expected_source_types: list[str] = Field(min_length=1)
    preferred_domains: list[str] = Field(default_factory=list)


class EvidenceGap(BaseModel):
    """A retrieval shortfall that may justify expanding language coverage."""

    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    affected_subquestion: str = Field(min_length=1)
    missing_evidence: str = Field(min_length=1)
    languages_considered: list[str] = Field(default_factory=list)


class LanguageExpansionDecision(BaseModel):
    """Post-retrieval decision to add languages only when gaps warrant it."""

    model_config = ConfigDict(extra="forbid")

    should_add_languages: bool
    rationale: str = Field(min_length=1)
    evidence_gaps: list[EvidenceGap] = Field(default_factory=list)
    additional_languages: list[ResearchLanguage] = Field(default_factory=list)
    additional_query_variants: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_addition(self) -> "LanguageExpansionDecision":
        if self.should_add_languages and not self.additional_languages:
            raise ValueError(
                "additional_languages is required when should_add_languages is true"
            )
        if not self.should_add_languages and self.additional_languages:
            raise ValueError(
                "additional_languages must be empty when should_add_languages is false"
            )
        languages = {language.language for language in self.additional_languages}
        if self.should_add_languages and languages - set(self.additional_query_variants):
            raise ValueError(
                "additional_query_variants is required for every additional language"
            )
        return self


class ResearchPlan(BaseModel):
    """Reproducible research-plan decision recorded for a run."""

    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    subquestions: list[AtomicSubquestion] = Field(min_length=1)
    entities: list[ResearchEntity] = Field(default_factory=list)
    ranked_languages: list[ResearchLanguage] = Field(min_length=1)
    language_rationale: dict[str, str] = Field(default_factory=dict)
    query_variants: dict[str, list[str]] = Field(default_factory=dict)
    target_source_types: list[str] = Field(default_factory=list)
    target_domains: list[str] = Field(default_factory=list)
    anticipated_conflict_dimensions: list[str] = Field(default_factory=list)
    post_retrieval_decision: LanguageExpansionDecision | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_id: str | None = None
    prompt_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_language_plan(self) -> "ResearchPlan":
        """Ensure ranked languages have non-duplicated, executable query plans."""
        languages = [language.language for language in self.ranked_languages]
        priorities = [language.priority for language in self.ranked_languages]
        if len(languages) != len(set(languages)):
            raise ValueError("ranked_languages must not repeat a language")
        if len(priorities) != len(set(priorities)):
            raise ValueError("ranked_languages must not repeat a priority")
        if priorities != sorted(priorities):
            raise ValueError("ranked_languages must be ordered by ascending priority")
        missing_rationales = set(languages) - set(self.language_rationale)
        if missing_rationales:
            raise ValueError(
                "language_rationale is required for every ranked language: "
                f"{sorted(missing_rationales)}"
            )
        if languages:
            missing_queries = set(languages) - set(self.query_variants)
            if missing_queries:
                raise ValueError(
                    "query_variants is required for every ranked language: "
                    f"{sorted(missing_queries)}"
                )
        if any(
            not queries or any(not query.strip() for query in queries)
            for queries in self.query_variants.values()
        ):
            raise ValueError("each query_variants entry must contain non-empty queries")
        return self
