"""Typed, passage-level evidence artifacts used throughout a research run."""

from datetime import date, datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VerificationStatus(StrEnum):
    """Permitted outcomes of claim verification."""

    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    OUTDATED = "outdated"
    NOT_COMPARABLE = "not_comparable"


class SourceRecord(BaseModel):
    """A retrieved source with immutable discovery and retrieval provenance."""

    id: UUID = Field(default_factory=uuid4)
    canonical_url: str
    title: str
    publisher: str | None = None
    author: str | None = None
    language: str | None = None
    planned_query_language: str | None = None
    source_type: str = "web"
    published_at: datetime | None = None
    updated_at: datetime | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content_hash: str | None = None
    extraction_quality: float | None = Field(default=None, ge=0, le=1)
    extraction_notes: list[str] = Field(default_factory=list)
    research_unit_id: UUID | None = None
    discovered_url: str | None = None
    redirect_chain: list[str] = Field(default_factory=list)


class SourceVersion(BaseModel):
    """Immutable fetched-content version for a source record."""

    id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    version_number: int = Field(ge=1)
    content_hash: str
    raw_content: str
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    http_metadata: dict[str, Any] = Field(default_factory=dict)
    extraction_method: str = "provider_content"
    extraction_quality: float | None = Field(default=None, ge=0, le=1)


class EvidencePassage(BaseModel):
    """Exact original-language text that may be cited by a claim."""

    id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    text: str = Field(min_length=1)
    locator: str
    original_language: str | None = None


class TranslationRecord(BaseModel):
    """A labeled translation derived from an original evidence passage."""

    id: UUID = Field(default_factory=uuid4)
    passage_id: UUID
    translated_text: str = Field(min_length=1)
    target_language: str = Field(min_length=1)
    model_id: str | None = None
    prompt_version: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class QueryRecord(BaseModel):
    """Provenance for a discovery query and provider-routing decision."""

    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    research_unit_id: UUID | None = None
    query: str = Field(min_length=1)
    language: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    locale: str | None = None
    target_source_type: str | None = None
    rationale: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    result_rank: int | None = Field(default=None, ge=1)
    result_url: str | None = None
    fallback_from: str | None = None
    failure: str | None = None
    executed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ProvenanceAttachment(BaseModel):
    """Immutable raw output retained for audit, never as a reasoning artifact."""

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    provider: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    raw_output: str
    content_hash: str
    captured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="before")
    @classmethod
    def add_content_hash(cls, values: Any) -> Any:
        """Derive a stable content hash when callers provide raw output only."""
        if not isinstance(values, dict) or "content_hash" in values:
            return values
        import hashlib

        payload = dict(values)
        payload["content_hash"] = hashlib.sha256(
            str(payload.get("raw_output", "")).encode("utf-8")
        ).hexdigest()
        return payload


class Claim(BaseModel):
    """An atomic, falsifiable proposition extracted from evidence passages."""

    id: UUID = Field(default_factory=uuid4)
    statement: str = Field(min_length=1)
    evidence_passage_ids: list[UUID] = Field(min_length=1)
    original_wording: str | None = None
    extraction_confidence: float = Field(ge=0, le=1)


class EvidenceLink(BaseModel):
    """Relationship between an evidence passage and a claim."""

    id: UUID = Field(default_factory=uuid4)
    claim_id: UUID
    passage_id: UUID
    relationship: Literal["supports", "contradicts", "contextualizes"]
    rationale: str | None = None


class VerificationResult(BaseModel):
    """A conservative verification judgement for a single claim."""

    id: UUID = Field(default_factory=uuid4)
    claim_id: UUID
    status: VerificationStatus
    confidence: float = Field(ge=0, le=1)
    rationale: str
    evidence_link_ids: list[UUID] = Field(default_factory=list)


class ReportStatement(BaseModel):
    """A rendered factual report statement linked to verified claims."""

    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    rendered_text: str = Field(min_length=1)
    claim_ids: list[UUID] = Field(min_length=1)
    citation_ids: list[UUID] = Field(default_factory=list)
    verification_status: VerificationStatus
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ReportStatementDraft(BaseModel):
    """Model-selected report wording that must resolve to known claim IDs."""

    rendered_text: str = Field(min_length=1)
    claim_ids: list[UUID] = Field(min_length=1)


class ReportDraft(BaseModel):
    """Structured report content before statement/citation records are created."""

    title: str = Field(default="Research report", min_length=1)
    statements: list[ReportStatementDraft] = Field(default_factory=list)


class ReportQaIssue(BaseModel):
    """A deterministic report-integrity finding produced before rendering."""

    code: str
    severity: Literal["error", "warning"]
    message: str
    statement_id: UUID | None = None


class ReportBundle(BaseModel):
    """The exported, auditable report artifacts for one research run."""

    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    markdown: str | None = None
    html: str | None = None
    provenance_json: dict[str, Any] = Field(default_factory=dict)
    qa_issues: list[ReportQaIssue] = Field(default_factory=list)
    qa_passed: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ClaimExtractionResult(BaseModel):
    """Structured result returned by the claim-extraction model."""

    claims: list[Claim] = Field(default_factory=list)
