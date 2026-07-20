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


class DisagreementDimension(StrEnum):
    """Reasons superficially similar evidence may appear to disagree."""

    TIME_PERIOD = "different_time_periods"
    GEOGRAPHIC_SCOPE = "different_geographic_scope"
    DEFINITION_OR_METHOD = "differing_definitions_or_measurement_methods"
    POPULATION_OR_SAMPLE = "different_populations_or_samples"
    TRANSLATION_AMBIGUITY = "translation_ambiguity"
    GENUINE_CONFLICT = "genuinely_conflicting_evidence"


class DisagreementAssessment(BaseModel):
    """A verifier's explicit assessment of one possible disagreement cause."""

    dimension: DisagreementDimension
    present: bool
    explanation: str = Field(min_length=1)


class DocumentSection(BaseModel):
    """A stable structural region of a fetched source document."""

    heading: str
    first_passage_locator: str
    last_passage_locator: str
    heading_level: int | None = Field(default=None, ge=1, le=6)


class SourceQualityAssessment(BaseModel):
    """Explainable, versioned initial assessment that can be recalculated later."""

    score: float = Field(ge=0, le=1)
    scoring_version: str = Field(min_length=1)
    factor_scores: dict[str, float] = Field(default_factory=dict)
    rationale: list[str] = Field(default_factory=list)


class SourceRecord(BaseModel):
    """A retrieved source with immutable discovery and retrieval provenance."""

    id: UUID = Field(default_factory=uuid4)
    canonical_url: str
    title: str
    publisher: str | None = None
    author: str | None = None
    language: str | None = None
    content_language: str | None = None
    metadata_language: str | None = None
    language_detection_method: str | None = None
    planned_query_language: str | None = None
    language_matches_planned_query: bool | None = None
    source_type: str = "web"
    published_at: datetime | None = None
    updated_at: datetime | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content_hash: str | None = None
    publisher_family: str | None = None
    shared_origin_cluster_id: str | None = None
    near_duplicate_cluster_id: str | None = None
    initial_quality_assessment: SourceQualityAssessment | None = None
    extraction_quality: float | None = Field(default=None, ge=0, le=1)
    extraction_notes: list[str] = Field(default_factory=list)
    document_structure: list[DocumentSection] = Field(default_factory=list)
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

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    text: str = Field(min_length=1)
    original_text_hash: str
    locator: str
    heading: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    character_start: int | None = Field(default=None, ge=0)
    character_end: int | None = Field(default=None, ge=0)
    original_language: str | None = None

    @model_validator(mode="before")
    @classmethod
    def add_original_text_hash(cls, values: Any) -> Any:
        """Bind a passage ID to its exact pre-summary, pre-translation text."""
        if not isinstance(values, dict) or "original_text_hash" in values:
            return values
        import hashlib

        payload = dict(values)
        payload["original_text_hash"] = hashlib.sha256(
            str(payload.get("text", "")).encode("utf-8")
        ).hexdigest()
        return payload

    @model_validator(mode="after")
    def validate_character_range(self) -> "EvidencePassage":
        """Keep character offsets useful and unambiguous when extraction supplies them."""
        if (self.character_start is None) != (self.character_end is None):
            raise ValueError("character_start and character_end must be recorded together")
        if self.character_start is not None and self.character_end <= self.character_start:
            raise ValueError("character_end must be greater than character_start")
        return self


class TranslationRecord(BaseModel):
    """A labeled translation derived from an original evidence passage."""

    model_config = ConfigDict(frozen=True)

    id: UUID = Field(default_factory=uuid4)
    passage_id: UUID
    translated_text: str = Field(min_length=1)
    target_language: str = Field(min_length=1)
    source_original_text_hash: str | None = None
    model_id: str | None = None
    prompt_version: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TranslationDraft(BaseModel):
    """One model-produced translation before immutable provenance is attached."""

    translated_text: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)


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
    atomic_proposition: str | None = None
    entities: list["ClaimEntity"] = Field(default_factory=list)
    quantities: list["ClaimQuantity"] = Field(default_factory=list)
    dates: list["ClaimDate"] = Field(default_factory=list)
    locations: list["ClaimLocation"] = Field(default_factory=list)
    scope: "ClaimScope | None" = None
    qualifiers: list[str] = Field(default_factory=list)
    modality: str | None = None
    claim_cluster_id: UUID | None = None
    claim_cluster_method: str | None = None
    claim_cluster_confidence: float | None = Field(default=None, ge=0, le=1)


class ClaimEntity(BaseModel):
    """An entity retaining all source forms and any cautious resolution outcome."""

    original_name: str = Field(min_length=1)
    normalized_name: str | None = None
    entity_type: str | None = None
    aliases: list[str] = Field(default_factory=list)
    script_variants: list[str] = Field(default_factory=list)
    transliterations: list[str] = Field(default_factory=list)
    historical_names: list[str] = Field(default_factory=list)
    resolved_entity_id: UUID | None = None
    resolution_status: Literal["resolved", "ambiguous", "unresolved"] = "unresolved"
    resolution_confidence: float | None = Field(default=None, ge=0, le=1)
    resolution_rationale: str | None = None


class ClaimQuantity(BaseModel):
    """A quantity that retains its source rendering before later normalization."""

    original_value: str = Field(min_length=1)
    normalized_value: str | None = None
    unit: str | None = None
    normalized_unit: str | None = None
    currency_code: str | None = None
    normalization_status: Literal["normalized", "original_retained"] = "original_retained"
    normalization_notes: list[str] = Field(default_factory=list)


class ClaimDate(BaseModel):
    """A source date expression and, where safe, an ISO-normalized derivative."""

    original_value: str = Field(min_length=1)
    normalized_value: str | None = None
    normalization_status: Literal["normalized", "original_retained"] = "original_retained"
    normalization_notes: list[str] = Field(default_factory=list)


class ClaimLocation(BaseModel):
    """A location as written in evidence, preserving uncertain normalization."""

    original_name: str = Field(min_length=1)
    normalized_name: str | None = None


class ClaimScope(BaseModel):
    """Boundaries that prevent an atomic claim being generalized beyond evidence."""

    description: str = Field(min_length=1)
    temporal: str | None = None
    geographic: str | None = None
    population: str | None = None


class ClaimExtractionDraft(BaseModel):
    """Strict Qwen output for one atomic claim grounded in selected passages."""

    id: UUID = Field(default_factory=uuid4)
    atomic_proposition: str = Field(min_length=1)
    original_wording: str | None = None
    normalized_statement: str = Field(min_length=1)
    entities: list[ClaimEntity] = Field(default_factory=list)
    quantities: list[ClaimQuantity] = Field(default_factory=list)
    dates: list[ClaimDate] = Field(default_factory=list)
    locations: list[ClaimLocation] = Field(default_factory=list)
    scope: ClaimScope
    qualifiers: list[str] = Field(default_factory=list)
    modality: Literal[
        "asserted", "reported", "estimated", "possible", "required", "prohibited", "unknown"
    ]
    extraction_confidence: float = Field(ge=0, le=1)
    evidence_passage_ids: list[UUID] = Field(min_length=1)


class EvidenceLink(BaseModel):
    """An immutable assertion or verification relationship between evidence and a claim."""

    id: UUID = Field(default_factory=uuid4)
    claim_id: UUID
    passage_id: UUID
    relationship: Literal["supports", "contradicts", "contextualizes"]
    rationale: str | None = None
    origin: Literal["claim_extraction", "verification"] = "claim_extraction"
    verification_result_id: UUID | None = None


class VerificationResult(BaseModel):
    """An immutable, versioned verification judgement for a single claim."""

    id: UUID = Field(default_factory=uuid4)
    claim_id: UUID
    status: VerificationStatus
    confidence: float = Field(ge=0, le=1)
    rationale: str
    evidence_link_ids: list[UUID] = Field(default_factory=list)
    disagreement_assessments: list[DisagreementAssessment] = Field(default_factory=list)
    confidence_factors: dict[str, float] = Field(default_factory=dict)
    independent_source_count: int = Field(default=0, ge=0)
    attempt_number: int = Field(default=1, ge=1)
    supersedes_verification_result_id: UUID | None = None
    trigger: Literal["initial_verification", "conflict_resolution"] = "initial_verification"
    verifier_model_id: str = Field(default="unrecorded", min_length=1)
    verifier_prompt_version: str = Field(default="unrecorded", min_length=1)
    verified_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ClaimClusterVerificationDraft(BaseModel):
    """Qwen judgement for a deterministic cluster of related claims."""

    cluster_id: UUID
    cluster_rationale: str = Field(min_length=1)
    claim_assessments: list["ClaimVerificationAssessment"] = Field(min_length=1)
    disagreement_assessments: list[DisagreementAssessment] = Field(min_length=6)

    @model_validator(mode="after")
    def require_all_disagreement_dimensions(self) -> "ClaimClusterVerificationDraft":
        dimensions = {assessment.dimension for assessment in self.disagreement_assessments}
        required = set(DisagreementDimension)
        if dimensions != required or len(self.disagreement_assessments) != len(required):
            raise ValueError("Every disagreement dimension must be assessed exactly once")
        return self


class ClaimVerificationAssessment(BaseModel):
    """One claim-level classification reached with cluster-level context."""

    claim_id: UUID
    status: VerificationStatus
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)
    evidence_assessments: list["VerificationEvidenceAssessment"] = Field(
        default_factory=list
    )


class VerificationEvidenceAssessment(BaseModel):
    """Verifier classification of one input evidence link for a claim."""

    evidence_link_id: UUID
    relationship: Literal["supports", "contradicts", "contextualizes"]
    rationale: str = Field(min_length=1)


class ClaimClusterVerificationResult(BaseModel):
    """Structured verification output for deterministic claim clusters."""

    clusters: list[ClaimClusterVerificationDraft] = Field(default_factory=list)


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


class UnresolvedDisagreement(BaseModel):
    """A durable account of evidence that remains materially unresolved.

    This is intentionally separate from report prose: consumers can render or
    inspect the conflict without inferring it from a verifier rationale.
    """

    cluster_id: UUID
    claim_ids: list[UUID] = Field(min_length=1)
    conflicting_claims: list[str] = Field(min_length=1)
    verification_statuses: dict[str, VerificationStatus] = Field(default_factory=dict)
    disagreement_assessments: list[DisagreementAssessment] = Field(default_factory=list)
    why_it_may_conflict: list[str] = Field(min_length=1)
    evidence_needed: list[str] = Field(min_length=1)


class ReportBundle(BaseModel):
    """The exported, auditable report artifacts for one research run."""

    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    markdown: str | None = None
    html: str | None = None
    provenance_json: dict[str, Any] = Field(default_factory=dict)
    unresolved_disagreements: list[UnresolvedDisagreement] = Field(default_factory=list)
    qa_issues: list[ReportQaIssue] = Field(default_factory=list)
    qa_passed: bool = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ClaimExtractionResult(BaseModel):
    """Structured result returned by the claim-extraction model."""

    claims: list[ClaimExtractionDraft] = Field(default_factory=list)
