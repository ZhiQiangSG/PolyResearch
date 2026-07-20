"""Typed, passage-level evidence artifacts used throughout a research run."""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


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
    language: str | None = None
    source_type: str = "web"
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    content_hash: str | None = None


class EvidencePassage(BaseModel):
    """Exact original-language text that may be cited by a claim."""

    id: UUID = Field(default_factory=uuid4)
    source_id: UUID
    text: str = Field(min_length=1)
    locator: str
    original_language: str | None = None


class Claim(BaseModel):
    """An atomic, falsifiable proposition extracted from evidence passages."""

    id: UUID = Field(default_factory=uuid4)
    statement: str = Field(min_length=1)
    evidence_passage_ids: list[UUID] = Field(min_length=1)
    original_wording: str | None = None
    extraction_confidence: float = Field(ge=0, le=1)


class EvidenceLink(BaseModel):
    """Relationship between an evidence passage and a claim."""

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


class ClaimExtractionResult(BaseModel):
    """Structured result returned by the claim-extraction model."""

    claims: list[Claim] = Field(default_factory=list)
