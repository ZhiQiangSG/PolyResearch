"""Persistence boundary for PolyResearch's typed evidence ledger."""

from abc import ABC, abstractmethod
from typing import Sequence
from uuid import UUID

from polyresearch.models import (
    Claim,
    EvidenceLink,
    EvidencePassage,
    ProvenanceAttachment,
    QueryRecord,
    ReportBundle,
    ReportStatement,
    ResearchPlan,
    ResearchRun,
    SourceRecord,
    SourceVersion,
    TranslationRecord,
    VerificationResult,
)


class RepositoryNotFoundError(LookupError):
    """Raised when a requested run or evidence artifact does not exist."""


class ArtifactConflictError(ValueError):
    """Raised when an immutable artifact ID is reused with different content."""


class EvidenceRepository(ABC):
    """Abstract storage interface for a run's typed evidence artifacts.

    Implementations append immutable evidence records. Re-submitting an identical
    artifact is idempotent; reusing an ID with materially different content raises
    :class:`ArtifactConflictError`. Storage implementations must preserve the
    supplied artifact IDs and relationships rather than generating replacement IDs.
    """

    @abstractmethod
    async def create_run(self, run: ResearchRun) -> None:
        """Create a research run, or no-op when the identical run already exists."""

    @abstractmethod
    async def get_run(self, run_id: UUID) -> ResearchRun:
        """Return a research run or raise :class:`RepositoryNotFoundError`."""

    @abstractmethod
    async def append_research_plans(
        self, run_id: UUID, plans: Sequence[ResearchPlan]
    ) -> None:
        """Persist reproducible research-planning decisions for a run."""

    @abstractmethod
    async def append_query_records(
        self, run_id: UUID, queries: Sequence[QueryRecord]
    ) -> None:
        """Persist provider-routing and discovery-query provenance."""

    @abstractmethod
    async def append_provenance_attachments(
        self, run_id: UUID, attachments: Sequence[ProvenanceAttachment]
    ) -> None:
        """Persist immutable raw tool outputs as audit-only attachments."""

    @abstractmethod
    async def append_sources(
        self, run_id: UUID, sources: Sequence[SourceRecord]
    ) -> None:
        """Persist immutable source records associated with a run."""

    @abstractmethod
    async def append_source_versions(
        self, run_id: UUID, versions: Sequence[SourceVersion]
    ) -> None:
        """Persist immutable fetched-content versions for run sources."""

    @abstractmethod
    async def append_passages(
        self, run_id: UUID, passages: Sequence[EvidencePassage]
    ) -> None:
        """Persist immutable original-language passages associated with a run."""

    @abstractmethod
    async def append_translations(
        self, run_id: UUID, translations: Sequence[TranslationRecord]
    ) -> None:
        """Persist translations as linked derivatives of evidence passages."""

    @abstractmethod
    async def append_claims(self, run_id: UUID, claims: Sequence[Claim]) -> None:
        """Persist atomic claims associated with a run."""

    @abstractmethod
    async def append_evidence_links(
        self, run_id: UUID, evidence_links: Sequence[EvidenceLink]
    ) -> None:
        """Persist claim-to-passage support, contradiction, or context links."""

    @abstractmethod
    async def append_verification_results(
        self, run_id: UUID, results: Sequence[VerificationResult]
    ) -> None:
        """Persist conservative verification outcomes for claims."""

    @abstractmethod
    async def append_report_statements(
        self, run_id: UUID, statements: Sequence[ReportStatement]
    ) -> None:
        """Persist auditable rendered report statements for a run."""

    @abstractmethod
    async def append_report_bundles(
        self, run_id: UUID, bundles: Sequence[ReportBundle]
    ) -> None:
        """Persist exported Markdown, HTML, and provenance report bundles."""

    @abstractmethod
    async def list_research_plans(self, run_id: UUID) -> list[ResearchPlan]:
        """List planning decisions for a run in persistence order."""

    @abstractmethod
    async def list_query_records(self, run_id: UUID) -> list[QueryRecord]:
        """List discovery-query provenance for a run in persistence order."""

    @abstractmethod
    async def list_provenance_attachments(
        self, run_id: UUID
    ) -> list[ProvenanceAttachment]:
        """List audit-only raw tool output attachments for a run."""

    @abstractmethod
    async def list_sources(self, run_id: UUID) -> list[SourceRecord]:
        """List source records for a run in persistence order."""

    @abstractmethod
    async def list_source_versions(self, run_id: UUID) -> list[SourceVersion]:
        """List immutable fetched-content versions for a run."""

    @abstractmethod
    async def list_passages(self, run_id: UUID) -> list[EvidencePassage]:
        """List exact evidence passages for a run in persistence order."""

    @abstractmethod
    async def list_translations(self, run_id: UUID) -> list[TranslationRecord]:
        """List translations linked to passages in a run."""

    @abstractmethod
    async def list_claims(self, run_id: UUID) -> list[Claim]:
        """List claims for a run in persistence order."""

    @abstractmethod
    async def list_evidence_links(self, run_id: UUID) -> list[EvidenceLink]:
        """List claim-to-passage evidence links for a run."""

    @abstractmethod
    async def list_verification_results(
        self, run_id: UUID
    ) -> list[VerificationResult]:
        """List verification outcomes for a run."""

    @abstractmethod
    async def list_report_statements(self, run_id: UUID) -> list[ReportStatement]:
        """List rendered report statements for a run."""

    @abstractmethod
    async def list_report_bundles(self, run_id: UUID) -> list[ReportBundle]:
        """List exported report bundles for a run."""
