"""Persistence boundary for PolyResearch's typed evidence ledger."""

from abc import ABC, abstractmethod
from typing import Sequence
from uuid import UUID

from polyresearch.models import (
    Claim,
    EvidenceLink,
    EvidencePassage,
    ResearchRun,
    SourceRecord,
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
    async def append_sources(
        self, run_id: UUID, sources: Sequence[SourceRecord]
    ) -> None:
        """Persist immutable source records associated with a run."""

    @abstractmethod
    async def append_passages(
        self, run_id: UUID, passages: Sequence[EvidencePassage]
    ) -> None:
        """Persist immutable original-language passages associated with a run."""

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
    async def list_sources(self, run_id: UUID) -> list[SourceRecord]:
        """List source records for a run in persistence order."""

    @abstractmethod
    async def list_passages(self, run_id: UUID) -> list[EvidencePassage]:
        """List exact evidence passages for a run in persistence order."""

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
