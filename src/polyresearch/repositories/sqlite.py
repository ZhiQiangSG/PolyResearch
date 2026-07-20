"""SQLite implementation of the local PolyResearch evidence ledger."""

import asyncio
import json
import logging
import sqlite3
import threading
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar
from uuid import UUID
from uuid import uuid4

from pydantic import BaseModel

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
    TraceRecord,
    TranslationRecord,
    VerificationResult,
)
from polyresearch.repositories.base import (
    ArtifactConflictError,
    EvidenceRepository,
    ReportProvenanceError,
    RepositoryNotFoundError,
    DiscoveryBudgetReservation,
)
from polyresearch.security import redacted_exception_info

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)


class SqliteEvidenceRepository(EvidenceRepository):
    """Durable local ledger with idempotent, immutable artifact writes.

    The implementation stores a canonical JSON payload alongside relational run IDs.
    This keeps the initial SQLite schema migration-friendly while preserving complete
    Pydantic artifacts and their stable IDs for later graph traversal.
    """

    _MIGRATION_VERSION = 4
    _ARTIFACT_TABLES = (
        "research_plans",
        "query_records",
        "sources",
        "source_versions",
        "passages",
        "translations",
        "claims",
        "evidence_links",
        "verification_results",
        "report_statements",
        "report_bundles",
    )

    def __init__(self, database_path: str | Path = "polyresearch.db") -> None:
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._database_lock = threading.RLock()
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._migrate()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._connection.close()

    def _migrate(self) -> None:
        with self._transaction():
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            initial_applied = self._connection.execute(
                "SELECT 1 FROM schema_migrations WHERE version = 1"
            ).fetchone()
            if not initial_applied:
                self._connection.execute(
                    """
                    CREATE TABLE research_runs (
                        id TEXT PRIMARY KEY,
                        payload TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                for table in self._ARTIFACT_TABLES:
                    self._create_artifact_table(table)
                self._connection.execute("INSERT INTO schema_migrations(version) VALUES (1)")

            attachments_applied = self._connection.execute(
                "SELECT 1 FROM schema_migrations WHERE version = 2"
            ).fetchone()
            if not attachments_applied:
                self._create_artifact_table("provenance_attachments")
                self._connection.execute("INSERT INTO schema_migrations(version) VALUES (2)")

            traces_applied = self._connection.execute(
                "SELECT 1 FROM schema_migrations WHERE version = 3"
            ).fetchone()
            if not traces_applied:
                self._create_artifact_table("trace_records")
                self._connection.execute("INSERT INTO schema_migrations(version) VALUES (3)")

            reservations_applied = self._connection.execute(
                "SELECT 1 FROM schema_migrations WHERE version = 4"
            ).fetchone()
            if not reservations_applied:
                self._connection.execute(
                    """
                    CREATE TABLE discovery_budget_reservations (
                        id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL,
                        query_slots INTEGER NOT NULL,
                        source_slots INTEGER NOT NULL,
                        finalized INTEGER NOT NULL DEFAULT 0,
                        FOREIGN KEY(run_id) REFERENCES research_runs(id)
                    )
                    """
                )
                self._connection.execute(
                    "CREATE INDEX idx_discovery_budget_reservations_run_id "
                    "ON discovery_budget_reservations(run_id)"
                )
                self._connection.execute("INSERT INTO schema_migrations(version) VALUES (4)")

    def _create_artifact_table(self, table: str) -> None:
        self._connection.execute(
            f"""
            CREATE TABLE {table} (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(run_id) REFERENCES research_runs(id)
            )
            """
        )
        self._connection.execute(f"CREATE INDEX idx_{table}_run_id ON {table}(run_id)")

    @contextmanager
    def _transaction(self):
        try:
            yield
        except Exception as error:
            logger.warning(
                "SQLite transaction rolled back",
                extra={"operation": "sqlite_transaction"},
                exc_info=redacted_exception_info(error),
            )
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    @staticmethod
    def _payload(artifact: BaseModel) -> str:
        return json.dumps(
            artifact.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def _ensure_run(self, run_id: UUID) -> None:
        if not self._connection.execute(
            "SELECT 1 FROM research_runs WHERE id = ?", (str(run_id),)
        ).fetchone():
            raise RepositoryNotFoundError(f"Research run {run_id} does not exist")

    def _ensure_artifact(self, table: str, run_id: UUID, artifact_id: UUID) -> None:
        if not self._connection.execute(
            f"SELECT 1 FROM {table} WHERE id = ? AND run_id = ?",
            (str(artifact_id), str(run_id)),
        ).fetchone():
            raise RepositoryNotFoundError(
                f"{table} artifact {artifact_id} does not exist in run {run_id}"
            )

    def _append(
        self, table: str, run_id: UUID, artifacts: Sequence[ModelT]
    ) -> None:
        self._ensure_run(run_id)
        for artifact in artifacts:
            artifact_run_id = getattr(artifact, "run_id", run_id)
            if artifact_run_id != run_id:
                raise ValueError(
                    f"Artifact {artifact.id} belongs to run {artifact_run_id}, not {run_id}"
                )
            payload = self._payload(artifact)
            existing = self._connection.execute(
                f"SELECT payload FROM {table} WHERE id = ?", (str(artifact.id),)
            ).fetchone()
            if existing:
                if existing["payload"] != payload:
                    raise ArtifactConflictError(
                        f"Immutable {table} artifact {artifact.id} has conflicting content"
                    )
                continue
            self._connection.execute(
                f"INSERT INTO {table}(id, run_id, payload) VALUES (?, ?, ?)",
                (str(artifact.id), str(run_id), payload),
            )

    def _list(self, table: str, run_id: UUID, model_type: type[ModelT]) -> list[ModelT]:
        self._ensure_run(run_id)
        rows = self._connection.execute(
            f"SELECT payload FROM {table} WHERE run_id = ? ORDER BY created_at, rowid",
            (str(run_id),),
        ).fetchall()
        return [model_type.model_validate_json(row["payload"]) for row in rows]

    async def _execute(self, operation, *args):
        """Run one serialized SQLite operation outside the event-loop thread."""
        return await asyncio.to_thread(self._execute_locked, operation, *args)

    def _execute_locked(self, operation, *args):
        with self._database_lock:
            return operation(*args)

    async def create_run(self, run: ResearchRun) -> None:
        await self._execute(self._create_run, run)

    def _create_run(self, run: ResearchRun) -> None:
        with self._transaction():
            payload = self._payload(run)
            existing = self._connection.execute(
                "SELECT payload FROM research_runs WHERE id = ?", (str(run.id),)
            ).fetchone()
            if existing:
                if existing["payload"] != payload:
                    raise ArtifactConflictError(
                        f"Immutable research run {run.id} has conflicting content"
                    )
                return
            self._connection.execute(
                "INSERT INTO research_runs(id, payload) VALUES (?, ?)",
                (str(run.id), payload),
            )

    async def get_run(self, run_id: UUID) -> ResearchRun:
        return await self._execute(self._get_run, run_id)

    def _get_run(self, run_id: UUID) -> ResearchRun:
        row = self._connection.execute(
            "SELECT payload FROM research_runs WHERE id = ?", (str(run_id),)
        ).fetchone()
        if not row:
            raise RepositoryNotFoundError(f"Research run {run_id} does not exist")
        return ResearchRun.model_validate_json(row["payload"])

    async def append_research_plans(self, run_id: UUID, plans: Sequence[ResearchPlan]) -> None:
        await self._execute(self._append_research_plans, run_id, plans)

    def _append_research_plans(self, run_id: UUID, plans: Sequence[ResearchPlan]) -> None:
        with self._transaction():
            self._append("research_plans", run_id, plans)

    async def append_query_records(self, run_id: UUID, queries: Sequence[QueryRecord]) -> None:
        await self._execute(self._append_query_records, run_id, queries)

    def _append_query_records(self, run_id: UUID, queries: Sequence[QueryRecord]) -> None:
        with self._transaction():
            self._append("query_records", run_id, queries)

    async def reserve_discovery_budget(
        self, run_id: UUID, *, max_queries: int, max_sources: int, requested_sources: int
    ) -> DiscoveryBudgetReservation:
        return await self._execute(
            self._reserve_discovery_budget, run_id, max_queries, max_sources, requested_sources
        )

    def _reserve_discovery_budget(
        self, run_id: UUID, max_queries: int, max_sources: int, requested_sources: int
    ) -> DiscoveryBudgetReservation:
        if requested_sources < 1:
            raise ValueError("requested_sources must be positive")
        with self._transaction():
            self._ensure_run(run_id)
            reserved_queries = self._connection.execute(
                "SELECT COALESCE(SUM(query_slots), 0) FROM discovery_budget_reservations WHERE run_id = ?",
                (str(run_id),),
            ).fetchone()[0]
            if reserved_queries >= max_queries:
                raise ValueError("Query budget exhausted for this research run.")
            source_count = self._connection.execute(
                "SELECT COUNT(*) FROM sources WHERE run_id = ?", (str(run_id),)
            ).fetchone()[0]
            pending_sources = self._connection.execute(
                "SELECT COALESCE(SUM(source_slots), 0) FROM discovery_budget_reservations "
                "WHERE run_id = ? AND finalized = 0",
                (str(run_id),),
            ).fetchone()[0]
            source_slots = min(requested_sources, max_sources - source_count - pending_sources)
            if source_slots < 1:
                raise ValueError("Source-fetch budget exhausted for this research run.")
            reservation = DiscoveryBudgetReservation(uuid4(), run_id, source_slots)
            self._connection.execute(
                "INSERT INTO discovery_budget_reservations(id, run_id, query_slots, source_slots) "
                "VALUES (?, ?, 1, ?)",
                (str(reservation.id), str(run_id), source_slots),
            )
            return reservation

    async def finalize_discovery_budget(
        self, reservation: DiscoveryBudgetReservation, *, sources_used: int
    ) -> None:
        await self._execute(self._finalize_discovery_budget, reservation, sources_used)

    def _finalize_discovery_budget(
        self, reservation: DiscoveryBudgetReservation, sources_used: int
    ) -> None:
        if not 0 <= sources_used <= reservation.source_slots:
            raise ValueError("sources_used must fit the reserved source capacity")
        with self._transaction():
            cursor = self._connection.execute(
                "UPDATE discovery_budget_reservations SET source_slots = ?, finalized = 1 "
                "WHERE id = ? AND run_id = ? AND finalized = 0",
                (sources_used, str(reservation.id), str(reservation.run_id)),
            )
            if cursor.rowcount != 1:
                raise RepositoryNotFoundError(f"Discovery budget reservation {reservation.id} is unavailable")

    async def append_provenance_attachments(
        self, run_id: UUID, attachments: Sequence[ProvenanceAttachment]
    ) -> None:
        await self._execute(self._append_provenance_attachments, run_id, attachments)

    def _append_provenance_attachments(
        self, run_id: UUID, attachments: Sequence[ProvenanceAttachment]
    ) -> None:
        with self._transaction():
            self._append("provenance_attachments", run_id, attachments)

    async def append_sources(self, run_id: UUID, sources: Sequence[SourceRecord]) -> None:
        await self._execute(self._append_sources, run_id, sources)

    def _append_sources(self, run_id: UUID, sources: Sequence[SourceRecord]) -> None:
        with self._transaction():
            self._append("sources", run_id, sources)

    async def append_source_versions(self, run_id: UUID, versions: Sequence[SourceVersion]) -> None:
        await self._execute(self._append_source_versions, run_id, versions)

    def _append_source_versions(self, run_id: UUID, versions: Sequence[SourceVersion]) -> None:
        with self._transaction():
            for version in versions:
                self._ensure_artifact("sources", run_id, version.source_id)
            self._append("source_versions", run_id, versions)

    async def append_passages(self, run_id: UUID, passages: Sequence[EvidencePassage]) -> None:
        await self._execute(self._append_passages, run_id, passages)

    def _append_passages(self, run_id: UUID, passages: Sequence[EvidencePassage]) -> None:
        with self._transaction():
            for passage in passages:
                self._ensure_artifact("sources", run_id, passage.source_id)
            self._append("passages", run_id, passages)

    async def append_translations(self, run_id: UUID, translations: Sequence[TranslationRecord]) -> None:
        await self._execute(self._append_translations, run_id, translations)

    def _append_translations(self, run_id: UUID, translations: Sequence[TranslationRecord]) -> None:
        with self._transaction():
            for translation in translations:
                self._ensure_artifact("passages", run_id, translation.passage_id)
                if translation.source_original_text_hash is not None:
                    row = self._connection.execute(
                        "SELECT payload FROM passages WHERE id = ? AND run_id = ?",
                        (str(translation.passage_id), str(run_id)),
                    ).fetchone()
                    passage = EvidencePassage.model_validate_json(row["payload"])
                    if translation.source_original_text_hash != passage.original_text_hash:
                        raise ArtifactConflictError(
                            "Translation source hash does not match its immutable original passage"
                        )
            self._append("translations", run_id, translations)

    async def append_claims(self, run_id: UUID, claims: Sequence[Claim]) -> None:
        await self._execute(self._append_claims, run_id, claims)

    def _append_claims(self, run_id: UUID, claims: Sequence[Claim]) -> None:
        with self._transaction():
            for claim in claims:
                for passage_id in claim.evidence_passage_ids:
                    self._ensure_artifact("passages", run_id, passage_id)
            self._append("claims", run_id, claims)

    async def append_evidence_links(self, run_id: UUID, evidence_links: Sequence[EvidenceLink]) -> None:
        await self._execute(self._append_evidence_links, run_id, evidence_links)

    def _append_evidence_links(self, run_id: UUID, evidence_links: Sequence[EvidenceLink]) -> None:
        with self._transaction():
            for link in evidence_links:
                self._ensure_artifact("claims", run_id, link.claim_id)
                self._ensure_artifact("passages", run_id, link.passage_id)
            self._append("evidence_links", run_id, evidence_links)

    async def append_verification_results(self, run_id: UUID, results: Sequence[VerificationResult]) -> None:
        await self._execute(self._append_verification_results, run_id, results)

    def _append_verification_results(self, run_id: UUID, results: Sequence[VerificationResult]) -> None:
        with self._transaction():
            for result in results:
                self._ensure_artifact("claims", run_id, result.claim_id)
                for link_id in result.evidence_link_ids:
                    self._ensure_artifact("evidence_links", run_id, link_id)
            self._append("verification_results", run_id, results)

    async def append_report_statements(self, run_id: UUID, statements: Sequence[ReportStatement]) -> None:
        await self._execute(self._append_report_statements, run_id, statements)

    def _append_report_statements(self, run_id: UUID, statements: Sequence[ReportStatement]) -> None:
        with self._transaction():
            for statement in statements:
                self._ensure_complete_report_trace(run_id, statement)
            self._append("report_statements", run_id, statements)

    def _ensure_complete_report_trace(self, run_id: UUID, statement: ReportStatement) -> None:
        """Reject report prose that cannot reach discovered original evidence.

        Translation records are linked derivatives, so they do not determine
        persistence eligibility. Translation needs remain visible to provenance
        diagnostics when the output language differs from original evidence.
        """
        query_rows = self._connection.execute(
            "SELECT payload FROM query_records WHERE run_id = ?", (str(run_id),)
        ).fetchall()
        query_urls = {
            query.result_url
            for query_row in query_rows
            if (query := QueryRecord.model_validate_json(query_row["payload"])).result_url
        }
        for claim_id in statement.claim_ids:
            claim_row = self._connection.execute(
                "SELECT payload FROM claims WHERE id = ? AND run_id = ?",
                (str(claim_id), str(run_id)),
            ).fetchone()
            if claim_row is None:
                raise RepositoryNotFoundError(
                    f"claims artifact {claim_id} does not exist in run {run_id}"
                )
            claim = Claim.model_validate_json(claim_row["payload"])
            for passage_id in claim.evidence_passage_ids:
                passage_row = self._connection.execute(
                    "SELECT payload FROM passages WHERE id = ? AND run_id = ?",
                    (str(passage_id), str(run_id)),
                ).fetchone()
                if passage_row is None:
                    continue
                passage = EvidencePassage.model_validate_json(passage_row["payload"])
                source_row = self._connection.execute(
                    "SELECT payload FROM sources WHERE id = ? AND run_id = ?",
                    (str(passage.source_id), str(run_id)),
                ).fetchone()
                if source_row is None:
                    continue
                source = SourceRecord.model_validate_json(source_row["payload"])
                if {source.canonical_url, source.discovered_url} & query_urls:
                    return
        raise ReportProvenanceError(
            "Report statement "
            f"{statement.id} lacks a complete statement → claim → original passage "
            "→ source → query trace."
        )

    async def append_report_bundles(self, run_id: UUID, bundles: Sequence[ReportBundle]) -> None:
        await self._execute(self._append_report_bundles, run_id, bundles)

    def _append_report_bundles(self, run_id: UUID, bundles: Sequence[ReportBundle]) -> None:
        with self._transaction():
            self._append("report_bundles", run_id, bundles)

    async def append_trace_records(self, run_id: UUID, records: Sequence[TraceRecord]) -> None:
        await self._execute(self._append, "trace_records", run_id, records)

    async def list_research_plans(self, run_id: UUID) -> list[ResearchPlan]:
        return await self._execute(self._list, "research_plans", run_id, ResearchPlan)

    async def list_query_records(self, run_id: UUID) -> list[QueryRecord]:
        return await self._execute(self._list, "query_records", run_id, QueryRecord)

    async def list_provenance_attachments(
        self, run_id: UUID
    ) -> list[ProvenanceAttachment]:
        return await self._execute(
            self._list, "provenance_attachments", run_id, ProvenanceAttachment
        )

    async def list_sources(self, run_id: UUID) -> list[SourceRecord]:
        return await self._execute(self._list, "sources", run_id, SourceRecord)

    async def list_source_versions(self, run_id: UUID) -> list[SourceVersion]:
        return await self._execute(self._list, "source_versions", run_id, SourceVersion)

    async def list_passages(self, run_id: UUID) -> list[EvidencePassage]:
        return await self._execute(self._list, "passages", run_id, EvidencePassage)

    async def list_translations(self, run_id: UUID) -> list[TranslationRecord]:
        return await self._execute(self._list, "translations", run_id, TranslationRecord)

    async def list_claims(self, run_id: UUID) -> list[Claim]:
        return await self._execute(self._list, "claims", run_id, Claim)

    async def list_evidence_links(self, run_id: UUID) -> list[EvidenceLink]:
        return await self._execute(self._list, "evidence_links", run_id, EvidenceLink)

    async def list_verification_results(self, run_id: UUID) -> list[VerificationResult]:
        return await self._execute(
            self._list, "verification_results", run_id, VerificationResult
        )

    async def list_report_statements(self, run_id: UUID) -> list[ReportStatement]:
        return await self._execute(self._list, "report_statements", run_id, ReportStatement)

    async def list_report_bundles(self, run_id: UUID) -> list[ReportBundle]:
        return await self._execute(self._list, "report_bundles", run_id, ReportBundle)

    async def list_trace_records(self, run_id: UUID) -> list[TraceRecord]:
        return await self._execute(self._list, "trace_records", run_id, TraceRecord)
