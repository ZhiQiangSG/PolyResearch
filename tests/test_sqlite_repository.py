import tempfile
import threading
import unittest
from pathlib import Path

from polyresearch.models import (
    AtomicSubquestion,
    Claim,
    EvidenceLink,
    EvidencePassage,
    ProvenanceAttachment,
    QueryRecord,
    ReportBundle,
    ReportStatement,
    ResearchPlan,
    ResearchLanguage,
    ResearchRun,
    SourceRecord,
    SourceVersion,
    TranslationRecord,
    VerificationResult,
    VerificationStatus,
)
from polyresearch.repositories import SqliteEvidenceRepository


class SqliteEvidenceRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_repository_io_runs_off_the_event_loop_thread(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteEvidenceRepository(Path(directory) / "research.db")
            run = ResearchRun(question="What changed?", output_language="en")
            observed_thread_ids = []
            original_get_run = repository._get_run

            def observe_get_run(run_id):
                observed_thread_ids.append(threading.get_ident())
                return original_get_run(run_id)

            repository._get_run = observe_get_run
            try:
                await repository.create_run(run)
                await repository.get_run(run.id)

                self.assertEqual(len(observed_thread_ids), 1)
                self.assertNotEqual(observed_thread_ids[0], threading.get_ident())
            finally:
                repository.close()

    async def test_round_trips_the_full_evidence_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteEvidenceRepository(Path(directory) / "research.db")
            try:
                run = ResearchRun(question="What changed?", output_language="en")
                plan = ResearchPlan(
                    run_id=run.id,
                    subquestions=[
                        AtomicSubquestion(
                            question="What changed?",
                            answer_scope="Identify the policy change.",
                        )
                    ],
                    ranked_languages=[
                        ResearchLanguage(
                            language="en",
                            priority=1,
                            query_budget=2,
                            expected_unique_value="Primary source coverage.",
                            selection_rationale="The requested output is English.",
                            expected_source_types=["official"],
                        )
                    ],
                    language_rationale={"en": "Primary source language"},
                    query_variants={"en": ["policy update"]},
                )
                query = QueryRecord(
                    run_id=run.id,
                    query="policy update",
                    language="en",
                    provider="tavily",
                )
                attachment = ProvenanceAttachment(
                    run_id=run.id,
                    provider="tavily",
                    tool_name="tavily_search",
                    raw_output='{"results": ["untrusted tool payload"]}',
                )
                source = SourceRecord(
                    canonical_url="https://example.test/policy",
                    title="Policy update",
                    content_hash="source-hash",
                )
                version = SourceVersion(
                    source_id=source.id,
                    version_number=1,
                    content_hash="content-hash",
                    raw_content="The policy changed on 1 January.",
                )
                passage = EvidencePassage(
                    source_id=source.id,
                    text="The policy changed on 1 January.",
                    locator="paragraph 1",
                    original_language="en",
                )
                translation = TranslationRecord(
                    passage_id=passage.id,
                    translated_text="La política cambió el 1 de enero.",
                    target_language="es",
                    confidence=0.95,
                )
                claim = Claim(
                    statement="The policy changed on 1 January.",
                    evidence_passage_ids=[passage.id],
                    extraction_confidence=0.9,
                )
                evidence_link = EvidenceLink(
                    claim_id=claim.id,
                    passage_id=passage.id,
                    relationship="supports",
                )
                verification = VerificationResult(
                    claim_id=claim.id,
                    status=VerificationStatus.SUPPORTED,
                    confidence=0.9,
                    rationale="Direct official statement.",
                    evidence_link_ids=[evidence_link.id],
                )
                statement = ReportStatement(
                    run_id=run.id,
                    rendered_text="The policy changed on 1 January.",
                    claim_ids=[claim.id],
                    citation_ids=[passage.id],
                    verification_status=VerificationStatus.SUPPORTED,
                )
                bundle = ReportBundle(
                    run_id=run.id,
                    markdown="# Finding\n\nThe policy changed.",
                    provenance_json={"statement_ids": [str(statement.id)]},
                )

                await repository.create_run(run)
                await repository.append_research_plans(run.id, [plan])
                await repository.append_query_records(run.id, [query])
                await repository.append_provenance_attachments(run.id, [attachment])
                await repository.append_sources(run.id, [source])
                await repository.append_source_versions(run.id, [version])
                await repository.append_passages(run.id, [passage])
                await repository.append_translations(run.id, [translation])
                await repository.append_claims(run.id, [claim])
                await repository.append_evidence_links(run.id, [evidence_link])
                await repository.append_verification_results(run.id, [verification])
                await repository.append_report_statements(run.id, [statement])
                await repository.append_report_bundles(run.id, [bundle])

                # Reopening the database proves the ledger is durable rather than
                # merely retaining artifacts in the repository instance.
                repository.close()
                repository = SqliteEvidenceRepository(Path(directory) / "research.db")

                self.assertEqual(await repository.get_run(run.id), run)
                self.assertEqual(await repository.list_research_plans(run.id), [plan])
                self.assertEqual(await repository.list_query_records(run.id), [query])
                self.assertEqual(
                    await repository.list_provenance_attachments(run.id), [attachment]
                )
                self.assertEqual(await repository.list_sources(run.id), [source])
                self.assertEqual(await repository.list_source_versions(run.id), [version])
                self.assertEqual(await repository.list_passages(run.id), [passage])
                self.assertEqual(await repository.list_translations(run.id), [translation])
                self.assertEqual(await repository.list_claims(run.id), [claim])
                self.assertEqual(await repository.list_evidence_links(run.id), [evidence_link])
                self.assertEqual(
                    await repository.list_verification_results(run.id), [verification]
                )
                self.assertEqual(await repository.list_report_statements(run.id), [statement])
                self.assertEqual(await repository.list_report_bundles(run.id), [bundle])

                # An identical immutable write is idempotent.
                await repository.append_sources(run.id, [source])
                self.assertEqual(len(await repository.list_sources(run.id)), 1)

                # A resumed process can append new artifacts to the same durable run.
                resumed_query = QueryRecord(
                    run_id=run.id,
                    query="policy update implementation date",
                    language="en",
                    provider="tavily",
                )
                await repository.append_query_records(run.id, [resumed_query])
                self.assertEqual(
                    await repository.list_query_records(run.id), [query, resumed_query]
                )
            finally:
                repository.close()


if __name__ == "__main__":
    unittest.main()
