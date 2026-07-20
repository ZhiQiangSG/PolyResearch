import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from polyresearch.cli import build_ledger_inspection, build_report_trace_inspection
from polyresearch.models import (
    Claim,
    EvidencePassage,
    QueryRecord,
    ReportStatement,
    ResearchRun,
    SourceRecord,
    TranslationRecord,
    VerificationStatus,
)
from polyresearch.repositories import SqliteEvidenceRepository


class LedgerInspectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_inspection_resolves_source_to_passages_translations_queries_and_claims(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteEvidenceRepository(Path(directory) / "research.db")
            run = ResearchRun(id=uuid4(), question="What changed?", output_language="en")
            source = SourceRecord(
                canonical_url="https://example.test/policy", title="Policy update"
            )
            passage = EvidencePassage(
                source_id=source.id, text="政策已变更。", locator="paragraph-1", original_language="zh"
            )
            translation = TranslationRecord(
                passage_id=passage.id,
                translated_text="The policy changed.",
                target_language="en",
                source_original_text_hash=passage.original_text_hash,
            )
            claim = Claim(
                statement="The policy changed.",
                evidence_passage_ids=[passage.id],
                extraction_confidence=0.9,
            )
            query = QueryRecord(
                run_id=run.id,
                query="政策",
                language="zh",
                provider="bailian_web_search",
                result_url=source.canonical_url,
            )
            try:
                await repository.create_run(run)
                await repository.append_sources(run.id, [source])
                await repository.append_passages(run.id, [passage])
                await repository.append_translations(run.id, [translation])
                await repository.append_claims(run.id, [claim])
                await repository.append_query_records(run.id, [query])

                inspection = await build_ledger_inspection(repository, run.id, source.id)

                self.assertEqual(inspection["run_id"], str(run.id))
                self.assertEqual(len(inspection["sources"]), 1)
                source_view = inspection["sources"][0]
                self.assertEqual(source_view["source"]["id"], str(source.id))
                self.assertEqual(source_view["passages"][0]["id"], str(passage.id))
                self.assertEqual(source_view["translations"][0]["passage_id"], str(passage.id))
                self.assertEqual(source_view["queries"][0]["result_url"], source.canonical_url)
                self.assertEqual(source_view["downstream_claims"][0]["id"], str(claim.id))
            finally:
                repository.close()

    async def test_trace_inspection_prints_complete_path_for_report_statement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteEvidenceRepository(Path(directory) / "research.db")
            run = ResearchRun(id=uuid4(), question="What changed?", output_language="en")
            source = SourceRecord(canonical_url="https://example.test/policy", title="Policy")
            passage = EvidencePassage(
                source_id=source.id, text="政策已变更。", locator="paragraph-1", original_language="zh"
            )
            translation = TranslationRecord(
                passage_id=passage.id, translated_text="The policy changed.", target_language="en",
                source_original_text_hash=passage.original_text_hash,
            )
            claim = Claim(statement="The policy changed.", evidence_passage_ids=[passage.id], extraction_confidence=0.9)
            statement = ReportStatement(
                run_id=run.id, rendered_text="The policy changed.", claim_ids=[claim.id],
                citation_ids=[passage.id], verification_status=VerificationStatus.SUPPORTED,
            )
            query = QueryRecord(
                run_id=run.id, query="政策", language="zh", provider="bailian_web_search",
                result_url=source.canonical_url,
            )
            try:
                await repository.create_run(run)
                await repository.append_query_records(run.id, [query])
                await repository.append_sources(run.id, [source])
                await repository.append_passages(run.id, [passage])
                await repository.append_translations(run.id, [translation])
                await repository.append_claims(run.id, [claim])
                await repository.append_report_statements(run.id, [statement])

                inspection = await build_report_trace_inspection(repository, run.id, statement.id)

                self.assertTrue(inspection["complete"])
                self.assertEqual(inspection["traces"][0]["query_id"], str(query.id))
                self.assertEqual(inspection["traces"][0]["source_id"], str(source.id))
                self.assertEqual(inspection["traces"][0]["translation_id"], str(translation.id))
                self.assertEqual(inspection["diagnostics"], [])
            finally:
                repository.close()
