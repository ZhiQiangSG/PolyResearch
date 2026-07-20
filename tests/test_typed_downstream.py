import importlib
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from polyresearch.models import (
    Claim,
    ClaimExtractionResult,
    EvidencePassage,
    ReportDraft,
    ReportStatementDraft,
    ResearchRun,
    SourceRecord,
    TranslationDraft,
)
from polyresearch.repositories import SqliteEvidenceRepository

graph_module = importlib.import_module("polyresearch.graph")


class _ClaimExtractorStub:
    def __init__(self, claim: Claim) -> None:
        self.claim = claim
        self.messages = None

    def with_structured_output(self, schema):
        return self

    def with_retry(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        return ClaimExtractionResult(claims=[self.claim])


class _ReportWriterStub:
    def __init__(self, draft: ReportDraft) -> None:
        self.draft = draft

    def with_structured_output(self, schema):
        return self

    def with_retry(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        return self.draft


class _TranslationStub:
    def with_structured_output(self, schema):
        return self

    def with_retry(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        return TranslationDraft(translated_text="The policy changed.", confidence=0.9)


class TypedDownstreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_translates_only_claim_evidence_needed_for_output_language(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteEvidenceRepository(Path(directory) / "research.db")
            run = ResearchRun(id=uuid4(), question="What changed?", output_language="en")
            source = SourceRecord(canonical_url="https://example.test/policy", title="Policy")
            passage = EvidencePassage(
                source_id=source.id,
                text="政策已变更。",
                locator="paragraph-1",
                original_language="zh",
            )
            claim = Claim(
                statement="The policy changed.",
                evidence_passage_ids=[passage.id],
                extraction_confidence=0.9,
            )
            original_factory = graph_module.create_qwen_chat_model
            graph_module.create_qwen_chat_model = lambda *args, **kwargs: _TranslationStub()
            try:
                await repository.create_run(run)
                await repository.append_sources(run.id, [source])
                await repository.append_passages(run.id, [passage])
                await repository.append_claims(run.id, [claim])
                await graph_module.translate_claim_evidence(
                    {},
                    {"configurable": {
                        "run_id": str(run.id),
                        "evidence_repository": repository,
                        "output_language": "en",
                    }},
                )
                translations = await repository.list_translations(run.id)
                self.assertEqual(len(translations), 1)
                self.assertEqual(translations[0].passage_id, passage.id)
                self.assertEqual(translations[0].source_original_text_hash, passage.original_text_hash)
                self.assertEqual(translations[0].target_language, "en")
            finally:
                graph_module.create_qwen_chat_model = original_factory
                repository.close()
    async def test_claim_extraction_reads_and_writes_the_durable_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteEvidenceRepository(Path(directory) / "research.db")
            run = ResearchRun(
                id=uuid4(), question="What changed?", output_language="en"
            )
            source = SourceRecord(
                canonical_url="https://example.test/policy", title="Policy update"
            )
            passage = EvidencePassage(
                source_id=source.id,
                text="The policy changed on 1 January.",
                locator="paragraph-1",
                original_language="en",
            )
            claim = Claim(
                statement="The policy changed on 1 January.",
                evidence_passage_ids=[passage.id],
                extraction_confidence=0.9,
            )
            extractor = _ClaimExtractorStub(claim)
            original_factory = graph_module.create_qwen_chat_model
            graph_module.create_qwen_chat_model = lambda *args, **kwargs: extractor
            try:
                await repository.create_run(run)
                await repository.append_sources(run.id, [source])
                await repository.append_passages(run.id, [passage])

                result = await graph_module.extract_claims(
                    {},
                    {
                        "configurable": {
                            "run_id": str(run.id),
                            "evidence_repository": repository,
                        }
                    },
                )

                self.assertEqual(result["claims"], [claim])
                self.assertEqual(await repository.list_claims(run.id), [claim])
                links = await repository.list_evidence_links(run.id)
                self.assertEqual(len(links), 1)
                self.assertEqual(links[0].claim_id, claim.id)
                self.assertEqual(links[0].passage_id, passage.id)
                self.assertEqual(len(extractor.messages), 2)
                self.assertIn("EvidenceLedger", extractor.messages[1].content)
                self.assertNotIn("ToolMessage", extractor.messages[1].content)
            finally:
                graph_module.create_qwen_chat_model = original_factory
                repository.close()

    async def test_report_generation_persists_statement_and_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteEvidenceRepository(Path(directory) / "research.db")
            run = ResearchRun(
                id=uuid4(), question="What changed?", output_language="en"
            )
            source = SourceRecord(
                canonical_url="https://example.test/policy", title="Policy update"
            )
            passage = EvidencePassage(
                source_id=source.id,
                text="The policy changed on 1 January.",
                locator="paragraph-1",
            )
            claim = Claim(
                statement="The policy changed on 1 January.",
                evidence_passage_ids=[passage.id],
                extraction_confidence=0.9,
            )
            writer = _ReportWriterStub(
                ReportDraft(
                    title="Policy update",
                    statements=[
                        ReportStatementDraft(
                            rendered_text="The policy changed on 1 January.",
                            claim_ids=[claim.id],
                        )
                    ],
                )
            )
            original_factory = graph_module.create_qwen_chat_model
            graph_module.create_qwen_chat_model = lambda *args, **kwargs: writer
            try:
                await repository.create_run(run)
                await repository.append_sources(run.id, [source])
                await repository.append_passages(run.id, [passage])
                await repository.append_claims(run.id, [claim])

                result = await graph_module.final_report_generation(
                    {"messages": [], "research_brief": run.question},
                    {
                        "configurable": {
                            "run_id": str(run.id),
                            "evidence_repository": repository,
                        }
                    },
                )

                statements = await repository.list_report_statements(run.id)
                bundles = await repository.list_report_bundles(run.id)
                self.assertEqual(len(statements), 1)
                self.assertEqual(statements[0].claim_ids, [claim.id])
                self.assertEqual(statements[0].citation_ids, [passage.id])
                self.assertIn(f"[P:{passage.id}]", result["final_report"])
                self.assertEqual(bundles[0].markdown, result["final_report"])
                self.assertTrue(bundles[0].qa_passed)
                self.assertEqual(
                    bundles[0].qa_issues[0].code,
                    "wording_exceeds_verification_status",
                )
            finally:
                graph_module.create_qwen_chat_model = original_factory
                repository.close()

    async def test_claim_extraction_isolated_to_its_research_unit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteEvidenceRepository(Path(directory) / "research.db")
            run = ResearchRun(
                id=uuid4(), question="What changed?", output_language="en"
            )
            unit_a, unit_b = uuid4(), uuid4()
            source_a = SourceRecord(
                canonical_url="https://example.test/a",
                title="Unit A",
                research_unit_id=unit_a,
            )
            source_b = SourceRecord(
                canonical_url="https://example.test/b",
                title="Unit B",
                research_unit_id=unit_b,
            )
            passage_a = EvidencePassage(
                source_id=source_a.id, text="Unit A evidence.", locator="paragraph-1"
            )
            passage_b = EvidencePassage(
                source_id=source_b.id, text="Unit B evidence.", locator="paragraph-1"
            )
            claim_a = Claim(
                statement="Unit A claim.",
                evidence_passage_ids=[passage_a.id],
                extraction_confidence=0.9,
            )
            extractor = _ClaimExtractorStub(claim_a)
            original_factory = graph_module.create_qwen_chat_model
            graph_module.create_qwen_chat_model = lambda *args, **kwargs: extractor
            try:
                await repository.create_run(run)
                await repository.append_sources(run.id, [source_a, source_b])
                await repository.append_passages(run.id, [passage_a, passage_b])

                await graph_module.extract_claims(
                    {},
                    {
                        "configurable": {
                            "run_id": str(run.id),
                            "research_unit_id": str(unit_a),
                            "evidence_repository": repository,
                        }
                    },
                )

                ledger_content = extractor.messages[1].content
                self.assertIn(str(source_a.id), ledger_content)
                self.assertNotIn(str(source_b.id), ledger_content)
                self.assertEqual(await repository.list_claims(run.id), [claim_a])
            finally:
                graph_module.create_qwen_chat_model = original_factory
                repository.close()
