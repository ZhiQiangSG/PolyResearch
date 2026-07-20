import importlib
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from polyresearch.models import (
    Claim,
    ClaimExtractionDraft,
    ClaimExtractionResult,
    ClaimClusterVerificationResult,
    ClaimScope,
    EvidenceLink,
    EvidencePassage,
    QueryRecord,
    ReportDraft,
    ReportStatementDraft,
    ResearchRun,
    SourceRecord,
    TranslationDraft,
    VerificationStatus,
)
from polyresearch.repositories import SqliteEvidenceRepository

researcher_module = importlib.import_module("polyresearch.workflows.researcher")
report_module = importlib.import_module("polyresearch.workflows.report_generator")


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
        return ClaimExtractionResult(
            claims=[
                ClaimExtractionDraft(
                    id=self.claim.id,
                    atomic_proposition=self.claim.statement,
                    original_wording=self.claim.original_wording,
                    normalized_statement=self.claim.statement,
                    scope=ClaimScope(description="Limited to the cited passage."),
                    modality="asserted",
                    evidence_passage_ids=self.claim.evidence_passage_ids,
                    extraction_confidence=self.claim.extraction_confidence,
                )
            ]
        )


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


class _ClusterVerificationStub:
    def with_structured_output(self, schema):
        return self

    def with_retry(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        return ClaimClusterVerificationResult.model_validate(
            {
                "clusters": [
                    {
                        "cluster_id": str(self.cluster_id),
                        "cluster_rationale": "The cited passages support the shared proposition.",
                        "claim_assessments": [
                            {
                                "claim_id": str(claim_id),
                                "status": "supported",
                                "confidence": 0.95,
                                "rationale": "The cited passage directly supports this claim.",
                            }
                            for claim_id in self.claim_ids
                        ],
                        "disagreement_assessments": [
                            {
                                "dimension": dimension,
                                "present": False,
                                "explanation": "The cited passages use the same scope.",
                            }
                            for dimension in (
                                "different_time_periods",
                                "different_geographic_scope",
                                "differing_definitions_or_measurement_methods",
                                "different_populations_or_samples",
                                "translation_ambiguity",
                                "genuinely_conflicting_evidence",
                            )
                        ],
                    }
                ]
            }
        )

    def __init__(self, cluster_id, claim_ids):
        self.cluster_id = cluster_id
        self.claim_ids = claim_ids
        self.messages = None


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
            original_factory = researcher_module.create_qwen_chat_model
            researcher_module.create_qwen_chat_model = lambda *args, **kwargs: _TranslationStub()
            try:
                await repository.create_run(run)
                await repository.append_sources(run.id, [source])
                await repository.append_passages(run.id, [passage])
                await repository.append_claims(run.id, [claim])
                await researcher_module.translate_claim_evidence(
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
                researcher_module.create_qwen_chat_model = original_factory
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
            original_factory = researcher_module.create_qwen_chat_model
            researcher_module.create_qwen_chat_model = lambda *args, **kwargs: extractor
            try:
                await repository.create_run(run)
                await repository.append_sources(run.id, [source])
                await repository.append_passages(run.id, [passage])

                result = await researcher_module.extract_claims(
                    {},
                    {
                        "configurable": {
                            "run_id": str(run.id),
                            "evidence_repository": repository,
                        }
                    },
                )

                self.assertEqual([item.id for item in result["claims"]], [claim.id])
                self.assertEqual(
                    [item.id for item in await repository.list_claims(run.id)], [claim.id]
                )
                self.assertEqual(result["claims"][0].scope.description, "Limited to the cited passage.")
                links = await repository.list_evidence_links(run.id)
                self.assertEqual(len(links), 1)
                self.assertEqual(links[0].claim_id, claim.id)
                self.assertEqual(links[0].passage_id, passage.id)
                self.assertEqual(len(extractor.messages), 2)
                self.assertIn("EvidenceLedger", extractor.messages[1].content)
                self.assertNotIn("ToolMessage", extractor.messages[1].content)
            finally:
                researcher_module.create_qwen_chat_model = original_factory
                repository.close()

    async def test_claim_cluster_verification_persists_results_for_every_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteEvidenceRepository(Path(directory) / "research.db")
            run = ResearchRun(id=uuid4(), question="What changed?", output_language="en")
            source = SourceRecord(canonical_url="https://example.test/policy", title="Policy")
            passage = EvidencePassage(
                source_id=source.id,
                text="The policy changed on 1 January.",
                locator="paragraph-1",
            )
            claim = Claim(
                statement="The policy changed on 1 January.",
                evidence_passage_ids=[passage.id],
                extraction_confidence=0.9,
                claim_cluster_id=uuid4(),
            )
            corroborating_claim = Claim(
                statement="The policy was changed on January 1.",
                evidence_passage_ids=[passage.id],
                extraction_confidence=0.85,
                claim_cluster_id=claim.claim_cluster_id,
            )
            original_factory = researcher_module.create_qwen_chat_model
            verifier = _ClusterVerificationStub(
                claim.claim_cluster_id, [claim.id, corroborating_claim.id]
            )
            researcher_module.create_qwen_chat_model = lambda *args, **kwargs: verifier
            try:
                await repository.create_run(run)
                await repository.append_sources(run.id, [source])
                await repository.append_passages(run.id, [passage])
                await repository.append_claims(run.id, [claim, corroborating_claim])
                await repository.append_evidence_links(
                    run.id,
                    [
                        EvidenceLink(
                            claim_id=claim.id,
                            passage_id=passage.id,
                            relationship="supports",
                        ),
                        EvidenceLink(
                            claim_id=corroborating_claim.id,
                            passage_id=passage.id,
                            relationship="supports",
                        ),
                    ],
                )

                result = await researcher_module.verify_claim_clusters(
                    {}, {"configurable": {"run_id": str(run.id), "evidence_repository": repository}}
                )

                persisted = await repository.list_verification_results(run.id)
                self.assertEqual({item.claim_id for item in persisted}, {claim.id, corroborating_claim.id})
                self.assertTrue(all(item.status is VerificationStatus.SUPPORTED for item in persisted))
                self.assertTrue(
                    all(len(item.disagreement_assessments) == 6 for item in persisted)
                )
                self.assertTrue(
                    all(set(item.confidence_factors) == {
                        "directness",
                        "source_quality",
                        "independence",
                        "scope_fit",
                        "recency",
                        "agreement",
                        "translation_certainty",
                    } for item in persisted)
                )
                self.assertTrue(
                    all(
                        item.verifier_model_id == "qwen3.7-plus"
                        and item.verifier_prompt_version == "claim-cluster-verification-v2"
                        and item.verified_at is not None
                        for item in persisted
                    )
                )
                self.assertIn(str(claim.claim_cluster_id), verifier.messages[0].content)
                self.assertEqual(verifier.messages[0].content.count('"cluster_id"'), 1)
                self.assertEqual(result["verification_results"], persisted)
            finally:
                researcher_module.create_qwen_chat_model = original_factory
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
            original_factory = report_module.create_qwen_chat_model
            report_module.create_qwen_chat_model = lambda *args, **kwargs: writer
            try:
                await repository.create_run(run)
                await repository.append_sources(run.id, [source])
                await repository.append_passages(run.id, [passage])
                await repository.append_claims(run.id, [claim])
                await repository.append_evidence_links(run.id, [EvidenceLink(
                    claim_id=claim.id, passage_id=passage.id, relationship="supports",
                )])
                await repository.append_query_records(
                    run.id,
                    [
                        QueryRecord(
                            run_id=run.id,
                            query=run.question,
                            language="en",
                            provider="tavily",
                            result_url=source.canonical_url,
                        )
                    ],
                )

                result = await report_module.final_report_generation(
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
                self.assertIn("## Findings supported by evidence", result["final_report"])
                self.assertIn("## Conflicting or uncertain claims", result["final_report"])
                self.assertIn("## Language coverage and source mix", result["final_report"])
                self.assertIn("## Method, retrieval date, and limitations", result["final_report"])
                self.assertIn("## Complete sources", result["final_report"])
                self.assertIn("## Citation provenance", result["final_report"])
                self.assertEqual(bundles[0].markdown, result["final_report"])
                self.assertEqual(
                    bundles[0].provenance_json["format"],
                    "polyresearch-markdown-provenance-v1",
                )
                self.assertEqual(
                    bundles[0].provenance_json["citations"][f"P:{passage.id}"]["id"],
                    str(passage.id),
                )
                self.assertEqual(
                    bundles[0].provenance_json["statements"][0]["claim_ids"],
                    [str(claim.id)],
                )
                self.assertTrue(bundles[0].qa_passed)
                self.assertEqual(
                    bundles[0].qa_issues[0].code,
                    "wording_exceeds_verification_status",
                )
            finally:
                report_module.create_qwen_chat_model = original_factory
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
            original_factory = researcher_module.create_qwen_chat_model
            researcher_module.create_qwen_chat_model = lambda *args, **kwargs: extractor
            try:
                await repository.create_run(run)
                await repository.append_sources(run.id, [source_a, source_b])
                await repository.append_passages(run.id, [passage_a, passage_b])

                await researcher_module.extract_claims(
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
                self.assertEqual(
                    [claim.id for claim in await repository.list_claims(run.id)], [claim_a.id]
                )
            finally:
                researcher_module.create_qwen_chat_model = original_factory
                repository.close()
