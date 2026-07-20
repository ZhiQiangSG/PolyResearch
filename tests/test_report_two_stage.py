import importlib
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from polyresearch.models import (
    Claim,
    EvidencePassage,
    QueryRecord,
    ReportDraft,
    EvidenceLink,
    ReportOutline,
    ReportOutlineSection,
    ReportStatementDraft,
    ResearchRun,
    SourceRecord,
    SourceQualityAssessment,
    TranslationRecord,
    VerificationResult,
    VerificationStatus,
)
from polyresearch.repositories import SqliteEvidenceRepository

report_module = importlib.import_module("polyresearch.workflows.report_generator")


class _SchemaAwareReportWriter:
    """Records the two structured Qwen stages without making a network call."""

    def __init__(self, claim_id):
        self.claim_id = claim_id
        self.schemas = []
        self.prompts = []
        self.schema = None

    def with_structured_output(self, schema):
        clone = _SchemaAwareReportWriter(self.claim_id)
        clone.schemas = self.schemas
        clone.prompts = self.prompts
        clone.schema = schema
        return clone

    def with_retry(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        self.schemas.append(self.schema)
        self.prompts.append(messages[0].content)
        if self.schema is ReportOutline:
            return ReportOutline(
                title="Policy update",
                sections=[ReportOutlineSection(heading="Finding", claim_ids=[self.claim_id])],
            )
        return ReportDraft(
            title="Policy update",
            statements=[
                ReportStatementDraft(
                    rendered_text="Available evidence indicates that the policy changed.",
                    claim_ids=[self.claim_id],
                )
            ],
        )


class ReportTwoStageTests(unittest.IsolatedAsyncioTestCase):
    async def test_writing_is_constrained_by_claim_bound_outline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteEvidenceRepository(Path(directory) / "research.db")
            run = ResearchRun(id=uuid4(), question="What changed?", output_language="en")
            source = SourceRecord(canonical_url="https://example.test/policy", title="Policy")
            passage = EvidencePassage(source_id=source.id, text="The policy changed.", locator="p1")
            claim = Claim(
                statement="The policy changed.",
                evidence_passage_ids=[passage.id],
                extraction_confidence=0.9,
            )
            writer = _SchemaAwareReportWriter(claim.id)
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
                await repository.append_query_records(run.id, [QueryRecord(
                    run_id=run.id, query=run.question, language="en", provider="tavily",
                    result_url=source.canonical_url,
                )])

                result = await report_module.final_report_generation(
                    {"messages": [], "research_brief": run.question},
                    {"configurable": {"run_id": str(run.id), "evidence_repository": repository}},
                )

                self.assertNotIn("Report QA failed", result["final_report"])
                self.assertEqual(writer.schemas, [ReportOutline, ReportDraft])
                self.assertIn("<ApprovedArtifacts>", writer.prompts[0])
                self.assertIn("<ReportOutline>", writer.prompts[1])
                self.assertNotIn("The policy changed.", writer.prompts[1].split("<ApprovedArtifacts>")[0])
                await report_module.final_report_generation(
                    {"messages": [], "research_brief": run.question},
                    {"configurable": {"run_id": str(run.id), "evidence_repository": repository}},
                )
                self.assertEqual(writer.schemas, [ReportOutline, ReportDraft])
                self.assertEqual(len(await repository.list_report_bundles(run.id)), 1)
                # A later verification/conflict-resolution pass changes the
                # evidence snapshot, so QA-passed prose must be regenerated.
                await repository.append_verification_results(run.id, [VerificationResult(
                    claim_id=claim.id,
                    status=VerificationStatus.SUPPORTED,
                    confidence=0.95,
                    rationale="A later verification pass confirmed the policy update.",
                    evidence_link_ids=[(await repository.list_evidence_links(run.id))[0].id],
                    verifier_model_id="qwen-test",
                    verifier_prompt_version="verification-v1",
                    attempt_number=2,
                    trigger="conflict_resolution",
                )])
                await report_module.final_report_generation(
                    {"messages": [], "research_brief": run.question},
                    {"configurable": {"run_id": str(run.id), "evidence_repository": repository}},
                )
                self.assertEqual(writer.schemas, [ReportOutline, ReportDraft, ReportOutline, ReportDraft])
                self.assertEqual(len(await repository.list_report_bundles(run.id)), 2)
            finally:
                report_module.create_qwen_chat_model = original_factory
                repository.close()

    def test_html_evidence_anchor_exposes_complete_provenance_panel_data(self) -> None:
        source = SourceRecord(
            canonical_url="https://example.test/policy", title="Official policy", publisher="Agency",
            language="zh", initial_quality_assessment=SourceQualityAssessment(
                score=0.95, scoring_version="v1", rationale=["Primary official publication."],
            ),
        )
        passage = EvidencePassage(
            source_id=source.id, text="政策已经变化。", locator="paragraph-3", original_language="zh",
        )
        claim = Claim(statement="The policy changed.", evidence_passage_ids=[passage.id], extraction_confidence=0.9)
        statement = report_module.ReportStatement(
            run_id=uuid4(), rendered_text="The policy changed.", claim_ids=[claim.id],
            citation_ids=[passage.id], verification_status=VerificationStatus.SUPPORTED,
        )
        translation = TranslationRecord(passage_id=passage.id, translated_text="The policy changed.", target_language="en")
        link = EvidenceLink(claim_id=claim.id, passage_id=passage.id, relationship="supports", rationale="Direct statement.")
        result = VerificationResult(
            claim_id=claim.id, status=VerificationStatus.SUPPORTED, confidence=0.92,
            rationale="Direct official support.", evidence_link_ids=[link.id],
            verifier_model_id="qwen-test", verifier_prompt_version="verification-v1",
        )

        html = report_module._render_statement_html(
            title="Policy update", statements=[statement], claims=[claim], passages=[passage],
            sources=[source], translations=[translation], evidence_links=[link], verification_results=[result],
        )

        self.assertIn('class="report-statement"', html)
        self.assertIn('id="evidence-panel"', html)
        self.assertIn("Findings supported by evidence", html)
        self.assertIn("Conflicting or uncertain claims", html)
        self.assertIn("Language coverage and source mix", html)
        self.assertIn("Method, retrieval date, and limitations", html)
        self.assertIn("Complete sources", html)
        self.assertIn("Official policy", html)
        self.assertIn("政策已经变化。", html)
        self.assertIn("The policy changed.", html)
        self.assertIn("qwen-test", html)
        self.assertIn("verification-v1", html)
        self.assertIn("Primary official publication.", html)
