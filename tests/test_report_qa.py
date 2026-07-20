import unittest
from uuid import uuid4

from polyresearch.models import (
    Claim,
    EvidencePassage,
    ReportDraft,
    ReportStatementDraft,
    ReportStatement,
    SourceRecord,
    VerificationStatus,
)
from polyresearch import graph as graph_module
from polyresearch.report_qa import validate_report_statements


class ReportQaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.run_id = uuid4()
        self.source = SourceRecord(
            canonical_url="https://example.test/policy", title="Policy update"
        )
        self.passage = EvidencePassage(
            source_id=self.source.id,
            text="The policy changed on 1 January.",
            locator="paragraph-1",
        )
        self.claim = Claim(
            statement="The policy changed on 1 January.",
            evidence_passage_ids=[self.passage.id],
            extraction_confidence=0.9,
        )

    def test_blocks_unknown_claims_and_unresolvable_citations(self) -> None:
        statement = ReportStatement(
            run_id=self.run_id,
            rendered_text="The policy changed on 1 January.",
            claim_ids=[uuid4()],
            citation_ids=[uuid4()],
            verification_status=VerificationStatus.SUPPORTED,
        )

        issues = validate_report_statements(
            statements=[statement],
            claims=[self.claim],
            passages=[self.passage],
            sources=[self.source],
        )

        self.assertEqual(
            {issue.code for issue in issues},
            {"unknown_claim_id", "unresolvable_citation"},
        )
        self.assertTrue(all(issue.severity == "error" for issue in issues))

    def test_flags_assertive_wording_when_evidence_is_insufficient(self) -> None:
        statement = ReportStatement(
            run_id=self.run_id,
            rendered_text="The policy changed on 1 January.",
            claim_ids=[self.claim.id],
            citation_ids=[self.passage.id],
            verification_status=VerificationStatus.INSUFFICIENT_EVIDENCE,
        )

        issues = validate_report_statements(
            statements=[statement],
            claims=[self.claim],
            passages=[self.passage],
            sources=[self.source],
        )

        self.assertEqual([issue.code for issue in issues], ["wording_exceeds_verification_status"])
        self.assertEqual(issues[0].severity, "warning")

    def test_unknown_claim_draft_remains_visible_to_qa(self) -> None:
        statement = graph_module._build_report_statements(
            run_id=self.run_id,
            report_draft=ReportDraft(
                statements=[
                    ReportStatementDraft(
                        rendered_text="The policy changed on 1 January.",
                        claim_ids=[uuid4()],
                    )
                ]
            ),
            claims=[self.claim],
            verification_results=[],
        )[0]

        issues = validate_report_statements(
            statements=[statement],
            claims=[self.claim],
            passages=[self.passage],
            sources=[self.source],
        )

        self.assertEqual(
            {issue.code for issue in issues},
            {"unknown_claim_id", "missing_citation", "wording_exceeds_verification_status"},
        )
