"""Tests for durable, report-visible unresolved evidence conflicts."""

from uuid import uuid4

from polyresearch.models import (
    Claim,
    DisagreementAssessment,
    DisagreementDimension,
    VerificationResult,
    VerificationStatus,
)
from polyresearch.workflows.report_generator import (
    _build_unresolved_disagreements,
    _render_statement_markdown,
)


def test_unresolved_conflict_is_durable_and_rendered_with_resolution_needs() -> None:
    cluster_id = uuid4()
    passage_id = uuid4()
    first_claim = Claim(
        statement="The programme reached 80% of its target.",
        evidence_passage_ids=[passage_id],
        extraction_confidence=0.8,
        claim_cluster_id=cluster_id,
    )
    second_claim = Claim(
        statement="The programme reached 55% of its target.",
        evidence_passage_ids=[passage_id],
        extraction_confidence=0.8,
        claim_cluster_id=cluster_id,
    )
    assessment = DisagreementAssessment(
        dimension=DisagreementDimension.GENUINE_CONFLICT,
        present=True,
        explanation="Independent records report incompatible values for the same target.",
    )
    disagreements = _build_unresolved_disagreements(
        claims=[first_claim, second_claim],
        verification_results=[
            VerificationResult(
                claim_id=first_claim.id,
                status=VerificationStatus.SUPPORTED,
                confidence=0.7,
                rationale="Supported by the first record.",
                disagreement_assessments=[assessment],
            ),
            VerificationResult(
                claim_id=second_claim.id,
                status=VerificationStatus.CONTRADICTED,
                confidence=0.7,
                rationale="Contradicted by the first record.",
                disagreement_assessments=[assessment],
            ),
        ],
    )

    assert len(disagreements) == 1
    disagreement = disagreements[0]
    assert disagreement.cluster_id == cluster_id
    assert disagreement.conflicting_claims == [first_claim.statement, second_claim.statement]
    assert disagreement.why_it_may_conflict == [assessment.explanation]
    assert disagreement.evidence_needed == [
        "Additional independent primary or official records that directly address the conflicting proposition."
    ]

    markdown = _render_statement_markdown(
        title="Test report",
        statements=[],
        passages=[],
        sources=[],
        qa_issues=[],
        unresolved_disagreements=disagreements,
    )
    assert "## Unresolved disagreements" in markdown
    assert "**What conflicts:**" in markdown
    assert "**Why it may conflict:**" in markdown
    assert "**Evidence needed to resolve it:**" in markdown

