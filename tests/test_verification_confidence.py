import unittest

from polyresearch.evidence.verification_confidence import verification_confidence
from polyresearch.models import (
    Claim,
    DisagreementAssessment,
    DisagreementDimension,
    EvidenceLink,
    EvidencePassage,
    SourceRecord,
    SourceQualityAssessment,
    VerificationStatus,
)


class VerificationConfidenceTests(unittest.TestCase):
    def test_syndicated_pages_count_as_one_independent_source(self) -> None:
        source_a = SourceRecord(
            canonical_url="https://publisher-a.example/policy",
            title="Policy",
            shared_origin_cluster_id="origin:wire-copy",
            initial_quality_assessment=SourceQualityAssessment(
                score=0.8, scoring_version="test"
            ),
        )
        source_b = SourceRecord(
            canonical_url="https://publisher-b.example/policy",
            title="Policy repost",
            shared_origin_cluster_id="origin:wire-copy",
            initial_quality_assessment=SourceQualityAssessment(
                score=0.8, scoring_version="test"
            ),
        )
        passages = [
            EvidencePassage(source_id=source_a.id, text="Policy changed.", locator="p1"),
            EvidencePassage(source_id=source_b.id, text="Policy changed.", locator="p1"),
        ]
        claim = Claim(
            statement="Policy changed.",
            evidence_passage_ids=[passage.id for passage in passages],
            extraction_confidence=0.9,
        )
        links = [
            EvidenceLink(claim_id=claim.id, passage_id=passage.id, relationship="supports")
            for passage in passages
        ]
        assessments = [
            DisagreementAssessment(
                dimension=dimension,
                present=False,
                explanation="The sources use the same scope.",
            )
            for dimension in DisagreementDimension
        ]

        _, factors, independent_source_count = verification_confidence(
            claim=claim,
            status=VerificationStatus.SUPPORTED,
            model_confidence=0.95,
            evidence_links=links,
            passages=passages,
            sources=[source_a, source_b],
            translations=[],
            output_language="en",
            disagreement_assessments=assessments,
        )

        self.assertEqual(independent_source_count, 1)
        self.assertEqual(factors["independence"], 0.65)
