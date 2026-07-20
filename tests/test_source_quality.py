import unittest
from datetime import datetime, timezone

from polyresearch.models import SourceRecord
from polyresearch.source_quality import score_initial_source_quality


class SourceQualityTests(unittest.TestCase):
    def test_scores_primary_transparent_attributed_relevant_source_explainably(self) -> None:
        source = SourceRecord(
            canonical_url="https://agency.gov/policy",
            title="Official policy update",
            publisher="National Policy Agency",
            author="Policy Office",
            published_at=datetime(2026, 7, 20, tzinfo=timezone.utc),
            source_type="official",
        )

        assessment = score_initial_source_quality(
            source, "The official policy update begins Monday.", query="official policy update"
        )

        self.assertGreater(assessment.score, 0.9)
        self.assertEqual(assessment.scoring_version, "initial-source-quality-v1")
        self.assertEqual(assessment.factor_scores["source_type"], 1.0)
        self.assertIn("author_present", assessment.rationale)

    def test_score_is_conservative_when_publisher_attribution_and_relevance_are_missing(self) -> None:
        source = SourceRecord(
            canonical_url="https://unknown.example/post",
            title="Post",
            source_type="commentary",
        )

        assessment = score_initial_source_quality(source, "Unrelated material.", query="policy update")

        self.assertLess(assessment.score, 0.4)
        self.assertIn("author_missing", assessment.rationale)
        self.assertIn("date_missing", assessment.rationale)
