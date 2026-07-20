import unittest
from uuid import uuid4

from polyresearch.evidence.claim_clustering import cluster_claims
from polyresearch.models import Claim


class ClaimClusteringTests(unittest.TestCase):
    def test_groups_highly_similar_propositions_and_retains_distinct_claims(self) -> None:
        first = Claim(
            statement="The policy begins on Monday.",
            atomic_proposition="The policy begins on Monday.",
            evidence_passage_ids=[uuid4()], extraction_confidence=0.9,
        )
        corroborating = Claim(
            statement="The policy begins Monday.",
            atomic_proposition="The policy begins Monday.",
            evidence_passage_ids=[uuid4()], extraction_confidence=0.8,
        )
        distinct = Claim(
            statement="The policy applies nationally.",
            atomic_proposition="The policy applies nationally.",
            evidence_passage_ids=[uuid4()], extraction_confidence=0.8,
        )

        clustered = cluster_claims([first, corroborating, distinct])

        self.assertIsNotNone(clustered[0].claim_cluster_id)
        self.assertEqual(clustered[0].claim_cluster_id, clustered[1].claim_cluster_id)
        self.assertEqual(clustered[0].claim_cluster_method, "lexical_proposition_similarity_v1")
        self.assertIsNone(clustered[2].claim_cluster_id)
