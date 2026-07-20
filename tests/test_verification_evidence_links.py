"""Tests for immutable verifier-derived support and contradiction paths."""

import asyncio
import importlib
import tempfile
from pathlib import Path
from uuid import UUID, uuid4

from polyresearch.evidence.provenance_graph import build_provenance_graph
from polyresearch.models import (
    Claim,
    ClaimClusterVerificationResult,
    EvidenceLink,
    EvidencePassage,
    ResearchRun,
    SourceRecord,
    VerificationStatus,
)
from polyresearch.repositories import SqliteEvidenceRepository


researcher_module = importlib.import_module("polyresearch.workflows.researcher")


class _ContradictionVerifier:
    def __init__(self, cluster_id: UUID, claim_id: UUID, evidence_link_id: UUID) -> None:
        self.cluster_id = cluster_id
        self.claim_id = claim_id
        self.evidence_link_id = evidence_link_id

    def with_structured_output(self, schema):
        return self

    def with_retry(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        return ClaimClusterVerificationResult.model_validate(
            {
                "clusters": [
                    {
                        "cluster_id": str(self.cluster_id),
                        "cluster_rationale": "The passage directly negates the claim.",
                        "claim_assessments": [
                            {
                                "claim_id": str(self.claim_id),
                                "status": "contradicted",
                                "confidence": 0.9,
                                "rationale": "The cited passage says the policy did not begin.",
                                "evidence_assessments": [
                                    {
                                        "evidence_link_id": str(self.evidence_link_id),
                                        "relationship": "contradicts",
                                        "rationale": "The passage explicitly negates the claim.",
                                    }
                                ],
                            }
                        ],
                        "disagreement_assessments": [
                            {
                                "dimension": dimension,
                                "present": dimension == "genuinely_conflicting_evidence",
                                "explanation": "The records conflict directly."
                                if dimension == "genuinely_conflicting_evidence"
                                else "This dimension does not explain the disagreement.",
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


def test_verifier_preserves_extraction_support_and_appends_contradiction_path() -> None:
    asyncio.run(_verify_evidence_relationships())


async def _verify_evidence_relationships() -> None:
    with tempfile.TemporaryDirectory() as directory:
        repository = SqliteEvidenceRepository(Path(directory) / "research.db")
        run = ResearchRun(id=uuid4(), question="Did the policy begin?", output_language="en")
        cluster_id = uuid4()
        source = SourceRecord(canonical_url="https://official.example/notice", title="Notice")
        passage = EvidencePassage(
            source_id=source.id,
            text="The policy did not begin in January.",
            locator="paragraph 1",
        )
        claim = Claim(
            statement="The policy began in January.",
            evidence_passage_ids=[passage.id],
            extraction_confidence=0.8,
            claim_cluster_id=cluster_id,
        )
        extraction_link = EvidenceLink(
            claim_id=claim.id,
            passage_id=passage.id,
            relationship="supports",
            rationale="Extraction linked the passage before verification.",
        )
        original_factory = researcher_module.create_qwen_chat_model
        researcher_module.create_qwen_chat_model = lambda *args, **kwargs: _ContradictionVerifier(
            cluster_id, claim.id, extraction_link.id
        )
        try:
            await repository.create_run(run)
            await repository.append_sources(run.id, [source])
            await repository.append_passages(run.id, [passage])
            await repository.append_claims(run.id, [claim])
            await repository.append_evidence_links(run.id, [extraction_link])

            await researcher_module.verify_claim_clusters(
                {}, {"configurable": {"run_id": str(run.id), "evidence_repository": repository}}
            )

            results = await repository.list_verification_results(run.id)
            links = await repository.list_evidence_links(run.id)
            derived_link = next(link for link in links if link.origin == "verification")
            assert len(links) == 2
            assert extraction_link.relationship == "supports"
            assert derived_link.relationship == "contradicts"
            assert derived_link.verification_result_id == results[0].id
            assert results[0].status is VerificationStatus.CONTRADICTED
            assert results[0].evidence_link_ids == [derived_link.id]

            graph = await build_provenance_graph(repository, run.id)
            relationships = {
                edge.kind
                for edge in graph.edges
                if edge.from_node_id == f"passage:{passage.id}"
                and edge.to_node_id == f"claim:{claim.id}"
            }
            assert relationships == {"ASSERTS", "SUPPORTS", "CONTRADICTS"}
        finally:
            researcher_module.create_qwen_chat_model = original_factory
            repository.close()

