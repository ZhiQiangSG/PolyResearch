"""Regression tests for append-only conflict-resolution verification attempts."""

import importlib
import tempfile
import asyncio
from pathlib import Path
from uuid import UUID, uuid4

from polyresearch.evidence.provenance_graph import build_provenance_graph
from polyresearch.evidence.verification_results import latest_results_by_claim_id
from polyresearch.models import (
    Claim,
    ClaimClusterVerificationResult,
    EvidenceLink,
    EvidencePassage,
    ResearchRun,
    SourceRecord,
    VerificationResult,
    VerificationStatus,
)
from polyresearch.repositories import SqliteEvidenceRepository
from polyresearch.workflows.report_generator import _build_unresolved_disagreements


researcher_module = importlib.import_module("polyresearch.workflows.researcher")


def test_latest_verification_selection_uses_newest_attempt() -> None:
    claim_id = uuid4()
    first = VerificationResult(
        claim_id=claim_id,
        status=VerificationStatus.CONTRADICTED,
        confidence=0.3,
        rationale="Earlier attempt.",
        attempt_number=1,
    )
    latest = VerificationResult(
        claim_id=claim_id,
        status=VerificationStatus.SUPPORTED,
        confidence=0.9,
        rationale="Later attempt.",
        attempt_number=2,
    )

    assert latest_results_by_claim_id([latest, first]) == {claim_id: latest}


class _SecondAttemptVerifier:
    """Returns a resolved cluster after the bounded conflict-search pass."""

    def __init__(self, cluster_id: UUID, claim_ids: list[UUID]) -> None:
        self.cluster_id = cluster_id
        self.claim_ids = claim_ids

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
                        "cluster_rationale": "New official evidence resolves the earlier conflict.",
                        "claim_assessments": [
                            {
                                "claim_id": str(claim_id),
                                "status": "supported",
                                "confidence": 0.9,
                                "rationale": "The updated official record directly supports the claim.",
                            }
                            for claim_id in self.claim_ids
                        ],
                        "disagreement_assessments": [
                            {
                                "dimension": dimension,
                                "present": False,
                                "explanation": "The updated official record aligns the evidence scope.",
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


def test_conflict_resolution_reruns_cluster_as_a_versioned_attempt() -> None:
    asyncio.run(_verify_conflict_resolution_rerun())


async def _verify_conflict_resolution_rerun() -> None:
    with tempfile.TemporaryDirectory() as directory:
        repository = SqliteEvidenceRepository(Path(directory) / "research.db")
        run = ResearchRun(id=uuid4(), question="What changed?", output_language="en")
        cluster_id = uuid4()
        source = SourceRecord(canonical_url="https://official.example/record", title="Record")
        passages = [
            EvidencePassage(source_id=source.id, text=text, locator=f"p{index}")
            for index, text in enumerate(("The policy began in January.", "The policy began in January."), 1)
        ]
        claims = [
            Claim(
                statement=f"The policy began in January ({index}).",
                evidence_passage_ids=[passage.id],
                extraction_confidence=0.8,
                claim_cluster_id=cluster_id,
            )
            for index, passage in enumerate(passages, 1)
        ]
        links = [
            EvidenceLink(claim_id=claim.id, passage_id=passage.id, relationship="supports")
            for claim, passage in zip(claims, passages, strict=True)
        ]
        first_attempts = [
            VerificationResult(
                claim_id=claim.id,
                status=VerificationStatus.CONTRADICTED,
                confidence=0.35,
                rationale="The first retrieval contained conflicting evidence.",
                evidence_link_ids=[link.id],
            )
            for claim, link in zip(claims, links, strict=True)
        ]
        original_factory = researcher_module.create_qwen_chat_model
        researcher_module.create_qwen_chat_model = lambda *args, **kwargs: _SecondAttemptVerifier(
            cluster_id, [claim.id for claim in claims]
        )
        try:
            await repository.create_run(run)
            await repository.append_sources(run.id, [source])
            await repository.append_passages(run.id, passages)
            await repository.append_claims(run.id, claims)
            await repository.append_evidence_links(run.id, links)
            await repository.append_verification_results(run.id, first_attempts)

            result = await researcher_module.verify_claim_clusters(
                {"claim_cluster_ids_to_reverify": [cluster_id]},
                {"configurable": {"run_id": str(run.id), "evidence_repository": repository}},
            )

            attempts = await repository.list_verification_results(run.id)
            assert len(attempts) == 4
            second_attempts = [attempt for attempt in attempts if attempt.attempt_number == 2]
            assert len(second_attempts) == 2
            assert {attempt.status for attempt in second_attempts} == {VerificationStatus.SUPPORTED}
            assert {attempt.trigger for attempt in second_attempts} == {"conflict_resolution"}
            assert {
                attempt.supersedes_verification_result_id for attempt in second_attempts
            } == {attempt.id for attempt in first_attempts}
            assert len(result["verification_results"]) == 4
            assert not _build_unresolved_disagreements(
                claims=claims, verification_results=attempts
            ), "Superseded conflict attempts must not keep the report unresolved."

            graph = await build_provenance_graph(repository, run.id)
            assert len([edge for edge in graph.edges if edge.kind == "SUPERSEDES"]) == 2
        finally:
            researcher_module.create_qwen_chat_model = original_factory
            repository.close()
