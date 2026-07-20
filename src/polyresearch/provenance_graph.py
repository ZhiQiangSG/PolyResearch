"""Build a typed graph projection from the SQLite evidence ledger."""

from __future__ import annotations

import asyncio
from typing import Iterable
from uuid import UUID

from pydantic import BaseModel

from polyresearch.models import ProvenanceGraph, ProvenanceGraphNode
from polyresearch.repositories.base import EvidenceRepository


def _nodes(kind: str, artifacts: Iterable[BaseModel]) -> list[ProvenanceGraphNode]:
    return [
        ProvenanceGraphNode(
            node_id=f"{kind}:{artifact.id}",
            artifact_id=artifact.id,
            kind=kind,  # type: ignore[arg-type]
            attributes=artifact.model_dump(mode="json"),
        )
        for artifact in artifacts
    ]


async def build_provenance_graph(
    repository: EvidenceRepository, run_id: UUID
) -> ProvenanceGraph:
    """Project all requested durable run artifacts into typed provenance nodes."""
    (
        run,
        queries,
        sources,
        passages,
        translations,
        claims,
        evidence_links,
        verification_results,
        report_statements,
    ) = await asyncio.gather(
        repository.get_run(run_id),
        repository.list_query_records(run_id),
        repository.list_sources(run_id),
        repository.list_passages(run_id),
        repository.list_translations(run_id),
        repository.list_claims(run_id),
        repository.list_evidence_links(run_id),
        repository.list_verification_results(run_id),
        repository.list_report_statements(run_id),
    )
    nodes = _nodes("research_run", [run])
    nodes.extend(_nodes("query", queries))
    nodes.extend(_nodes("source", sources))
    nodes.extend(_nodes("passage", passages))
    nodes.extend(_nodes("translation", translations))
    nodes.extend(_nodes("claim", claims))
    nodes.extend(_nodes("evidence_link", evidence_links))
    nodes.extend(_nodes("verification_result", verification_results))
    nodes.extend(_nodes("report_statement", report_statements))
    return ProvenanceGraph(run_id=run_id, nodes=nodes)
