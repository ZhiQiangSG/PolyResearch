"""Build a typed graph projection from the SQLite evidence ledger."""

from __future__ import annotations

import asyncio
from typing import Iterable
from uuid import UUID

from pydantic import BaseModel

from polyresearch.models import (
    ProvenanceGraph,
    ProvenanceGraphEdge,
    ProvenanceGraphNode,
    ReportStatementEvidencePath,
)
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


def _edge(
    kind: str, from_node_id: str, to_node_id: str, *, attributes: dict | None = None
) -> ProvenanceGraphEdge:
    return ProvenanceGraphEdge(
        edge_id=f"{kind}:{from_node_id}->{to_node_id}",
        from_node_id=from_node_id,
        to_node_id=to_node_id,
        kind=kind,  # type: ignore[arg-type]
        attributes=attributes or {},
    )


def trace_report_statements_to_evidence(
    graph: ProvenanceGraph,
) -> dict[UUID, list[ReportStatementEvidencePath]]:
    """Traverse each report statement backward to its original evidence passages."""
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    rendered_by_statement: dict[str, list[ProvenanceGraphEdge]] = {}
    assertions_by_claim: dict[str, list[ProvenanceGraphEdge]] = {}
    for edge in graph.edges:
        if edge.kind == "RENDERED_AS":
            rendered_by_statement.setdefault(edge.to_node_id, []).append(edge)
        elif edge.kind == "ASSERTS":
            assertions_by_claim.setdefault(edge.to_node_id, []).append(edge)

    paths: dict[UUID, list[ReportStatementEvidencePath]] = {}
    for node in graph.nodes:
        if node.kind != "report_statement":
            continue
        statement_paths: list[ReportStatementEvidencePath] = []
        for rendered_edge in rendered_by_statement.get(node.node_id, []):
            claim_node = nodes_by_id.get(rendered_edge.from_node_id)
            if claim_node is None:
                continue
            for assertion_edge in assertions_by_claim.get(claim_node.node_id, []):
                passage_node = nodes_by_id.get(assertion_edge.from_node_id)
                if passage_node is None:
                    continue
                statement_paths.append(
                    ReportStatementEvidencePath(
                        report_statement_id=node.artifact_id,
                        claim_id=claim_node.artifact_id,
                        passage_id=passage_node.artifact_id,
                    )
                )
        paths[node.artifact_id] = statement_paths
    return paths


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
    source_ids_by_url = {}
    for source in sources:
        source_node_id = f"source:{source.id}"
        source_ids_by_url[source.canonical_url] = source_node_id
        if source.discovered_url:
            source_ids_by_url[source.discovered_url] = source_node_id

    edges: list[ProvenanceGraphEdge] = []
    for query in queries:
        if query.result_url and query.result_url in source_ids_by_url:
            edges.append(
                _edge(
                    "FOUND_BY",
                    f"query:{query.id}",
                    source_ids_by_url[query.result_url],
                )
            )
    for passage in passages:
        edges.append(_edge("CONTAINS", f"source:{passage.source_id}", f"passage:{passage.id}"))
    for translation in translations:
        edges.append(
            _edge("TRANSLATED_AS", f"passage:{translation.passage_id}", f"translation:{translation.id}")
        )
    for claim in claims:
        for passage_id in claim.evidence_passage_ids:
            edges.append(_edge("ASSERTS", f"passage:{passage_id}", f"claim:{claim.id}"))
    for link in evidence_links:
        relationship = {
            "supports": "SUPPORTS",
            "contradicts": "CONTRADICTS",
            "contextualizes": "CONTEXTUALIZES",
        }[link.relationship]
        edges.append(
            _edge(
                relationship,
                f"passage:{link.passage_id}",
                f"claim:{link.claim_id}",
                attributes={"evidence_link_id": str(link.id), "rationale": link.rationale},
            )
        )
    for result in verification_results:
        edges.append(
            _edge("VERIFIED_BY", f"claim:{result.claim_id}", f"verification_result:{result.id}")
        )
    for statement in report_statements:
        for claim_id in statement.claim_ids:
            edges.append(_edge("RENDERED_AS", f"claim:{claim_id}", f"report_statement:{statement.id}"))
    return ProvenanceGraph(run_id=run_id, nodes=nodes, edges=edges)
