"""SQLite-compatible in-memory projection of durable provenance artifacts."""

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


GraphNodeKind = Literal[
    "research_run",
    "query",
    "source",
    "passage",
    "translation",
    "claim",
    "evidence_link",
    "verification_result",
    "report_statement",
]
GraphEdgeKind = Literal[
    "FOUND_BY",
    "CONTAINS",
    "TRANSLATED_AS",
    "ASSERTS",
    "SUPPORTS",
    "CONTRADICTS",
    "CONTEXTUALIZES",
    "VERIFIED_BY",
    "RENDERED_AS",
]


class ProvenanceGraphNode(BaseModel):
    """One immutable-ledger artifact projected as a typed graph node."""

    node_id: str = Field(min_length=1)
    artifact_id: UUID
    kind: GraphNodeKind
    attributes: dict[str, Any] = Field(default_factory=dict)


class ProvenanceGraphEdge(BaseModel):
    """A directed, typed relationship derived from immutable ledger references."""

    edge_id: str = Field(min_length=1)
    from_node_id: str = Field(min_length=1)
    to_node_id: str = Field(min_length=1)
    kind: GraphEdgeKind
    attributes: dict[str, Any] = Field(default_factory=dict)


class ReportStatementEvidencePath(BaseModel):
    """One auditable report-statement → claim → original-passage traversal."""

    report_statement_id: UUID
    claim_id: UUID
    passage_id: UUID


class ReportEvidenceTrace(BaseModel):
    """One complete auditable path from rendered report text back to discovery."""

    report_statement_id: UUID
    claim_id: UUID
    evidence_passage_id: UUID
    source_id: UUID
    query_id: UUID
    translation_id: UUID | None = None


class ProvenanceDiagnostic(BaseModel):
    """An explicit gap found while tracing a rendered statement to evidence."""

    code: Literal["missing_source", "missing_query", "missing_expected_translation"]
    report_statement_id: UUID
    claim_id: UUID
    passage_id: UUID
    source_id: UUID | None = None
    message: str


class ProvenanceGraph(BaseModel):
    """A SQLite-compatible graph-shaped view over durable run artifacts."""

    run_id: UUID
    nodes: list[ProvenanceGraphNode] = Field(default_factory=list)
    edges: list[ProvenanceGraphEdge] = Field(default_factory=list)
