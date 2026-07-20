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


class ProvenanceGraphNode(BaseModel):
    """One immutable-ledger artifact projected as a typed graph node."""

    node_id: str = Field(min_length=1)
    artifact_id: UUID
    kind: GraphNodeKind
    attributes: dict[str, Any] = Field(default_factory=dict)


class ProvenanceGraph(BaseModel):
    """A graph-shaped view over a run; relationship edges are added separately."""

    run_id: UUID
    nodes: list[ProvenanceGraphNode] = Field(default_factory=list)
