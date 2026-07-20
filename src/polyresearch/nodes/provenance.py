"""Typed evidence-ledger helpers shared by LangGraph nodes."""

import asyncio
import json
import logging

from langchain_core.runnables import RunnableConfig

from polyresearch.models import (
    Claim,
    EvidencePassage,
    ProvenanceAttachment,
    SourceRecord,
)
from polyresearch.repositories import RunContext
from polyresearch.security import redact_secrets

logger = logging.getLogger(__name__)


async def persist_non_tavily_tool_outputs(
    config: RunnableConfig, tool_calls: list[dict], observations: list[object]
) -> None:
    """Retain non-Tavily raw tool output as audit-only provenance attachments."""
    try:
        context = RunContext.from_runnable_config(config)
    except ValueError:
        logger.debug("Skipping non-Tavily output persistence without run context")
        return

    attachments = [
        ProvenanceAttachment(
            run_id=context.run_id,
            provider="runtime_tool",
            tool_name=tool_call["name"],
            raw_output=redact_secrets(str(observation)),
        )
        for tool_call, observation in zip(tool_calls, observations)
        if tool_call["name"] != "tavily_search"
    ]
    if attachments:
        await context.repository.append_provenance_attachments(
            context.run_id, attachments
        )


async def load_evidence_ledger(config: RunnableConfig):
    """Load run-scoped typed artifacts, optionally limited to a research unit."""
    context = RunContext.from_runnable_config(config)
    repository = context.repository
    sources, passages, claims, evidence_links, verification_results = await asyncio.gather(
        repository.list_sources(context.run_id),
        repository.list_passages(context.run_id),
        repository.list_claims(context.run_id),
        repository.list_evidence_links(context.run_id),
        repository.list_verification_results(context.run_id),
    )
    if context.research_unit_id is None:
        return context, sources, passages, claims, evidence_links, verification_results

    scoped_sources = [
        source for source in sources if source.research_unit_id == context.research_unit_id
    ]
    scoped_source_ids = {source.id for source in scoped_sources}
    scoped_passages = [
        passage for passage in passages if passage.source_id in scoped_source_ids
    ]
    scoped_passage_ids = {passage.id for passage in scoped_passages}
    scoped_claims = [
        claim
        for claim in claims
        if set(claim.evidence_passage_ids).issubset(scoped_passage_ids)
    ]
    scoped_claim_ids = {claim.id for claim in scoped_claims}
    scoped_links = [
        link
        for link in evidence_links
        if link.claim_id in scoped_claim_ids and link.passage_id in scoped_passage_ids
    ]
    scoped_results = [
        result for result in verification_results if result.claim_id in scoped_claim_ids
    ]
    return (
        context,
        scoped_sources,
        scoped_passages,
        scoped_claims,
        scoped_links,
        scoped_results,
    )


def researcher_evidence_summary(observation: dict) -> str:
    """Render typed researcher artifacts for the supervisor's tool context."""
    claims = observation.get("claims", [])
    if not claims:
        return "No passage-linked claims were extracted from this research task."
    return json.dumps(
        {
            "type": "polyresearch_claims",
            "claims": [
                claim.model_dump(mode="json") if isinstance(claim, Claim) else claim
                for claim in claims
            ],
        },
        ensure_ascii=False,
    )


def serialize_artifacts(artifacts, model_type):
    """Serialize Pydantic artifacts returned as models or graph dictionaries."""
    return [
        (
            artifact.model_dump(mode="json")
            if isinstance(artifact, model_type)
            else model_type.model_validate(artifact).model_dump(mode="json")
        )
        for artifact in artifacts
    ]
