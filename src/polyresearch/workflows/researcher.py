"""Researcher subgraph for discovery, extraction, translation, and verification."""

import asyncio
import json
import logging
from typing import Literal, cast
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from polyresearch.evidence.claim_clustering import cluster_claims
from polyresearch.configuration import Configuration
from polyresearch.evidence.entity_resolution import resolve_claim_entities
from polyresearch.evidence.verification_confidence import verification_confidence
from polyresearch.models import (
    Claim, ClaimExtractionDraft, ClaimExtractionResult, ClaimClusterVerificationResult,
    EvidenceLink, EvidencePassage, ResearcherOutputState, ResearcherState,
    ResearchPlan, SourceRecord, TranslationDraft, TranslationRecord, VerificationResult,
    VerificationStatus,
)
from polyresearch.nodes.provenance import (
    load_evidence_ledger as _load_evidence_ledger,
    persist_non_tavily_tool_outputs as _persist_non_tavily_tool_outputs,
    serialize_artifacts as _serialize_artifacts,
)
from polyresearch.prompts import (
    CLAIM_CLUSTER_VERIFICATION_PROMPT_VERSION,
    claim_cluster_verification_prompt,
    research_system_prompt,
)

from polyresearch.retrieval.source_ingestion import languages_match
from polyresearch.runtime.model_utils import create_qwen_chat_model
from polyresearch.retrieval.search_utils import select_citable_passages
from polyresearch.runtime.text_utils import get_today_str
from polyresearch.runtime.tool_registry import get_all_tools
from polyresearch.runtime.retry import retry_async
from polyresearch.security import redacted_exception_info, redact_prompt_injection
from polyresearch.evidence.value_normalization import normalize_claim_values
from polyresearch.retrieval.search_providers import planned_web_search

logger = logging.getLogger(__name__)


async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.
    
    This researcher is given a specific research topic by the supervisor and uses
    available routed search and reflection tools to gather comprehensive information.
    It can use think_tool for strategic planning between searches.
    
    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability
        
    Returns:
        Command to proceed to researcher_tools for tool execution
    """
    # Step 1: Load configuration and validate tool availability
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    
    # Get all available research tools (search, MCP, think_tool)
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search provider configuration."
        )
    
    # Step 2: Configure the researcher model with tools
    research_model = create_qwen_chat_model(
        configurable,
        configurable.research_model,
        configurable.research_model_max_tokens,
        config,
    )
    
    researcher_prompt = research_system_prompt.format(date=get_today_str())
    evidence_task = state.get("evidence_task")
    if evidence_task is not None:
        researcher_prompt += "\n\n<EvidenceTask>\n" + evidence_task.model_dump_json() + "\n</EvidenceTask>"
    
    # Configure model with tools, retry logic, and settings
    research_model = (
        research_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )
    
    # Step 3: Generate researcher response with system context
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)
    
    # Step 4: Update state and proceed to tool execution
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

# Tool Execution Helper Function
async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    try:
        configurable = Configuration.from_runnable_config(config)
        return await retry_async(
            lambda: tool.ainvoke(args, config),
            attempts=configurable.max_structured_output_retries,
        )
    except Exception as e:
        logger.warning(
            "Researcher tool execution failed",
            extra={"operation": "execute_tool", "tool_name": getattr(tool, "name", type(tool).__name__)},
            exc_info=redacted_exception_info(e),
        )
        return f"Error executing tool: {str(e)}"


def unknown_tool_observation(tool_name: str) -> str:
    """Return a recoverable observation when a model calls an unavailable tool."""
    return (
        f"Error executing tool: '{tool_name}' is unavailable in this research run. "
        "Use one of the tools currently provided."
    )


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "extract_claims"]]:
    """Execute tools called by the researcher, including search tools and strategic thinking.
    
    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. planned_web_search - Provider-routed evidence discovery
    3. ResearchComplete - Signals completion of individual research task
    
    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings
        
    Returns:
        Command to either continue research loop or extract typed claims
    """
    # Step 1: Extract current state and check early exit conditions
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    # Early exit if no tool calls were made (including native web search)
    has_tool_calls = bool(most_recent_message.tool_calls)
    
    if not has_tool_calls:
        return Command(goto="extract_claims")
    
    # Step 2: Handle routed search and completion tool calls.
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool 
        for tool in tools
    }
    
    # Execute all tool calls in parallel
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(
            tools_by_name[tool_call["name"]], tool_call.get("args", {}), config
        )
        if tool_call["name"] in tools_by_name
        else asyncio.sleep(0, result=unknown_tool_observation(tool_call["name"]))
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)
    await _persist_non_tavily_tool_outputs(config, tool_calls, observations)
    
    # Create tool messages from execution results
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) 
        for observation, tool_call in zip(observations, tool_calls)
    ]
    
    # Step 3: Check late exit conditions (after processing tools)
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    if exceeded_iterations or research_complete_called:
        # End research and extract claims from original passages.
        return Command(
            goto="extract_claims",
            update={"researcher_messages": tool_outputs}
        )
    
    # Continue research loop with tool results
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )

async def extract_claims(state: ResearcherState, config: RunnableConfig):
    """Extract passage-linked claims from the researcher's typed evidence.
    
    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings
        
    Returns:
        Typed source, passage, claim, and verification-result collections
    """
    # Step 1: Load citable source passages from the durable evidence ledger.
    context, sources, passages, existing_claims, _, _ = await _load_evidence_ledger(config)
    selected_passages = select_citable_passages(
        sources, passages, state.get("research_topic", "")
    )
    extracted_passage_ids = {
        passage_id for claim in existing_claims for passage_id in claim.evidence_passage_ids
    }
    selected_passages = [
        passage for passage in selected_passages if passage.id not in extracted_passage_ids
    ]
    if not selected_passages:
        return {
            "sources": sources,
            "passages": passages,
            "claims": existing_claims,
            "verification_results": [],
        }

    # Step 2: Configure structured claim extraction.
    configurable = Configuration.from_runnable_config(config)
    claim_extractor = create_qwen_chat_model(
        configurable,
        configurable.compression_model,
        configurable.compression_model_max_tokens,
        config,
    ).with_structured_output(ClaimExtractionResult).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    )
    extraction_prompt = (
        "Extract only atomic, falsifiable claims directly supported by the supplied "
        "selected evidence passages. For every claim emit: an atomic proposition; "
        "original-language wording when available; normalized wording in the requested "
        "output language; entities, quantities, dates, locations, and a bounded scope; "
        "qualifiers and modality; extraction confidence; and one or more exact passage "
        "IDs. Preserve original values and terminology. Do not invent sources, passage "
        "IDs, normalizations, or verification results. For entities, include aliases, "
        "native-script variants, transliterations, and historical names when supported; "
        "leave uncertain mappings explicit. Preserve every original date, currency, "
        "number, and unit alongside any proposed normalized form. Raw tool output is audit-only."
    )
    evidence_ledger = json.dumps(
        {
            "sources": _serialize_artifacts(sources, SourceRecord),
            "output_language": config.get("configurable", {}).get("output_language", "en"),
            "passages": [
                passage.model_copy(update={"text": redact_prompt_injection(passage.text)}).model_dump(mode="json")
                for passage in selected_passages
            ],
        },
        ensure_ascii=False,
    )

    try:
        response = cast(
            ClaimExtractionResult,
            await claim_extractor.ainvoke(
                [
                    SystemMessage(content=extraction_prompt),
                    HumanMessage(
                        content=f"<EvidenceLedger>\n{evidence_ledger}\n</EvidenceLedger>"
                    ),
                ]
            ),
        )
    except Exception as error:
        logger.warning(
            "Claim extraction failed; returning no extracted claims",
            extra={"operation": "extract_claims", "run_id": str(context.run_id)},
            exc_info=redacted_exception_info(error),
        )
        return {
            "sources": sources,
            "passages": passages,
            "claims": [],
            "verification_results": [],
        }

    known_passage_ids = {passage.id for passage in selected_passages}
    claims = [
        Claim(
            id=draft.id,
            statement=draft.normalized_statement,
            atomic_proposition=draft.atomic_proposition,
            original_wording=draft.original_wording,
            entities=draft.entities,
            quantities=draft.quantities,
            dates=draft.dates,
            locations=draft.locations,
            scope=draft.scope,
            qualifiers=draft.qualifiers,
            modality=draft.modality,
            evidence_passage_ids=draft.evidence_passage_ids,
            extraction_confidence=draft.extraction_confidence,
        )
        for draft in response.claims
        if set(draft.evidence_passage_ids).issubset(known_passage_ids)
    ]
    claims = cluster_claims(resolve_claim_entities(normalize_claim_values(claims)))
    evidence_links = [
        EvidenceLink(
            claim_id=claim.id,
            passage_id=passage_id,
            relationship="supports",
        )
        for claim in claims
        for passage_id in claim.evidence_passage_ids
    ]
    await context.repository.append_claims(context.run_id, claims)
    await context.repository.append_evidence_links(context.run_id, evidence_links)
    return {
        "sources": sources,
        "passages": passages,
        "claims": claims,
        "verification_results": [],
    }


async def translate_claim_evidence(state: ResearcherState, config: RunnableConfig):
    """Persist translations only for claim evidence needed in another output language."""
    context, sources, passages, claims, _, verification_results = await _load_evidence_ledger(
        config
    )
    output_language = config.get("configurable", {}).get("output_language", "en")
    required_passage_ids = {
        passage_id for claim in claims for passage_id in claim.evidence_passage_ids
    }
    translation_candidates = [
        passage
        for passage in passages
        if passage.id in required_passage_ids
        and passage.original_language
        and not languages_match(passage.original_language, output_language)
    ]
    if not translation_candidates:
        return {
            "sources": sources,
            "passages": passages,
            "claims": claims,
            "verification_results": verification_results,
        }

    existing_translations = await context.repository.list_translations(context.run_id)
    existing_keys = {
        (translation.passage_id, translation.target_language, translation.source_original_text_hash)
        for translation in existing_translations
    }
    configurable = Configuration.from_runnable_config(config)
    translator = create_qwen_chat_model(
        configurable,
        configurable.compression_model,
        configurable.compression_model_max_tokens,
        config,
    ).with_structured_output(TranslationDraft).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    )
    translations: list[TranslationRecord] = []
    for passage in translation_candidates:
        key = (passage.id, output_language, passage.original_text_hash)
        if key in existing_keys:
            continue
        try:
            draft = cast(
                TranslationDraft,
                await translator.ainvoke(
                    [
                        SystemMessage(
                            content=(
                                "Translate the supplied original-language evidence passage. "
                                "Do not summarize, omit qualifiers, or add information."
                            )
                        ),
                        HumanMessage(
                            content=(
                                f"Target language: {output_language}\n"
                                f"Passage ID: {passage.id}\n"
                                f"Original text:\n{passage.text}"
                            )
                        ),
                    ]
                ),
            )
        except Exception as error:
            logger.warning(
                "Evidence translation failed; retaining original-language passage",
                extra={"operation": "translate_evidence", "passage_id": str(passage.id), "target_language": output_language},
                exc_info=redacted_exception_info(error),
            )
            # Translation is an optional derivative; retain original evidence when
            # a model call fails rather than replacing it with an uncertain string.
            continue
        translations.append(
            TranslationRecord(
                passage_id=passage.id,
                translated_text=draft.translated_text,
                target_language=output_language,
                model_id=configurable.compression_model,
                prompt_version="translation-v1",
                confidence=draft.confidence,
                source_original_text_hash=passage.original_text_hash,
            )
        )
    if translations:
        await context.repository.append_translations(context.run_id, translations)
    return {
        "sources": sources,
        "passages": passages,
        "claims": claims,
        "verification_results": verification_results,
    }


async def verify_claim_clusters(state: ResearcherState, config: RunnableConfig):
    """Verify deterministic claim clusters and persist claim-addressable results."""
    context, sources, passages, claims, evidence_links, existing_results = (
        await _load_evidence_ledger(config)
    )
    translations = await context.repository.list_translations(context.run_id)
    output_language = config.get("configurable", {}).get("output_language", "en")
    latest_results_by_claim_id = _latest_results_by_claim_id(existing_results)
    requested_cluster_ids = {
        UUID(str(cluster_id))
        for cluster_id in state.get("claim_cluster_ids_to_reverify", [])
    }
    claims_to_verify = [
        claim
        for claim in claims
        if claim.id not in latest_results_by_claim_id
        or (claim.claim_cluster_id or claim.id) in requested_cluster_ids
    ]
    if not claims_to_verify:
        return {
            "sources": sources,
            "passages": passages,
            "claims": claims,
            "verification_results": existing_results,
        }
    configurable = Configuration.from_runnable_config(config)
    verifier_model_id = configurable.compression_model

    links_by_claim_id: dict[UUID, list[EvidenceLink]] = {}
    for link in evidence_links:
        # A re-verification always evaluates the original extraction links.  Prior
        # verification links remain immutable provenance rather than becoming new
        # input evidence and recursively multiplying the assessment set.
        if link.origin == "claim_extraction":
            links_by_claim_id.setdefault(link.claim_id, []).append(link)
    passages_by_id = {passage.id: passage for passage in passages}
    clusters: dict[UUID, list[Claim]] = {}
    for claim in claims_to_verify:
        clusters.setdefault(claim.claim_cluster_id or claim.id, []).append(claim)
    results: list[VerificationResult] = []
    verification_links: list[EvidenceLink] = []
    verifiable_clusters = {
        cluster_id: cluster_claims
        for cluster_id, cluster_claims in clusters.items()
        if any(
            links_by_claim_id.get(claim.id)
            and all(link.passage_id in passages_by_id for link in links_by_claim_id[claim.id])
            for claim in cluster_claims
        )
    }
    for cluster_id, cluster_claims in clusters.items():
        if cluster_id in verifiable_clusters:
            continue
        results.extend(
            VerificationResult(
                claim_id=claim.id,
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                confidence=0.0,
                rationale="No persisted claim-to-passage evidence is available for this claim cluster.",
                attempt_number=_next_attempt_number(claim.id, latest_results_by_claim_id),
                supersedes_verification_result_id=_superseded_result_id(
                    claim.id, latest_results_by_claim_id
                ),
                trigger=_verification_trigger(claim, requested_cluster_ids),
                verifier_model_id=verifier_model_id,
                verifier_prompt_version=CLAIM_CLUSTER_VERIFICATION_PROMPT_VERSION,
            )
            for claim in cluster_claims
        )
    if verifiable_clusters:
        verifiable_claim_ids = {
            claim.id for cluster_claims in verifiable_clusters.values() for claim in cluster_claims
        }
        relevant_passage_ids = {
            link.passage_id
            for claim_id in verifiable_claim_ids
            for link in links_by_claim_id[claim_id]
        }
        relevant_passages = [
            passage for passage in passages if passage.id in relevant_passage_ids
        ]
        relevant_source_ids = {passage.source_id for passage in relevant_passages}
        verification_ledger = json.dumps(
            {
                "clusters": [
                    {
                        "cluster_id": str(cluster_id),
                        "claims": _serialize_artifacts(cluster_claims, Claim),
                        "evidence_links": _serialize_artifacts(
                            [
                                link
                                for claim in cluster_claims
                                for link in links_by_claim_id.get(claim.id, [])
                            ],
                            EvidenceLink,
                        ),
                    }
                    for cluster_id, cluster_claims in verifiable_clusters.items()
                ],
                "passages": _serialize_artifacts(relevant_passages, EvidencePassage),
                "sources": _serialize_artifacts(
                    [source for source in sources if source.id in relevant_source_ids],
                    SourceRecord,
                ),
            },
            ensure_ascii=False,
        )
        verifier = create_qwen_chat_model(
            configurable,
            configurable.compression_model,
            configurable.compression_model_max_tokens,
            config,
        ).with_structured_output(ClaimClusterVerificationResult).with_retry(
            stop_after_attempt=configurable.max_structured_output_retries
        )
        try:
            response = cast(
                ClaimClusterVerificationResult,
                await verifier.ainvoke(
                    [HumanMessage(content=claim_cluster_verification_prompt.format(
                        verification_ledger=verification_ledger
                    ))]
                ),
            )
            drafts_by_cluster_id = {
                draft.cluster_id: draft
                for draft in response.clusters
                if draft.cluster_id in verifiable_clusters
                and {assessment.claim_id for assessment in draft.claim_assessments}
                == {claim.id for claim in verifiable_clusters[draft.cluster_id]}
                and _has_complete_evidence_assessments(
                    draft,
                    verifiable_clusters[draft.cluster_id],
                    links_by_claim_id,
                )
            }
        except Exception as error:
            logger.warning(
                "Claim-cluster verification failed; marking clusters unresolved",
                extra={"operation": "verify_claim_clusters", "run_id": str(context.run_id)},
                exc_info=redacted_exception_info(error),
            )
            drafts_by_cluster_id = {}
        for cluster_id, cluster_claims in verifiable_clusters.items():
            draft = drafts_by_cluster_id.get(cluster_id)
            if draft is None:
                results.extend(
                    VerificationResult(
                        claim_id=claim.id,
                        status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                        confidence=0.0,
                        rationale="Verification did not return a valid result for this claim cluster.",
                        evidence_link_ids=[link.id for link in links_by_claim_id.get(claim.id, [])],
                        attempt_number=_next_attempt_number(claim.id, latest_results_by_claim_id),
                        supersedes_verification_result_id=_superseded_result_id(
                            claim.id, latest_results_by_claim_id
                        ),
                        trigger=_verification_trigger(claim, requested_cluster_ids),
                        verifier_model_id=verifier_model_id,
                        verifier_prompt_version=CLAIM_CLUSTER_VERIFICATION_PROMPT_VERSION,
                    )
                    for claim in cluster_claims
                )
                continue
            assessments_by_claim_id = {
                assessment.claim_id: assessment for assessment in draft.claim_assessments
            }
            for claim in cluster_claims:
                assessment = assessments_by_claim_id[claim.id]
                outcome_links = _verification_outcome_links(
                    claim=claim,
                    assessment=assessment,
                    input_links=links_by_claim_id.get(claim.id, []),
                )
                result = _verification_result_with_confidence(
                    claim=claim,
                    assessment=assessment,
                    evidence_links=outcome_links,
                    passages=passages,
                    sources=sources,
                    translations=translations,
                    output_language=output_language,
                    disagreement_assessments=draft.disagreement_assessments,
                    previous_result=latest_results_by_claim_id.get(claim.id),
                    trigger=_verification_trigger(claim, requested_cluster_ids),
                    verifier_model_id=verifier_model_id,
                    verifier_prompt_version=CLAIM_CLUSTER_VERIFICATION_PROMPT_VERSION,
                )
                outcome_links = [
                    link.model_copy(update={"verification_result_id": result.id})
                    for link in outcome_links
                ]
                result = result.model_copy(
                    update={"evidence_link_ids": [link.id for link in outcome_links]}
                )
                results.append(result)
                verification_links.extend(outcome_links)
    if verification_links:
        await context.repository.append_evidence_links(context.run_id, verification_links)
    await context.repository.append_verification_results(context.run_id, results)
    return {
        "sources": sources,
        "passages": passages,
        "claims": claims,
        "verification_results": [*existing_results, *results],
    }


def _verification_result_with_confidence(
    *,
    claim: Claim,
    assessment,
    evidence_links: list[EvidenceLink],
    passages: list[EvidencePassage],
    sources: list[SourceRecord],
    translations: list[TranslationRecord],
    output_language: str,
    disagreement_assessments,
    verifier_model_id: str,
    verifier_prompt_version: str,
    previous_result: VerificationResult | None = None,
    trigger: Literal["initial_verification", "conflict_resolution"] = "initial_verification",
) -> VerificationResult:
    """Bind a Qwen claim classification to conservative evidence-derived confidence."""
    confidence, confidence_factors, independent_source_count = verification_confidence(
        claim=claim,
        status=assessment.status,
        model_confidence=assessment.confidence,
        evidence_links=evidence_links,
        passages=passages,
        sources=sources,
        translations=translations,
        output_language=output_language,
        disagreement_assessments=disagreement_assessments,
    )
    return VerificationResult(
        claim_id=claim.id,
        status=assessment.status,
        confidence=confidence,
        rationale=assessment.rationale,
        disagreement_assessments=disagreement_assessments,
        confidence_factors=confidence_factors,
        independent_source_count=independent_source_count,
        attempt_number=(previous_result.attempt_number + 1) if previous_result else 1,
        supersedes_verification_result_id=previous_result.id if previous_result else None,
        trigger=trigger,
        verifier_model_id=verifier_model_id,
        verifier_prompt_version=verifier_prompt_version,
    )


def _has_complete_evidence_assessments(
    draft,
    cluster_claims: list[Claim],
    links_by_claim_id: dict[UUID, list[EvidenceLink]],
) -> bool:
    """Reject partial verifier link mappings while accepting legacy empty outputs."""
    assessments_by_claim_id = {
        assessment.claim_id: assessment for assessment in draft.claim_assessments
    }
    for claim in cluster_claims:
        evidence_assessments = assessments_by_claim_id[claim.id].evidence_assessments
        if not evidence_assessments:
            continue
        expected = {link.id for link in links_by_claim_id.get(claim.id, [])}
        actual = {assessment.evidence_link_id for assessment in evidence_assessments}
        if actual != expected or len(evidence_assessments) != len(expected):
            return False
    return True


def _verification_outcome_links(
    *,
    claim: Claim,
    assessment,
    input_links: list[EvidenceLink],
) -> list[EvidenceLink]:
    """Append verifier-derived links without overwriting extraction provenance."""
    relationship_by_input_id = {
        evidence_assessment.evidence_link_id: evidence_assessment
        for evidence_assessment in assessment.evidence_assessments
    }
    fallback_relationship = {
        VerificationStatus.SUPPORTED: "supports",
        VerificationStatus.CONTRADICTED: "contradicts",
        VerificationStatus.PARTIALLY_SUPPORTED: "contextualizes",
        VerificationStatus.INSUFFICIENT_EVIDENCE: "contextualizes",
        VerificationStatus.OUTDATED: "contextualizes",
        VerificationStatus.NOT_COMPARABLE: "contextualizes",
    }[assessment.status]
    return [
        EvidenceLink(
            claim_id=claim.id,
            passage_id=input_link.passage_id,
            relationship=(
                evidence_assessment.relationship
                if (evidence_assessment := relationship_by_input_id.get(input_link.id))
                else fallback_relationship
            ),
            rationale=(
                evidence_assessment.rationale
                if evidence_assessment
                else assessment.rationale
            ),
            origin="verification",
        )
        for input_link in input_links
    ]
def _latest_results_by_claim_id(
    results: list[VerificationResult],
) -> dict[UUID, VerificationResult]:
    """Select the newest immutable verification attempt for each claim."""
    latest: dict[UUID, VerificationResult] = {}
    for result in results:
        current = latest.get(result.claim_id)
        if current is None or (result.attempt_number, result.created_at, str(result.id)) > (
            current.attempt_number,
            current.created_at,
            str(current.id),
        ):
            latest[result.claim_id] = result
    return latest


def _next_attempt_number(
    claim_id: UUID, latest_results: dict[UUID, VerificationResult]
) -> int:
    previous = latest_results.get(claim_id)
    return previous.attempt_number + 1 if previous else 1


def _superseded_result_id(
    claim_id: UUID, latest_results: dict[UUID, VerificationResult]
) -> UUID | None:
    previous = latest_results.get(claim_id)
    return previous.id if previous else None


def _verification_trigger(
    claim: Claim, requested_cluster_ids: set[UUID]
) -> Literal["initial_verification", "conflict_resolution"]:
    return (
        "conflict_resolution"
        if (claim.claim_cluster_id or claim.id) in requested_cluster_ids
        else "initial_verification"
    )


async def resolve_conflicts(state: ResearcherState, config: RunnableConfig):
    """Perform one bounded primary/official-source retrieval pass for conflicts."""
    if state.get("conflict_resolution_attempted", False):
        return Command(goto="__end__")
    context, _, _, claims, _, results = await _load_evidence_ledger(config)
    latest_results = _latest_results_by_claim_id(results)
    has_material_conflict = any(
        result.status is VerificationStatus.CONTRADICTED
        or any(
            assessment.dimension.value == "genuinely_conflicting_evidence"
            and assessment.present
            for assessment in result.disagreement_assessments
        )
        for result in latest_results.values()
    )
    if not has_material_conflict:
        return Command(goto="__end__", update={"conflict_resolution_attempted": True})

    raw_plan = config.get("configurable", {}).get("research_plan")
    if raw_plan is None:
        plans = await context.repository.list_research_plans(context.run_id)
        raw_plan = plans[-1] if plans else None
    if raw_plan is None:
        return Command(goto="__end__", update={"conflict_resolution_attempted": True})
    plan = raw_plan if isinstance(raw_plan, ResearchPlan) else ResearchPlan.model_validate(raw_plan)
    requests = []
    for language in plan.ranked_languages:
        source_type = next(
            (
                candidate
                for candidate in ("official", "primary")
                if candidate in language.expected_source_types
            ),
            None,
        )
        variants = plan.query_variants.get(language.language, [])
        if source_type and variants:
            requests.append((variants[0], language.language, source_type))
    selected_requests = requests[:3]
    if not selected_requests:
        return Command(goto="__end__", update={"conflict_resolution_attempted": True})
    for query, language, source_type in selected_requests:
        try:
            await planned_web_search.coroutine(
                query=query,
                language=language,
                target_source_type=source_type,
                query_rationale=(
                    "Conflict resolution: seek the strongest planned primary or official "
                    "evidence before reporting consensus."
                ),
                config=config,
            )
        except Exception as error:
            logger.warning(
                "Conflict-resolution search failed; retaining unresolved status",
                extra={"operation": "resolve_conflicts", "query_language": language, "target_source_type": source_type},
                exc_info=redacted_exception_info(error),
            )
            # The failed search is captured in query provenance by its provider path;
            # do not turn unresolved conflict into a fabricated consensus.
            continue
    return Command(
        goto="extract_claims",
        update={
            "conflict_resolution_attempted": True,
            "claim_cluster_ids_to_reverify": list({
                claim.claim_cluster_id or claim.id
                for claim in claims
                if (result := latest_results.get(claim.id))
                and (
                    result.status is VerificationStatus.CONTRADICTED
                    or any(
                        assessment.dimension.value == "genuinely_conflicting_evidence"
                        and assessment.present
                        for assessment in result.disagreement_assessments
                    )
                )
            }),
        },
    )

# Researcher Subgraph Construction
# Creates individual researcher workflow for conducting focused research on specific topics
researcher_builder = StateGraph(
    ResearcherState, 
    output_schema=ResearcherOutputState, 
    context_schema=Configuration
)

# Add researcher nodes for research execution and claim extraction.
researcher_builder.add_node("researcher", researcher)                 # Main researcher logic
researcher_builder.add_node("researcher_tools", researcher_tools)     # Tool execution handler
researcher_builder.add_node("extract_claims", extract_claims)         # Claim extraction
researcher_builder.add_node("translate_claim_evidence", translate_claim_evidence)
researcher_builder.add_node("verify_claim_clusters", verify_claim_clusters)
researcher_builder.add_node("resolve_conflicts", resolve_conflicts)

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")           # Entry point to researcher
researcher_builder.add_edge("extract_claims", "translate_claim_evidence")
researcher_builder.add_edge("translate_claim_evidence", "verify_claim_clusters")
researcher_builder.add_edge("verify_claim_clusters", "resolve_conflicts")
researcher_builder.add_edge("resolve_conflicts", END)

# Compatibility alias for callers that used the Milestone 5 node name.
verify_claims = verify_claim_clusters

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()
