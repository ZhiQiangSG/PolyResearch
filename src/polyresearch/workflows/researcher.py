"""Researcher subgraph for discovery, extraction, translation, and verification."""

import asyncio
import json
from typing import Literal, cast
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from polyresearch.evidence.claim_clustering import cluster_claims
from polyresearch.configuration import Configuration
from polyresearch.evidence.entity_resolution import resolve_claim_entities
from polyresearch.models import (
    Claim, ClaimExtractionDraft, ClaimExtractionResult, ClaimClusterVerificationResult,
    EvidenceLink, EvidencePassage, ResearcherOutputState, ResearcherState,
    SourceRecord, TranslationDraft, TranslationRecord, VerificationResult,
    VerificationStatus,
)
from polyresearch.nodes.provenance import (
    load_evidence_ledger as _load_evidence_ledger,
    persist_non_tavily_tool_outputs as _persist_non_tavily_tool_outputs,
    serialize_artifacts as _serialize_artifacts,
)
from polyresearch.prompts import claim_cluster_verification_prompt, research_system_prompt
from polyresearch.retrieval.source_ingestion import languages_match
from polyresearch.runtime.model_utils import create_qwen_chat_model
from polyresearch.retrieval.search_utils import select_citable_passages
from polyresearch.runtime.text_utils import get_today_str
from polyresearch.runtime.tool_registry import get_all_tools
from polyresearch.evidence.value_normalization import normalize_claim_values

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.
    
    This researcher is given a specific research topic by the supervisor and uses
    available tools (search, think_tool, MCP tools) to gather comprehensive information.
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
            "search API or add MCP tools to your configuration."
        )
    
    # Step 2: Configure the researcher model with tools
    research_model = create_qwen_chat_model(
        configurable,
        configurable.research_model,
        configurable.research_model_max_tokens,
        config,
    )
    
    researcher_prompt = research_system_prompt.format(date=get_today_str())
    
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
        return await tool.ainvoke(args, config)
    except Exception as e:
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
    2. Search tools (tavily_search, web_search) - Information gathering
    3. MCP tools - External tool integrations
    4. ResearchComplete - Signals completion of individual research task
    
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
    
    # Step 2: Handle other tool calls (search, MCP tools, etc.)
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
    context, sources, passages, _, _, _ = await _load_evidence_ledger(config)
    selected_passages = select_citable_passages(
        sources, passages, state.get("research_topic", "")
    )
    if not selected_passages:
        return {
            "sources": sources,
            "passages": passages,
            "claims": [],
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
            "passages": _serialize_artifacts(selected_passages, EvidencePassage),
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
    except Exception:
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
        except Exception:
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
    verified_claim_ids = {result.claim_id for result in existing_results}
    unverified_claims = [claim for claim in claims if claim.id not in verified_claim_ids]
    if not unverified_claims:
        return {
            "sources": sources,
            "passages": passages,
            "claims": claims,
            "verification_results": existing_results,
        }

    links_by_claim_id: dict[UUID, list[EvidenceLink]] = {}
    for link in evidence_links:
        links_by_claim_id.setdefault(link.claim_id, []).append(link)
    passages_by_id = {passage.id: passage for passage in passages}
    clusters: dict[UUID, list[Claim]] = {}
    for claim in unverified_claims:
        clusters.setdefault(claim.claim_cluster_id or claim.id, []).append(claim)
    results: list[VerificationResult] = []
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
        configurable = Configuration.from_runnable_config(config)
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
            }
        except Exception:
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
                    )
                    for claim in cluster_claims
                )
                continue
            assessments_by_claim_id = {
                assessment.claim_id: assessment for assessment in draft.claim_assessments
            }
            results.extend(
                VerificationResult(
                    claim_id=claim.id,
                    status=assessments_by_claim_id[claim.id].status,
                    confidence=assessments_by_claim_id[claim.id].confidence,
                    rationale=assessments_by_claim_id[claim.id].rationale,
                    evidence_link_ids=[link.id for link in links_by_claim_id.get(claim.id, [])],
                    disagreement_assessments=draft.disagreement_assessments,
                )
                for claim in cluster_claims
            )
    await context.repository.append_verification_results(context.run_id, results)
    return {
        "sources": sources,
        "passages": passages,
        "claims": claims,
        "verification_results": [*existing_results, *results],
    }

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

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")           # Entry point to researcher
researcher_builder.add_edge("extract_claims", "translate_claim_evidence")
researcher_builder.add_edge("translate_claim_evidence", "verify_claim_clusters")
researcher_builder.add_edge("verify_claim_clusters", END)

# Compatibility alias for callers that used the Milestone 5 node name.
verify_claims = verify_claim_clusters

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()
