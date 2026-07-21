"""Top-level PolyResearch workflow nodes and graph assembly."""

import json
import logging
from datetime import datetime, timezone
from typing import Literal, cast
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from polyresearch.configuration import Configuration
from polyresearch.models import (
    AgentInputState, AgentState, Claim, ClarifyWithUser, EvidencePassage,
    LanguageDecision, LanguageExpansionDecision, ResearchPlan, ResearchQuestion,
    ResearchRun, SourceRecord, VerificationResult,
)
from polyresearch.nodes.provenance import (
    load_evidence_ledger as _load_evidence_ledger,
    serialize_artifacts as _serialize_artifacts,
)
from polyresearch.prompts import (
    clarify_with_user_instructions, language_gap_analysis_prompt,
    lead_researcher_prompt, multilingual_planner_prompt,
    transform_messages_into_research_topic_prompt,
)
from polyresearch.repositories import RunContext
from polyresearch.runtime.model_utils import create_qwen_chat_model
from polyresearch.runtime.text_utils import get_today_str
from polyresearch.security import redacted_exception_info
from polyresearch.workflows.report_generator import final_report_generation
from polyresearch.evidence.report_qa import validate_report_statements
from polyresearch.workflows.supervisor import supervisor_subgraph

logger = logging.getLogger(__name__)

async def initialize_research_run(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["clarify_with_user"]]:
    """Create the durable run record before any model or tool work begins."""
    context = RunContext.from_runnable_config(config)
    configurable = config.get("configurable", {})
    runtime = Configuration.from_runnable_config(config)
    question = get_buffer_string(state.get("messages", [])).strip()
    run = ResearchRun(
        id=context.run_id,
        question=question,
        output_language=configurable.get("output_language", "en"),
        model_ids={
            "research": runtime.research_model,
            "claim_extraction": runtime.compression_model,
            "verification": runtime.compression_model,
            "final_report": runtime.final_report_model,
        },
        prompt_versions={
            "clarification": "clarify_with_user_v1",
            "research_brief": "research_brief_v1",
            "multilingual_planner": "multilingual_planner_v1",
            "claim_extraction": "claim_extraction_v1",
            "verification": "claim-cluster-verification-v2",
            "report_outline": "report_outline_generation_v1",
            "report_prose": "report_prose_generation_v1",
        },
        provider_routing={"zh": "bailian_web_search", "other_selected_languages": "tavily"},
        retrieval_started_at=datetime.now(timezone.utc),
    )
    await context.repository.create_run(run)
    return Command(goto="clarify_with_user", update={"run_id": context.run_id})


async def clarify_with_user(
    state: AgentState,
    config: RunnableConfig
) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is 
    unclear.
    
    This function determines whether the user's request needs clarification before 
    proceeding with research. If clarification is disabled or not needed, it proceeds
    directly to research.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    # Step 1: Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # Skip clarification step and proceed directly to research
        return Command(goto="write_research_brief")
    
    # Step 2: Prepare the model for structured clarification analysis
    messages = state["messages"]
    clarification_model = create_qwen_chat_model(
        configurable,
        configurable.research_model,
        configurable.research_model_max_tokens,
        config,
    )
    
    # Configure model with structured output and retry logic
    clarification_model = (
        clarification_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )
    
    # Step 3: Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    response = cast(
        ClarifyWithUser,
        await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    )
    
    # Step 4: Route based on clarification analysis
    if response.need_clarification:
        # End with clarifying question for user
        return Command(
            goto="__end__", 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # Proceed to research with verification message
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["multilingual_planner"]]:
    """Transform user messages into a structured research brief.
    
    This function analyzes the user's messages and generates a focused research brief
    that will guide the multilingual planning step.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to multilingual planning
    """
    # Step 1: Set up the research model for structured output
    configurable = Configuration.from_runnable_config(config)
    research_model = create_qwen_chat_model(
        configurable,
        configurable.research_model,
        configurable.research_model_max_tokens,
        config,
    )
    
    # Configure model for structured research question generation
    research_model = (
        research_model
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )
    
    # Step 2: Generate structured research brief from user messages
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = cast(
        ResearchQuestion,
        await research_model.ainvoke([HumanMessage(content=prompt_content)])
    )
    
    # The multilingual planner owns supervisor initialization so the durable plan
    # is present in the supervisor's first context window.
    return Command(
        goto="multilingual_planner",
        update={"research_brief": response.research_brief},
    )


async def multilingual_planner(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["provider_routed_discovery", "__end__"]]:
    """Create and persist Qwen's adaptive, structured multilingual research plan."""
    configurable = Configuration.from_runnable_config(config)
    context = RunContext.from_runnable_config(config)
    planner_model = create_qwen_chat_model(
        configurable,
        configurable.research_model,
        configurable.planner_model_max_tokens,
        config,
    ).with_structured_output(ResearchPlan).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    )
    research_brief = state.get("research_brief") or ""
    try:
        planned = cast(
            ResearchPlan,
            await planner_model.ainvoke(
                [
                    HumanMessage(
                        content=multilingual_planner_prompt.format(
                            research_brief=research_brief,
                            date=get_today_str(),
                            output_language=config.get("configurable", {}).get(
                                "output_language", "en"
                            ),
                            run_id=context.run_id,
                        )
                    )
                ]
            ),
        )
    except Exception as error:
        logger.error(
            "Multilingual planner failed; no research plan was persisted",
            extra={
                "run_id": str(context.run_id),
                "error_type": type(error).__name__,
                "status_code": getattr(error, "status_code", None),
            },
            exc_info=redacted_exception_info(error),
        )
        message = (
            "Unable to create the multilingual research plan after bounded retries. "
            "No research was performed and no partial plan was saved; please retry."
        )
        return Command(
            goto="__end__",
            update={"messages": [AIMessage(content=message)], "final_report": message},
        )
    # The model must not choose an identity for a different run.
    plan = planned.model_copy(
        update={
            "run_id": context.run_id,
            "model_id": configurable.research_model,
            "prompt_version": "multilingual_planner_v1",
            "metadata": {
                **planned.metadata,
                "run_configuration": {
                    "selected_languages": [
                        {
                            "language": language.language,
                            "priority": language.priority,
                            "query_budget": language.query_budget,
                            "expected_source_types": language.expected_source_types,
                        }
                        for language in planned.ranked_languages
                    ],
                    "provider_routing": {
                        language.language: (
                            "bailian_web_search" if language.language.casefold().startswith("zh")
                            else "tavily"
                        )
                        for language in planned.ranked_languages
                    },
                },
            },
        }
    )
    await context.repository.append_research_plans(context.run_id, [plan])

    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    
    return Command(
        goto="provider_routed_discovery",
        update={
                "research_brief": research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(
                        content=(
                            f"Research brief:\n{research_brief}\n\n"
                            "Persisted multilingual research plan:\n"
                            f"{plan.model_dump_json()}"
                        )
                    )
                ]
            },
            "research_plan": plan,
        }
    )


async def language_gap_analysis(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["provider_routed_discovery", "report_composition"]]:
    """Use typed retrieval gaps to make one bounded language-expansion decision."""
    if state.get("language_gap_reviewed", False):
        return Command(goto="report_composition")

    context = RunContext.from_runnable_config(config)
    configurable = Configuration.from_runnable_config(config)
    plan = state.get("research_plan")
    if plan is None:
        plans = await context.repository.list_research_plans(context.run_id)
        if not plans:
            return Command(
                goto="report_composition", update={"language_gap_reviewed": True}
            )
        plan = plans[-1]
    elif not isinstance(plan, ResearchPlan):
        plan = ResearchPlan.model_validate(plan)

    _, sources, passages, claims, _, verification_results = await _load_evidence_ledger(
        config
    )
    evidence_ledger = json.dumps(
        {
            "sources": _serialize_artifacts(sources, SourceRecord),
            "passages": _serialize_artifacts(passages, EvidencePassage),
            "claims": _serialize_artifacts(claims, Claim),
            "verification_results": _serialize_artifacts(
                verification_results, VerificationResult
            ),
        },
        ensure_ascii=False,
    )
    gap_model = create_qwen_chat_model(
        configurable,
        configurable.research_model,
        configurable.research_model_max_tokens,
        config,
    ).with_structured_output(LanguageExpansionDecision).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    )
    decision = cast(
        LanguageExpansionDecision,
        await gap_model.ainvoke(
            [
                HumanMessage(
                    content=language_gap_analysis_prompt.format(
                        research_brief=state.get("research_brief", ""),
                        research_plan=plan.model_dump_json(),
                        evidence_ledger=evidence_ledger,
                    )
                )
            ]
        ),
    )

    existing_languages = {language.language for language in plan.ranked_languages}
    existing_priorities = [language.priority for language in plan.ranked_languages]
    additional_languages = decision.additional_languages
    existing_decision_languages = {
        language_decision.language for language_decision in plan.language_decisions
    }
    newly_skipped_languages = {
        language_decision.language for language_decision in decision.considered_but_skipped
    }
    if existing_decision_languages & newly_skipped_languages:
        raise ValueError("Gap analysis cannot reconsider an already decided language")
    if decision.should_add_languages:
        additional_names = {language.language for language in additional_languages}
        if (
            additional_names & existing_languages
            or len(additional_names) != len(additional_languages)
            or any(language.priority <= max(existing_priorities) for language in additional_languages)
        ):
            raise ValueError(
                "Language expansion must add unique languages with priorities after "
                "the initial research plan"
            )

    updated_plan = plan.model_copy(
        update={
            "id": uuid4(),
            "ranked_languages": [*plan.ranked_languages, *additional_languages],
            "query_variants": {
                **plan.query_variants,
                **decision.additional_query_variants,
            },
            "language_decisions": [
                *plan.language_decisions,
                *[
                    LanguageDecision(
                        language=language.language,
                        status="added_after_initial_retrieval",
                        rationale=language.selection_rationale,
                    )
                    for language in additional_languages
                ],
                *decision.considered_but_skipped,
            ],
            "post_retrieval_decision": decision,
            "metadata": {
                **plan.metadata,
                "plan_phase": "post_retrieval_language_gap_analysis",
                "run_configuration": {
                    "selected_languages": [
                        {
                            "language": language.language,
                            "priority": language.priority,
                            "query_budget": language.query_budget,
                            "expected_source_types": language.expected_source_types,
                        }
                        for language in [*plan.ranked_languages, *additional_languages]
                    ],
                    "provider_routing": {
                        language.language: (
                            "bailian_web_search" if language.language.casefold().startswith("zh")
                            else "tavily"
                        )
                        for language in [*plan.ranked_languages, *additional_languages]
                    },
                },
            },
        }
    )
    # Validate the merged, append-only plan before it becomes durable provenance.
    updated_plan = ResearchPlan.model_validate(updated_plan.model_dump())
    await context.repository.append_research_plans(context.run_id, [updated_plan])

    update = {
        "research_plan": updated_plan,
        "language_expansion_decision": decision,
        "language_gap_reviewed": True,
    }
    if not decision.should_add_languages:
        return Command(goto="report_composition", update=update)

    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations,
    )
    update["supervisor_messages"] = {
        "type": "override",
        "value": [
            SystemMessage(content=supervisor_system_prompt),
            HumanMessage(
                content=(
                    "Perform focused follow-up retrieval only for the languages "
                    "added after the evidence-gap review. Use the persisted plan's "
                    "queries, source types, and budgets.\n\n"
                    f"{updated_plan.model_dump_json()}"
                )
            ),
        ],
    }
    return Command(goto="provider_routed_discovery", update=update)


async def fetch_extract(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["evidence_ledger"]]:
    """Expose the provider fetch/extract result as a durable graph checkpoint.

    The provider-routed discovery tools fetch, ingest, and chunk source content before
    returning. This node reloads those immutable source and passage artifacts rather
    than passing tool transcript text to downstream reasoning.
    """
    _, sources, passages, _, _, _ = await _load_evidence_ledger(config)
    return Command(goto="evidence_ledger", update={"sources": sources, "passages": passages})


async def evidence_ledger(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["claim_extraction"]]:
    """Reload the run-scoped typed ledger at the evidence-to-claim boundary."""
    _, sources, passages, claims, _, verification_results = await _load_evidence_ledger(config)
    return Command(
        goto="claim_extraction",
        update={
            "sources": sources,
            "passages": passages,
            "claims": claims,
            "verification_results": verification_results,
        },
    )


async def claim_extraction(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["verification_conflict_loop"]]:
    """Validate that the extracted claim stage remains passage-linked.

    Research units perform Qwen extraction immediately after their retrieval phase;
    the top-level checkpoint prevents later stages from treating prose or messages as
    claims and records the extracted claim collection in shared graph state.
    """
    _, _, passages, claims, _, _ = await _load_evidence_ledger(config)
    passage_ids = {passage.id for passage in passages}
    invalid_claims = [
        claim.id for claim in claims
        if not claim.evidence_passage_ids or not set(claim.evidence_passage_ids).issubset(passage_ids)
    ]
    if invalid_claims:
        raise ValueError("Claim extraction produced claims without durable evidence passages")
    return Command(goto="verification_conflict_loop", update={"claims": claims})


async def verification_conflict_loop(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["provider_routed_discovery", "report_composition"]]:
    """Checkpoint verified artifacts, then decide whether evidence gaps merit retrieval.

    Claim-cluster verification and conflict resolution run inside each bounded research
    unit. The language-gap decision is the explicit loop edge: it can request another
    provider-routed retrieval pass or advance only verified artifacts to composition.
    """
    _, _, _, claims, _, verification_results = await _load_evidence_ledger(config)
    gap_command = await language_gap_analysis(state, config)
    return Command(
        goto=gap_command.goto,
        update={
            **gap_command.update,
            "claims": claims,
            "verification_results": verification_results,
        },
    )


async def report_qa(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["__end__"]]:
    """Recheck the persisted report bundle as the explicit terminal QA stage."""
    existing_issues = state.get("report_qa_issues", [])
    if any(issue.severity == "error" for issue in existing_issues):
        return Command(goto="__end__")

    context, sources, passages, claims, evidence_links, _ = await _load_evidence_ledger(config)
    statements = await context.repository.list_report_statements(context.run_id)
    queries = await context.repository.list_query_records(context.run_id)
    issues = validate_report_statements(
        statements=statements,
        claims=claims,
        passages=passages,
        sources=sources,
        queries=queries,
        evidence_links=evidence_links,
    )
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        return Command(
            goto="__end__",
            update={
                "final_report": "Report QA failed:\n" + "\n".join(
                    f"- {issue.message}" for issue in errors
                ),
                "report_qa_issues": issues,
            },
        )
    return Command(goto="__end__", update={"report_qa_issues": issues})




def build_graph():
    """Assemble the public PolyResearch workflow."""
    graph_builder = StateGraph(
        AgentState, input_schema=AgentInputState, context_schema=Configuration
    )
    graph_builder.add_node("initialize_research_run", initialize_research_run)
    graph_builder.add_node("clarify_with_user", clarify_with_user)
    graph_builder.add_node("write_research_brief", write_research_brief)
    graph_builder.add_node("multilingual_planner", multilingual_planner)
    graph_builder.add_node("provider_routed_discovery", supervisor_subgraph)
    graph_builder.add_node("fetch_extract", fetch_extract)
    graph_builder.add_node("evidence_ledger", evidence_ledger)
    graph_builder.add_node("claim_extraction", claim_extraction)
    graph_builder.add_node("verification_conflict_loop", verification_conflict_loop)
    graph_builder.add_node("report_composition", final_report_generation)
    graph_builder.add_node("report_qa", report_qa)
    graph_builder.add_edge(START, "initialize_research_run")
    graph_builder.add_edge("provider_routed_discovery", "fetch_extract")
    graph_builder.add_edge("report_composition", "report_qa")
    graph_builder.add_edge("report_qa", END)
    return graph_builder.compile()


graph = build_graph()
