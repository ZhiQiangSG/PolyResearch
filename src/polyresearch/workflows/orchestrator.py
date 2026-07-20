"""Top-level PolyResearch workflow nodes and graph assembly."""

import json
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
from polyresearch.workflows.report_generator import final_report_generation
from polyresearch.workflows.supervisor import supervisor_subgraph

async def initialize_research_run(
    state: AgentState, config: RunnableConfig
) -> Command[Literal["clarify_with_user"]]:
    """Create the durable run record before any model or tool work begins."""
    context = RunContext.from_runnable_config(config)
    configurable = config.get("configurable", {})
    question = get_buffer_string(state.get("messages", [])).strip()
    run = ResearchRun(
        id=context.run_id,
        question=question,
        output_language=configurable.get("output_language", "en"),
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
) -> Command[Literal["research_supervisor"]]:
    """Create and persist Qwen's adaptive, structured multilingual research plan."""
    configurable = Configuration.from_runnable_config(config)
    context = RunContext.from_runnable_config(config)
    planner_model = create_qwen_chat_model(
        configurable,
        configurable.research_model,
        configurable.research_model_max_tokens,
        config,
    ).with_structured_output(ResearchPlan).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    )
    research_brief = state.get("research_brief") or ""
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
    # The model must not choose an identity for a different run.
    plan = planned.model_copy(
        update={
            "run_id": context.run_id,
            "model_id": configurable.research_model,
            "prompt_version": "multilingual_planner_v1",
        }
    )
    await context.repository.append_research_plans(context.run_id, [plan])

    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    
    return Command(
        goto="research_supervisor", 
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
) -> Command[Literal["research_supervisor", "final_report_generation"]]:
    """Use typed retrieval gaps to make one bounded language-expansion decision."""
    if state.get("language_gap_reviewed", False):
        return Command(goto="final_report_generation")

    context = RunContext.from_runnable_config(config)
    configurable = Configuration.from_runnable_config(config)
    plan = state.get("research_plan")
    if plan is None:
        plans = await context.repository.list_research_plans(context.run_id)
        if not plans:
            return Command(
                goto="final_report_generation", update={"language_gap_reviewed": True}
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
        return Command(goto="final_report_generation", update=update)

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
    return Command(goto="research_supervisor", update=update)




def build_graph():
    """Assemble the public PolyResearch workflow."""
    graph_builder = StateGraph(
        AgentState, input_schema=AgentInputState, context_schema=Configuration
    )
    graph_builder.add_node("initialize_research_run", initialize_research_run)
    graph_builder.add_node("clarify_with_user", clarify_with_user)
    graph_builder.add_node("write_research_brief", write_research_brief)
    graph_builder.add_node("multilingual_planner", multilingual_planner)
    graph_builder.add_node("research_supervisor", supervisor_subgraph)
    graph_builder.add_node("language_gap_analysis", language_gap_analysis)
    graph_builder.add_node("final_report_generation", final_report_generation)
    graph_builder.add_edge(START, "initialize_research_run")
    graph_builder.add_edge("research_supervisor", "language_gap_analysis")
    graph_builder.add_edge("final_report_generation", END)
    return graph_builder.compile()


graph = build_graph()
