import asyncio
import json
from typing import Literal, cast
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from polyresearch.configuration import (
    Configuration,
)
from polyresearch.models import (
    AgentInputState,
    AgentState,
    Claim,
    ClaimExtractionResult,
    ClarifyWithUser,
    ConductResearch,
    EvidenceLink,
    EvidencePassage,
    LanguageDecision,
    LanguageExpansionDecision,
    ProvenanceAttachment,
    ReportBundle,
    ReportDraft,
    ReportQaIssue,
    ReportStatement,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchPlan,
    ResearchQuestion,
    ResearchRun,
    SourceRecord,
    SupervisorState,
    VerificationResult,
    VerificationStatus,
)
from polyresearch.prompts import (
    clarify_with_user_instructions,
    final_report_generation_prompt,
    language_gap_analysis_prompt,
    lead_researcher_prompt,
    multilingual_planner_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from polyresearch.report_qa import validate_report_statements
from polyresearch.repositories import RunContext
from polyresearch.utils import (
    create_qwen_chat_model,
    get_all_tools,
    get_model_token_limit,
    get_today_str,
    is_token_limit_exceeded,
    think_tool,
)


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


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.
    
    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.
    
    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    # Step 1: Configure the supervisor model with available tools
    configurable = Configuration.from_runnable_config(config)
    research_model = create_qwen_chat_model(
        configurable,
        configurable.research_model,
        configurable.research_model_max_tokens,
        config,
    )
    
    # Available tools: research delegation, completion signaling, and strategic thinking
    lead_researcher_tools = [tool(ConductResearch), tool(ResearchComplete), think_tool]
    
    # Configure model with tools, retry logic, and model settings
    research_model = (
        research_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )
    
    # Step 2: Generate supervisor response based on current context
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    
    # Step 3: Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.
    
    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ConductResearch - Delegates research tasks to sub-researchers
    3. ResearchComplete - Signals completion of research phase
    
    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings
        
    Returns:
        Command to either continue supervision loop or end research phase
    """
    # Step 1: Extract current state and check exit conditions
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = cast(AIMessage, supervisor_messages[-1])
    
    # Define exit criteria for research phase
    exceeded_allowed_iterations = research_iterations >= configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    # Exit if any termination condition is met
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto="__end__",
            update={"research_brief": state.get("research_brief", "")},
        )
    
    # Step 2: Process all tool calls together (both think_tool and ConductResearch)
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    
    # Handle think_tool calls (strategic reflection)
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]
    
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))
    
    # Handle ConductResearch calls (research delegation)
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ConductResearch"
    ]
    
    if conduct_research_calls:
        try:
            # Limit concurrent research units to prevent resource exhaustion
            allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]
            
            # Execute research tasks in parallel
            research_units = [
                (tool_call, uuid4()) for tool_call in allowed_conduct_research_calls
            ]
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"],
                    "research_unit_id": research_unit_id,
                }, {
                    **config,
                    "configurable": {
                        **config.get("configurable", {}),
                        "research_unit_id": str(research_unit_id),
                    },
                })
                for tool_call, research_unit_id in research_units
            ]
            
            tool_results = await asyncio.gather(*research_tasks, return_exceptions=True)
            
            # Create tool messages with research results
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                if isinstance(observation, Exception):
                    all_tool_messages.append(ToolMessage(
                        content=f"Research task failed: {observation}",
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    ))
                    continue
                all_tool_messages.append(ToolMessage(
                    content=_researcher_evidence_summary(observation),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))
            
            # Handle overflow research calls with error messages
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))
            
            # Pass typed evidence records through the supervisor.
            for field in ("sources", "passages", "claims", "verification_results"):
                artifacts = [
                    artifact
                    for observation in tool_results
                    if not isinstance(observation, Exception)
                    for artifact in observation.get(field, [])
                ]
                if artifacts:
                    update_payload[field] = artifacts
                
        except Exception as e:
            raise RuntimeError("Failed to orchestrate parallel research tasks") from e
    
    # Step 3: Return command with all tool results
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    ) 

# Supervisor Subgraph Construction
# Creates the supervisor workflow that manages research delegation and coordination
supervisor_builder = StateGraph(SupervisorState, context_schema=Configuration)

# Add supervisor nodes for research management
supervisor_builder.add_node("supervisor", supervisor)           # Main supervisor logic
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # Tool execution handler

# Define supervisor workflow edges
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# Compile supervisor subgraph for use in main workflow
supervisor_subgraph = supervisor_builder.compile()

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
    
    # Prepare system prompt with MCP context if available
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", 
        date=get_today_str()
    )
    
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


async def _persist_non_tavily_tool_outputs(
    config: RunnableConfig, tool_calls: list[dict], observations: list[object]
) -> None:
    """Retain raw non-Tavily tool outputs as audit-only provenance attachments."""
    try:
        context = RunContext.from_runnable_config(config)
    except ValueError:
        return

    attachments = [
        ProvenanceAttachment(
            run_id=context.run_id,
            provider="runtime_tool",
            tool_name=tool_call["name"],
            raw_output=str(observation),
        )
        for tool_call, observation in zip(tool_calls, observations)
        if tool_call["name"] != "tavily_search"
    ]
    if attachments:
        await context.repository.append_provenance_attachments(
            context.run_id, attachments
        )


async def _load_evidence_ledger(config: RunnableConfig):
    """Load typed evidence artifacts from the durable run ledger."""
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
        source
        for source in sources
        if source.research_unit_id == context.research_unit_id
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


def _researcher_evidence_summary(observation: dict) -> str:
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


def _serialize_artifacts(artifacts, model_type):
    """Serialize state artifacts whether LangGraph returns models or dictionaries."""
    return [
        (
            artifact.model_dump(mode="json")
            if isinstance(artifact, model_type)
            else model_type.model_validate(artifact).model_dump(mode="json")
        )
        for artifact in artifacts
    ]


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
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config) 
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
    if not passages:
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
        "evidence. Each claim must cite one or more exact passage IDs from the "
        "typed evidence ledger. Do not invent sources, passage IDs, or verification "
        "results. Raw tool output is an audit attachment, not evidence for reasoning."
    )
    evidence_ledger = json.dumps(
        {
            "sources": _serialize_artifacts(sources, SourceRecord),
            "passages": _serialize_artifacts(passages, EvidencePassage),
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

    known_passage_ids = {passage.id for passage in passages}
    claims = [
        claim
        for claim in response.claims
        if set(claim.evidence_passage_ids).issubset(known_passage_ids)
    ]
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

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")           # Entry point to researcher
researcher_builder.add_edge("extract_claims", END)          # Exit point after claim extraction

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()


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


async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.
    
    This function takes all collected research findings and synthesizes them into a 
    well-structured, comprehensive final report using the configured report generation model.
    
    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys
        
    Returns:
        Dictionary containing the final report and cleared state
    """
    # Step 1: Load report inputs from the durable typed evidence ledger.
    context, sources, passages, claims, _, verification_results = await _load_evidence_ledger(
        config
    )
    findings = json.dumps(
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
    
    # Step 2: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    writer_model = create_qwen_chat_model(
        configurable,
        configurable.final_report_model,
        configurable.final_report_model_max_tokens,
        config,
    ).with_structured_output(ReportDraft).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    )
    
    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    
    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt with all research context
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )
            
            # Generate the final report
            report_draft = cast(ReportDraft, await writer_model.ainvoke([
                HumanMessage(content=final_report_prompt)
            ]))
            statements = _build_report_statements(
                run_id=context.run_id,
                report_draft=report_draft,
                claims=claims,
                verification_results=verification_results,
            )
            qa_issues = validate_report_statements(
                statements=statements,
                claims=claims,
                passages=passages,
                sources=sources,
            )
            if not statements:
                qa_issues.append(
                    ReportQaIssue(
                        code="no_report_statements",
                        severity="error",
                        message="Report draft produced no resolvable factual statements.",
                    )
                )
            qa_errors = [issue for issue in qa_issues if issue.severity == "error"]
            if qa_errors:
                return {
                    "final_report": "Report QA failed:\n"
                    + "\n".join(f"- {issue.message}" for issue in qa_errors),
                    "messages": [AIMessage(content="Report QA failed")],
                }
            markdown = _render_statement_markdown(
                title=report_draft.title,
                statements=statements,
                passages=passages,
                sources=sources,
                qa_issues=qa_issues,
            )
            await context.repository.append_report_statements(context.run_id, statements)
            bundle = ReportBundle(
                run_id=context.run_id,
                markdown=markdown,
                provenance_json={
                    "statement_ids": [str(statement.id) for statement in statements],
                    "claim_ids": [
                        str(claim_id)
                        for statement in statements
                        for claim_id in statement.claim_ids
                    ],
                },
                qa_issues=qa_issues,
                qa_passed=True,
            )
            await context.repository.append_report_bundles(context.run_id, [bundle])
            
            # Return successful report generation
            return {
                "final_report": markdown,
                "messages": [AIMessage(content=markdown)],
            }
            
        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                        }
                    # Use 4x token limit as character approximation for truncation
                    findings_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # Truncate findings and retry
                findings = findings[:findings_token_limit]
                continue
            else:
                # Non-token-limit error: return error immediately
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                }
    
    # Step 4: Return failure result if all retries exhausted
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
    }


def _build_report_statements(
    *,
    run_id,
    report_draft: ReportDraft,
    claims: list[Claim],
    verification_results: list[VerificationResult],
) -> list[ReportStatement]:
    """Resolve model-selected claim IDs into durable, cited report statements."""
    claims_by_id = {claim.id: claim for claim in claims}
    statuses_by_claim_id = {
        result.claim_id: result.status for result in verification_results
    }
    statements: list[ReportStatement] = []
    for draft in report_draft.statements:
        citation_ids = list(
            dict.fromkeys(
                passage_id
                for claim_id in draft.claim_ids
                if claim_id in claims_by_id
                for passage_id in claims_by_id[claim_id].evidence_passage_ids
            )
        )
        statuses = [statuses_by_claim_id.get(claim_id) for claim_id in draft.claim_ids]
        status = (
            statuses[0]
            if statuses and all(candidate == statuses[0] for candidate in statuses)
            and statuses[0] is not None
            else VerificationStatus.INSUFFICIENT_EVIDENCE
        )
        statements.append(
            ReportStatement(
                run_id=run_id,
                rendered_text=draft.rendered_text,
                claim_ids=draft.claim_ids,
                citation_ids=citation_ids,
                verification_status=status,
            )
        )
    return statements


def _render_statement_markdown(
    *,
    title: str,
    statements: list[ReportStatement],
    passages: list[EvidencePassage],
    sources: list[SourceRecord],
    qa_issues,
) -> str:
    """Render persisted statements and stable passage citations as Markdown."""
    passages_by_id = {passage.id: passage for passage in passages}
    sources_by_id = {source.id: source for source in sources}
    rendered_statements = []
    used_citation_ids = []
    for statement in statements:
        citations = " ".join(f"[P:{citation_id}]" for citation_id in statement.citation_ids)
        rendered_statements.append(f"{statement.rendered_text} {citations}".rstrip())
        used_citation_ids.extend(statement.citation_ids)

    lines = [f"# {title}", "", *rendered_statements]
    if used_citation_ids:
        lines.extend(["", "## Sources", ""])
        for citation_id in dict.fromkeys(used_citation_ids):
            passage = passages_by_id.get(citation_id)
            if not passage:
                continue
            source = sources_by_id.get(passage.source_id)
            source_label = source.title if source else "Unknown source"
            source_url = source.canonical_url if source else ""
            lines.append(
                f"- [P:{citation_id}] {source_label} — {source_url} ({passage.locator})"
            )
    warnings = [issue for issue in qa_issues if issue.severity == "warning"]
    if warnings:
        lines.extend(["", "## QA warnings", ""])
        lines.extend(f"- {warning.message}" for warning in warnings)
    return "\n".join(lines)

# Main Deep Researcher Graph Construction
# Creates the complete deep research workflow from user input to final report
graph_builder = StateGraph(
    AgentState, 
    input_schema=AgentInputState, 
    context_schema=Configuration
)

# Add main workflow nodes for the complete research process
graph_builder.add_node("initialize_research_run", initialize_research_run)
graph_builder.add_node("clarify_with_user", clarify_with_user)           # User clarification phase
graph_builder.add_node("write_research_brief", write_research_brief)     # Research planning phase
graph_builder.add_node("multilingual_planner", multilingual_planner)      # Adaptive language planning
graph_builder.add_node("research_supervisor", supervisor_subgraph)       # Research execution phase
graph_builder.add_node("language_gap_analysis", language_gap_analysis)   # One bounded expansion review
graph_builder.add_node("final_report_generation", final_report_generation)  # Report generation phase

# Define main workflow edges for sequential execution
graph_builder.add_edge(START, "initialize_research_run")                 # Durable run setup
graph_builder.add_edge("research_supervisor", "language_gap_analysis")   # Initial retrieval to gap review
graph_builder.add_edge("final_report_generation", END)                   # Final exit point

# Compile the complete research workflow
graph = graph_builder.compile()
