"""Supervisor subgraph for delegating research units."""

import asyncio
from typing import Literal, cast
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.types import Command

from polyresearch.configuration import Configuration
from polyresearch.models import ConductResearch, EvidenceTask, ResearchComplete, SupervisorState
from polyresearch.nodes.provenance import researcher_evidence_summary as _researcher_evidence_summary
from polyresearch.workflows.researcher import researcher_subgraph
from polyresearch.runtime.model_utils import create_qwen_chat_model
from polyresearch.runtime.tool_registry import think_tool

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
            research_units = []
            for tool_call in allowed_conduct_research_calls:
                task = EvidenceTask.model_validate(tool_call["args"]["task"])
                _validate_evidence_task(task, state.get("research_plan"))
                research_units.append((tool_call, task, uuid4()))
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=task.model_dump_json())
                    ],
                    "research_topic": task.subquestion,
                    "evidence_task": task,
                    "research_unit_id": research_unit_id,
                }, {
                    **config,
                    "configurable": {
                        **config.get("configurable", {}),
                        "research_unit_id": str(research_unit_id),
                        "research_plan": state.get("research_plan"),
                    },
                })
                for tool_call, task, research_unit_id in research_units
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


def _validate_evidence_task(task: EvidenceTask, plan) -> None:
    """Keep delegated work inside the selected language/source-type evidence plan."""
    if plan is None:
        return
    planned_language = next(
        (item for item in plan.ranked_languages if item.language == task.language), None
    )
    if planned_language is None:
        raise ValueError("Evidence task language is not selected by the research plan")
    if task.target_source_type not in planned_language.expected_source_types:
        raise ValueError("Evidence task source type is not selected for its language")



supervisor_builder = StateGraph(SupervisorState, context_schema=Configuration)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_subgraph = supervisor_builder.compile()
