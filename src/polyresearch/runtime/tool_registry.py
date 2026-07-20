"""Research-tool definitions and composition."""

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from polyresearch.models import ResearchComplete


@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Record a research-planning reflection for the active model loop."""
    return f"Reflection recorded: {reflection}"


async def get_all_tools(config: RunnableConfig):
    """Expose only routed research tools; MCP adapters remain provider-internal."""
    tools = [tool(ResearchComplete), think_tool]
    from polyresearch.retrieval.search_providers import planned_web_search

    tools.append(planned_web_search)
    return tools
