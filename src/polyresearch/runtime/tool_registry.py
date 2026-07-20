"""Research-tool definitions and composition."""

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from polyresearch.retrieval.mcp_utils import load_bailian_web_search_tool
from polyresearch.models import ResearchComplete


@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Record a research-planning reflection for the active model loop."""
    return f"Reflection recorded: {reflection}"


async def get_all_tools(config: RunnableConfig):
    """Assemble the plan-constrained research toolset."""
    tools = [tool(ResearchComplete), think_tool]
    from polyresearch.retrieval.search_providers import planned_web_search

    tools.append(planned_web_search)
    tools.extend(await load_bailian_web_search_tool(config, {item.name for item in tools}))
    return tools
