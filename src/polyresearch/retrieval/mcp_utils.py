"""Allowlisted Bailian MCP tool loading."""

import asyncio
import logging
import os

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from polyresearch.configuration import Configuration
from polyresearch.security import redacted_exception_info

logger = logging.getLogger(__name__)


async def load_bailian_web_search_tool(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    """Load only the explicitly allowlisted Bailian Web Search MCP tool.

    Generic MCP configuration is deliberately not consulted here: Milestone 3
    permits Bailian Web Search only, and only for Chinese-source discovery.
    """
    configurable = Configuration.from_runnable_config(config)
    bailian = configurable.bailian_web_search
    if bailian is None or bailian.tool_name in existing_tool_names:
        return []

    api_key = bailian.authentication.api_key or os.getenv(
        bailian.authentication.api_key_env_var
    )
    if not api_key:
        return []
    mcp_server_config = {
        "bailian_web_search": {
            "url": bailian.server_url,
            "headers": {"Authorization": f"Bearer {api_key}"},
            "transport": "streamable_http",
        }
    }
    try:
        client = MultiServerMCPClient(mcp_server_config)
        available_tools = await asyncio.wait_for(
            client.get_tools(), timeout=bailian.timeout_seconds
        )
    except Exception as error:
        logger.warning(
            "Bailian MCP tool loading failed",
            extra={
                "operation": "load_mcp_tools",
                "provider": "bailian_web_search",
                "endpoint": bailian.server_url,
            },
            exc_info=redacted_exception_info(error),
        )
        return []

    # Never expose an arbitrary server tool, even if the server advertises it.
    return [
        mcp_tool
        for mcp_tool in available_tools
        if mcp_tool.name == bailian.tool_name
    ]
