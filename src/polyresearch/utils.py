"""Utility functions and helpers for the Deep Research agent."""

import asyncio
import hashlib
import json
import logging
import os
import re
import warnings
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Literal, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import aiohttp
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    HumanMessage,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import (
    BaseTool,
    InjectedToolArg,
    StructuredTool,
    ToolException,
    tool,
)
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.config import get_store
from mcp import McpError
from tavily import AsyncTavilyClient

from polyresearch.configuration import Configuration
from polyresearch.models import (
    EvidencePassage,
    ProvenanceAttachment,
    QueryRecord,
    ResearchComplete,
    SourceRecord,
    SourceVersion,
)
from polyresearch.repositories import RunContext

# --- Tavily Search Tool Utils ---

TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events."
)

_TRACKING_QUERY_PARAMETERS = {"fbclid", "gclid", "mc_cid", "mc_eid"}


def canonicalize_url(url: str) -> str:
    """Normalize a discovery URL while removing only known tracking parameters."""
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Unsupported discovery URL: {url}")
    hostname = (parsed.hostname or "").lower()
    netloc = hostname
    if parsed.port and not (
        (parsed.scheme.lower() == "http" and parsed.port == 80)
        or (parsed.scheme.lower() == "https" and parsed.port == 443)
    ):
        netloc = f"{hostname}:{parsed.port}"
    query = urlencode(
        sorted(
            (
                (key, value)
                for key, value in parse_qsl(parsed.query, keep_blank_values=True)
                if not key.lower().startswith("utm_")
                and key.lower() not in _TRACKING_QUERY_PARAMETERS
            ),
            key=lambda item: (item[0], item[1]),
        ),
        doseq=True,
    )
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", query, ""))


def _redirect_chain(result: dict[str, Any], discovered_url: str) -> list[str]:
    """Retain provider-supplied redirect metadata without treating it as evidence."""
    chain = result.get("redirect_chain") or result.get("redirects") or []
    if isinstance(chain, str):
        chain = [chain]
    return [str(url) for url in chain] or [discovered_url]


@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: list[str],
    max_results: Annotated[int, InjectedToolArg] = 5,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    query_language: Annotated[str, InjectedToolArg] = "en",
    locale: Annotated[str | None, InjectedToolArg] = None,
    start_date: Annotated[str | None, InjectedToolArg] = None,
    end_date: Annotated[str | None, InjectedToolArg] = None,
    target_source_type: Annotated[str | None, InjectedToolArg] = None,
    query_rationale: Annotated[str | None, InjectedToolArg] = None,
    fallback_from: Annotated[str | None, InjectedToolArg] = None,
    config: RunnableConfig = None
) -> str:
    """Fetch source records and exact passages from Tavily search API.

    Args:
        queries: List of search queries to execute
        max_results: Maximum number of results to return per query
        topic: Topic filter for search results (general, news, or finance)
        config: Runtime configuration for API keys and model settings

    Returns:
        Formatted string containing summarized search results
    """
    # Step 1: Execute search queries asynchronously
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
        start_date=start_date,
        end_date=end_date,
        config=config
    )
    
    # Step 2: Deduplicate results by URL to avoid processing the same content multiple times.
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            discovered_url = result['url']
            try:
                canonical_url = canonicalize_url(discovered_url)
            except ValueError:
                continue
            if canonical_url not in unique_results:
                unique_results[canonical_url] = {
                    **result,
                    "query": response['query'],
                    "discovered_url": discovered_url,
                    "canonical_url": canonical_url,
                    "redirect_chain": _redirect_chain(result, discovered_url),
                }
    
    # Step 3: Persist original tool output and typed source/passages immediately.
    # The returned payload is for the researcher's typed evidence ledger; raw tool
    # output remains an audit attachment rather than a reasoning input.
    sources: list[SourceRecord] = []
    source_versions: list[SourceVersion] = []
    passages: list[EvidencePassage] = []
    for result in unique_results.values():
        original_text = result.get("raw_content") or result.get("content")
        if not original_text:
            continue

        content_hash = hashlib.sha256(original_text.encode("utf-8")).hexdigest()
        source = SourceRecord(
            canonical_url=result["canonical_url"],
            title=result.get("title") or result["canonical_url"],
            content_hash=content_hash,
            research_unit_id=_research_unit_id_from_config(config),
            discovered_url=result["discovered_url"],
            redirect_chain=result["redirect_chain"],
        )
        source_version = SourceVersion(
            source_id=source.id,
            version_number=1,
            content_hash=content_hash,
            raw_content=original_text,
        )
        sources.append(source)
        source_versions.append(source_version)
        passages.extend(_chunk_evidence_passages(source, original_text))

    if not sources:
        return "No valid search results found. Please try different search queries or use a different search API."

    await _persist_tavily_ingestion(
        config=config,
        search_results=search_results,
        queries=queries,
        query_language=query_language,
        locale=locale,
        target_source_type=target_source_type,
        query_rationale=query_rationale,
        fallback_from=fallback_from,
        start_date=start_date,
        end_date=end_date,
        sources=sources,
        source_versions=source_versions,
        passages=passages,
    )

    return json.dumps(
        {
            "type": "polyresearch_evidence",
            "sources": [source.model_dump(mode="json") for source in sources],
            "passages": [passage.model_dump(mode="json") for passage in passages],
        },
        ensure_ascii=False,
    )


def _chunk_evidence_passages(
    source: SourceRecord, original_text: str
) -> list[EvidencePassage]:
    """Split fetched text into original-language paragraphs with stable locators."""
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", original_text)
        if paragraph.strip()
    ]
    if not paragraphs:
        return []
    return [
        EvidencePassage(
            source_id=source.id,
            text=paragraph,
            locator=f"paragraph-{index}",
            original_language=source.language,
        )
        for index, paragraph in enumerate(paragraphs, start=1)
    ]


async def _persist_tavily_ingestion(
    *,
    config: RunnableConfig | None,
    search_results: list[dict],
    queries: list[str],
    query_language: str,
    locale: str | None,
    target_source_type: str | None,
    query_rationale: str | None,
    fallback_from: str | None,
    start_date: str | None,
    end_date: str | None,
    sources: list[SourceRecord],
    source_versions: list[SourceVersion],
    passages: list[EvidencePassage],
) -> None:
    """Write all Tavily discovery artifacts before returning to the researcher."""
    if not config:
        return
    try:
        context = RunContext.from_runnable_config(config)
    except ValueError:
        # Direct tool use remains available for isolated unit tests; graph execution
        # always supplies a durable context from the CLI.
        return

    raw_output = json.dumps(search_results, ensure_ascii=False, sort_keys=True, default=str)
    query_records = []
    for response in search_results:
        for result_rank, result in enumerate(response.get("results", []), start=1):
            discovered_url = result.get("url")
            if not discovered_url:
                continue
            try:
                canonical_url = canonicalize_url(discovered_url)
            except ValueError:
                continue
            query_records.append(
                QueryRecord(
                    run_id=context.run_id,
                    research_unit_id=context.research_unit_id,
                    query=response.get("query") or queries[0],
                    language=query_language,
                    provider="tavily",
                    locale=locale,
                    target_source_type=target_source_type,
                    rationale=query_rationale,
                    date_from=start_date,
                    date_to=end_date,
                    fallback_from=fallback_from,
                    result_rank=result_rank,
                    result_url=canonical_url,
                )
            )
    await context.repository.append_query_records(context.run_id, query_records)
    await context.repository.append_provenance_attachments(
        context.run_id,
        [
            ProvenanceAttachment(
                run_id=context.run_id,
                provider="tavily",
                tool_name="tavily_search",
                raw_output=raw_output,
            )
        ],
    )
    await context.repository.append_sources(context.run_id, sources)
    await context.repository.append_source_versions(context.run_id, source_versions)
    await context.repository.append_passages(context.run_id, passages)


def _research_unit_id_from_config(config: RunnableConfig | None):
    """Read the optional unit scope used to isolate parallel researchers."""
    if not config:
        return None
    try:
        return RunContext.from_runnable_config(config).research_unit_id
    except ValueError:
        return None

async def tavily_search_async(
    search_queries, 
    max_results: int = 5, 
    topic: Literal["general", "news", "finance"] = "general", 
    include_raw_content: bool = True, 
    start_date: str | None = None,
    end_date: str | None = None,
    config: RunnableConfig = None
):
    """Execute multiple Tavily search queries asynchronously.
    
    Args:
        search_queries: List of search query strings to execute
        max_results: Maximum number of results per query
        topic: Topic category for filtering results
        include_raw_content: Whether to include full webpage content
        config: Runtime configuration for API key access
        
    Returns:
        List of search result dictionaries from Tavily API
    """
    # Initialize the Tavily client with API key from config
    tavily_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))
    
    # Create search tasks for parallel execution
    search_tasks = []
    for query in search_queries:
        search_args = {
            "max_results": max_results,
            "include_raw_content": include_raw_content,
            "topic": topic,
        }
        if start_date is not None:
            search_args["start_date"] = start_date
        if end_date is not None:
            search_args["end_date"] = end_date
        search_tasks.append(tavily_client.search(query, **search_args))
    
    # Execute all search queries in parallel and return results
    search_results = await asyncio.gather(*search_tasks)
    return search_results

# --- Reflection Tool Utils ---

@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"


# --- MCP Utils ---

async def get_mcp_access_token(
    supabase_token: str,
    base_mcp_url: str,
) -> Optional[dict[str, Any]]:
    """Exchange Supabase token for MCP access token using OAuth token exchange.
    
    Args:
        supabase_token: Valid Supabase authentication token
        base_mcp_url: Base URL of the MCP server
        
    Returns:
        Token data dictionary if successful, None if failed
    """
    try:
        # Prepare OAuth token exchange request data
        form_data = {
            "client_id": "mcp_default",
            "subject_token": supabase_token,
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "resource": base_mcp_url.rstrip("/") + "/mcp",
            "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }
        
        # Execute token exchange request
        async with aiohttp.ClientSession() as session:
            token_url = base_mcp_url.rstrip("/") + "/oauth/token"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            
            async with session.post(token_url, headers=headers, data=form_data) as response:
                if response.status == 200:
                    # Successfully obtained token
                    token_data = await response.json()
                    return token_data
                else:
                    # Log error details for debugging
                    response_text = await response.text()
                    logging.error(f"Token exchange failed: {response_text}")
                    
    except Exception as e:
        logging.error(f"Error during token exchange: {e}")
    
    return None

async def get_tokens(config: RunnableConfig):
    """Retrieve stored authentication tokens with expiration validation.
    
    Args:
        config: Runtime configuration containing thread and user identifiers
        
    Returns:
        Token dictionary if valid and not expired, None otherwise
    """
    store = get_store()
    
    # Extract required identifiers from config
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return None
        
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return None
    
    # Retrieve stored tokens
    tokens = await store.aget((user_id, "tokens"), "data")
    if not tokens:
        return None
    
    # Check token expiration
    expires_in = tokens.value.get("expires_in")  # seconds until expiration
    created_at = tokens.created_at  # datetime of token creation
    current_time = datetime.now(timezone.utc)
    expiration_time = created_at + timedelta(seconds=expires_in)
    
    if current_time > expiration_time:
        # Token expired, clean up and return None
        await store.adelete((user_id, "tokens"), "data")
        return None

    return tokens.value

async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    """Store authentication tokens in the configuration store.
    
    Args:
        config: Runtime configuration containing thread and user identifiers
        tokens: Token dictionary to store
    """
    store = get_store()
    
    # Extract required identifiers from config
    thread_id = config.get("configurable", {}).get("thread_id")
    if not thread_id:
        return
        
    user_id = config.get("metadata", {}).get("owner")
    if not user_id:
        return
    
    # Store the tokens
    await store.aput((user_id, "tokens"), "data", tokens)

async def fetch_tokens(config: RunnableConfig) -> dict[str, Any]:
    """Fetch and refresh MCP tokens, obtaining new ones if needed.
    
    Args:
        config: Runtime configuration with authentication details
        
    Returns:
        Valid token dictionary, or None if unable to obtain tokens
    """
    # Try to get existing valid tokens first
    current_tokens = await get_tokens(config)
    if current_tokens:
        return current_tokens
    
    # Extract Supabase token for new token exchange
    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    if not supabase_token:
        return None
    
    # Extract MCP configuration
    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not mcp_config or not mcp_config.get("url"):
        return None
    
    # Exchange Supabase token for MCP tokens
    mcp_tokens = await get_mcp_access_token(supabase_token, mcp_config.get("url"))
    if not mcp_tokens:
        return None

    # Store the new tokens and return them
    await set_tokens(config, mcp_tokens)
    return mcp_tokens

def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    """Wrap MCP tool with comprehensive authentication and error handling.
    
    Args:
        tool: The MCP structured tool to wrap
        
    Returns:
        Enhanced tool with authentication error handling
    """
    original_coroutine = tool.coroutine
    
    async def authentication_wrapper(**kwargs):
        """Enhanced coroutine with MCP error handling and user-friendly messages."""
        
        def _find_mcp_error_in_exception_chain(exc: BaseException) -> McpError | None:
            """Recursively search for MCP errors in exception chains."""
            if isinstance(exc, McpError):
                return exc
            
            # Handle ExceptionGroup (Python 3.11+) by checking attributes
            if hasattr(exc, 'exceptions'):
                for sub_exception in exc.exceptions:
                    if found_error := _find_mcp_error_in_exception_chain(sub_exception):
                        return found_error
            return None
        
        try:
            # Execute the original tool functionality
            return await original_coroutine(**kwargs)
            
        except BaseException as original_error:
            # Search for MCP-specific errors in the exception chain
            mcp_error = _find_mcp_error_in_exception_chain(original_error)
            if not mcp_error:
                # Not an MCP error, re-raise the original exception
                raise original_error
            
            # Handle MCP-specific error cases
            error_details = mcp_error.error
            error_code = getattr(error_details, "code", None)
            error_data = getattr(error_details, "data", None) or {}
            
            # Check for authentication/interaction required error
            if error_code == -32003:  # Interaction required error code
                message_payload = error_data.get("message", {})
                error_message = "Required interaction"
                
                # Extract user-friendly message if available
                if isinstance(message_payload, dict):
                    error_message = message_payload.get("text") or error_message
                
                # Append URL if provided for user reference
                if url := error_data.get("url"):
                    error_message = f"{error_message} {url}"
                
                raise ToolException(error_message) from original_error
            
            # For other MCP errors, re-raise the original
            raise original_error
    
    # Replace the tool's coroutine with our enhanced version
    tool.coroutine = authentication_wrapper
    return tool

async def load_mcp_tools(
    config: RunnableConfig,
    existing_tool_names: set[str],
) -> list[BaseTool]:
    """Load and configure MCP (Model Context Protocol) tools with authentication.
    
    Args:
        config: Runtime configuration containing MCP server details
        existing_tool_names: Set of tool names already in use to avoid conflicts
        
    Returns:
        List of configured MCP tools ready for use
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Step 1: Handle authentication if required
    if configurable.mcp_config and configurable.mcp_config.auth_required:
        mcp_tokens = await fetch_tokens(config)
    else:
        mcp_tokens = None
    
    # Step 2: Validate configuration requirements
    config_valid = (
        configurable.mcp_config and 
        configurable.mcp_config.url and 
        configurable.mcp_config.tools and 
        (mcp_tokens or not configurable.mcp_config.auth_required)
    )
    
    if not config_valid:
        return []
    
    # Step 3: Set up MCP server connection
    server_url = configurable.mcp_config.url.rstrip("/") + "/mcp"
    
    # Configure authentication headers if tokens are available
    auth_headers = None
    if mcp_tokens:
        auth_headers = {"Authorization": f"Bearer {mcp_tokens['access_token']}"}
    
    mcp_server_config = {
        "server_1": {
            "url": server_url,
            "headers": auth_headers,
            "transport": "streamable_http"
        }
    }
    # TODO: When Multi-MCP Server support is merged in OAP, update this code
    
    # Step 4: Load tools from MCP server
    try:
        client = MultiServerMCPClient(mcp_server_config)
        available_mcp_tools = await client.get_tools()
    except Exception:
        # If MCP server connection fails, return empty list
        return []
    
    # Step 5: Filter and configure tools
    configured_tools = []
    for mcp_tool in available_mcp_tools:
        # Skip tools with conflicting names
        if mcp_tool.name in existing_tool_names:
            warnings.warn(
                f"MCP tool '{mcp_tool.name}' conflicts with existing tool name - skipping"
            )
            continue
        
        # Only include tools specified in configuration
        if mcp_tool.name not in set(configurable.mcp_config.tools):
            continue
        
        # Wrap tool with authentication handling and add to list
        enhanced_tool = wrap_mcp_authenticate_tool(mcp_tool)
        configured_tools.append(enhanced_tool)
    
    return configured_tools


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
    except Exception:
        return []

    # Never expose an arbitrary server tool, even if the server advertises it.
    return [
        mcp_tool
        for mcp_tool in available_tools
        if mcp_tool.name == bailian.tool_name
    ]


# --- Tool Utils ---

async def get_all_tools(config: RunnableConfig):
    """Assemble complete toolkit including research, search, and MCP tools.
    
    Args:
        config: Runtime configuration specifying search API and MCP settings
        
    Returns:
        List of all configured and available tools for research operations
    """
    # Start with core research tools
    tools = [tool(ResearchComplete), think_tool]
    
    # Route discovery through the persisted multilingual research plan.
    from polyresearch.search_providers import planned_web_search

    tools.append(planned_web_search)
    
    # Track existing tool names to prevent conflicts
    existing_tool_names = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search") 
        for tool in tools
    }
    
    # Bailian Web Search is the sole MCP integration exposed in this phase.
    tools.extend(await load_bailian_web_search_tool(config, existing_tool_names))
    
    return tools

# --- Token Limit Exceeded Utils ---

def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    """Determine if an exception indicates a token/context limit was exceeded.
    
    Args:
        exception: The exception to analyze
        model_name: Optional model name to optimize provider detection
        
    Returns:
        True if the exception indicates a token limit was exceeded, False otherwise
    """
    error_str = str(exception).lower()
    return _check_qwen_token_limit(exception, error_str)

def _check_qwen_token_limit(exception: Exception, error_str: str) -> bool:
    """Check if exception indicates Qwen token limit exceeded."""
    # Check for Qwen-specific token limit patterns
    token_indicators = [
        'token', 'context', 'length', 'maximum context', 
        'reduce', 'exceed', 'limit', 'too long'
    ]
    
    # Check error message for token-related keywords
    if any(indicator in error_str.lower() for indicator in token_indicators):
        return True
    
    # Check exception attributes for Qwen-specific patterns
    if hasattr(exception, 'code'):
        error_code = getattr(exception, 'code', '').lower()
        if 'context_length' in error_code or 'token' in error_code:
            return True
    
    if hasattr(exception, 'type'):
        error_type = getattr(exception, 'type', '').lower()
        if 'token' in error_type or 'context' in error_type:
            return True
    
    return False


MODEL_TOKEN_LIMITS = {
    "qwen3.7-max": 100000,
    "qwen3.7-plus": 100000,
    "qwen3.6-plus": 100000
}

def get_model_token_limit(model_string) -> int:
    """Look up the token limit for a specific model.
    
    Args:
        model_string: The model identifier string to look up
        
    Returns:
        Token limit as integer if found, None if model not in lookup table
    """
    # Search through known model token limits
    for model_key, token_limit in MODEL_TOKEN_LIMITS.items():
        if model_key in model_string:
            return token_limit
    
    # Model not found in lookup table
    return None

# --- Misc Utils ---

def get_today_str() -> str:
    """Get current date formatted for display in prompts and outputs.
    
    Returns:
        Human-readable date string in format like 'Mon Jan 15, 2024'
    """
    now = datetime.now()
    return f"{now:%a} {now:%b} {now.day}, {now:%Y}"

def get_config_value(value):
    """Extract value from configuration, handling enums and None values."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    elif isinstance(value, dict):
        return value
    else:
        return value.value

def get_qwen_api_key(config: RunnableConfig) -> Optional[str]:
    """Get the Model Studio API key from runtime configuration or the environment."""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        return api_keys.get("QWEN_API_KEY") if api_keys else None
    return os.getenv("QWEN_API_KEY")


def create_qwen_chat_model(
    configuration: Configuration,
    model: str,
    max_tokens: int,
    config: RunnableConfig,
) -> BaseChatModel:
    """Create a Qwen model using the configured DashScope-compatible transport."""
    api_key = get_qwen_api_key(config)
    if not api_key:
        raise ValueError(
            "Qwen credentials are missing. Set QWEN_API_KEY or provide it through "
            "configurable.apiKeys when GET_API_KEYS_FROM_CONFIG=true."
        )
    return init_chat_model(
        **configuration.chat_model_config(
            model=model,
            max_tokens=max_tokens,
            api_key=api_key,
        )
    )

def get_tavily_api_key(config: RunnableConfig):
    """Get Tavily API key from environment or config."""
    should_get_from_config = os.getenv("GET_API_KEYS_FROM_CONFIG", "false")
    if should_get_from_config.lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        return api_keys.get("TAVILY_API_KEY") if api_keys else None
    return os.getenv("TAVILY_API_KEY")
