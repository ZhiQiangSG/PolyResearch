"""Provider-routed discovery constrained by the persisted multilingual plan."""

from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import ToolException, tool

from polyresearch.configuration import Configuration
from polyresearch.models import ResearchPlan


@dataclass(frozen=True)
class SearchRequest:
    """One language- and source-type-specific discovery request."""

    query: str
    language: str
    target_source_type: str


class TavilySearchProvider:
    """Direct Tavily integration for broad and non-Chinese discovery."""

    name = "tavily"

    async def search(self, request: SearchRequest, config: RunnableConfig) -> str:
        # Import lazily to keep the provider module independent of tool setup.
        from polyresearch.utils import tavily_search

        return await tavily_search.coroutine(
            [request.query], query_language=request.language, config=config
        )


class BailianWebSearchProvider:
    """The allowlisted Bailian MCP Web Search provider for Chinese discovery."""

    name = "bailian_web_search"

    async def search(self, request: SearchRequest, config: RunnableConfig) -> str:
        from polyresearch.utils import load_bailian_web_search_tool

        tools = await load_bailian_web_search_tool(config, existing_tool_names=set())
        if len(tools) != 1:
            raise ToolException(
                "Bailian Web Search is unavailable. Configure its allowlisted "
                "web_search MCP tool and API key before Chinese discovery."
            )
        # Bailian owns its input schema. Only pass the documented search query;
        # locale and language are controlled by the narrow Bailian configuration.
        return await tools[0].ainvoke({"query": request.query}, config=config)


class SearchProviderRouter:
    """Choose a provider only for a language/source type authorized by a plan."""

    def route(self, request: SearchRequest, plan: ResearchPlan):
        selected_language = next(
            (
                language
                for language in plan.ranked_languages
                if language.language.casefold() == request.language.casefold()
            ),
            None,
        )
        if selected_language is None:
            raise ToolException(
                f"Language '{request.language}' is not selected in the research plan."
            )
        if request.target_source_type not in selected_language.expected_source_types:
            raise ToolException(
                f"Source type '{request.target_source_type}' is not planned for "
                f"language '{request.language}'."
            )
        if selected_language.language.casefold().startswith("zh"):
            return BailianWebSearchProvider()
        return TavilySearchProvider()


def _research_plan_from_config(config: RunnableConfig) -> ResearchPlan:
    raw_plan = config.get("configurable", {}).get("research_plan")
    if raw_plan is None:
        raise ToolException("Discovery requires a persisted multilingual research plan.")
    if isinstance(raw_plan, ResearchPlan):
        return raw_plan
    return ResearchPlan.model_validate(raw_plan)


@tool("planned_web_search")
async def planned_web_search(
    query: str,
    language: str,
    target_source_type: str,
    config: RunnableConfig = None,
) -> str:
    """Search only a language and source type selected by the multilingual plan."""
    if config is None:
        raise ToolException("Discovery requires runtime configuration.")
    request = SearchRequest(
        query=query,
        language=language,
        target_source_type=target_source_type,
    )
    provider = SearchProviderRouter().route(request, _research_plan_from_config(config))
    return await provider.search(request, config)
