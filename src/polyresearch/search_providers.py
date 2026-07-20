"""Provider-routed discovery constrained by the persisted multilingual plan."""

import asyncio
import hashlib
import json
from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import ToolException, tool

from polyresearch.configuration import Configuration
from polyresearch.models import (
    EvidencePassage,
    ProvenanceAttachment,
    QueryRecord,
    ResearchPlan,
    SourceRecord,
    SourceVersion,
)
from polyresearch.repositories import RunContext


@dataclass(frozen=True)
class SearchRequest:
    """One language- and source-type-specific discovery request."""

    query: str
    language: str
    target_source_type: str
    locale: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    rationale: str | None = None


class TavilySearchProvider:
    """Direct Tavily integration for broad and non-Chinese discovery."""

    name = "tavily"

    async def search(
        self,
        request: SearchRequest,
        config: RunnableConfig,
        *,
        fallback_from: str | None = None,
    ) -> str:
        # Import lazily to keep the provider module independent of tool setup.
        from polyresearch.utils import tavily_search

        return await tavily_search.coroutine(
            [request.query],
            query_language=request.language,
            locale=request.locale,
            start_date=request.date_from,
            end_date=request.date_to,
            target_source_type=request.target_source_type,
            query_rationale=request.rationale,
            fallback_from=fallback_from,
            config=config,
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
        bailian = Configuration.from_runnable_config(config).bailian_web_search
        if bailian is None:  # Defensive guard; the loader already checks this.
            raise ToolException("Bailian Web Search is not configured.")
        result = await asyncio.wait_for(
            tools[0].ainvoke({"query": request.query}, config=config),
            timeout=bailian.timeout_seconds,
        )
        sources, passages = await _persist_bailian_ingestion(request, config, result)
        return json.dumps(
            {
                "type": "polyresearch_evidence",
                "sources": [source.model_dump(mode="json") for source in sources],
                "passages": [passage.model_dump(mode="json") for passage in passages],
            },
            ensure_ascii=False,
        )


async def _persist_bailian_query_record(
    request: SearchRequest,
    config: RunnableConfig,
    *,
    failure: str | None = None,
    fallback_from: str | None = None,
) -> None:
    """Record Bailian's plan-driven query metadata before later ingestion stages."""
    try:
        context = RunContext.from_runnable_config(config)
    except ValueError:
        return
    await context.repository.append_query_records(
        context.run_id,
        [
            QueryRecord(
                run_id=context.run_id,
                research_unit_id=context.research_unit_id,
                query=request.query,
                language=request.language,
                locale=request.locale,
                provider="bailian_web_search",
                target_source_type=request.target_source_type,
                rationale=request.rationale,
                date_from=request.date_from,
                date_to=request.date_to,
                failure=failure,
                fallback_from=fallback_from,
            )
        ],
    )


def _bailian_result_rows(result: object) -> list[dict]:
    """Extract common MCP Web Search result shapes without trusting tool content."""
    payload = result
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return []
    if not isinstance(payload, dict):
        return []
    for key in ("results", "items", "data"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


async def _persist_bailian_ingestion(
    request: SearchRequest, config: RunnableConfig, raw_result: object
) -> tuple[list[SourceRecord], list[EvidencePassage]]:
    """Normalize Bailian discovery into the same typed ledger artifacts as Tavily."""
    try:
        context = RunContext.from_runnable_config(config)
    except ValueError:
        return [], []

    # Keep remote output immutable for audit while treating only extracted fields as data.
    raw_output = (
        raw_result
        if isinstance(raw_result, str)
        else json.dumps(raw_result, ensure_ascii=False, sort_keys=True, default=str)
    )
    await context.repository.append_provenance_attachments(
        context.run_id,
        [
            ProvenanceAttachment(
                run_id=context.run_id,
                provider="bailian_web_search",
                tool_name="web_search",
                raw_output=raw_output,
            )
        ],
    )

    # Reuse the same URL and passage semantics as the direct Tavily path.
    from polyresearch.utils import _chunk_evidence_passages, _redirect_chain, canonicalize_url

    sources: list[SourceRecord] = []
    source_versions: list[SourceVersion] = []
    passages: list[EvidencePassage] = []
    query_records: list[QueryRecord] = []
    seen_urls: set[str] = set()
    for result_rank, row in enumerate(_bailian_result_rows(raw_result), start=1):
        discovered_url = row.get("url") or row.get("link") or row.get("source_url")
        if not isinstance(discovered_url, str):
            continue
        try:
            canonical_url = canonicalize_url(discovered_url)
        except ValueError:
            continue
        query_records.append(
            QueryRecord(
                run_id=context.run_id,
                research_unit_id=context.research_unit_id,
                query=request.query,
                language=request.language,
                locale=request.locale,
                provider="bailian_web_search",
                target_source_type=request.target_source_type,
                rationale=request.rationale,
                date_from=request.date_from,
                date_to=request.date_to,
                result_rank=result_rank,
                result_url=canonical_url,
            )
        )
        if canonical_url in seen_urls:
            continue
        seen_urls.add(canonical_url)
        original_text = (
            row.get("raw_content")
            or row.get("content")
            or row.get("snippet")
            or row.get("text")
        )
        if not isinstance(original_text, str) or not original_text.strip():
            continue
        content_hash = hashlib.sha256(original_text.encode("utf-8")).hexdigest()
        source = SourceRecord(
            canonical_url=canonical_url,
            discovered_url=discovered_url,
            redirect_chain=_redirect_chain(row, discovered_url),
            title=row.get("title") or canonical_url,
            language=request.language,
            source_type=request.target_source_type,
            content_hash=content_hash,
            research_unit_id=context.research_unit_id,
        )
        sources.append(source)
        source_versions.append(
            SourceVersion(
                source_id=source.id,
                version_number=1,
                content_hash=content_hash,
                raw_content=original_text,
            )
        )
        passages.extend(_chunk_evidence_passages(source, original_text))

    if not query_records:
        query_records = [
            QueryRecord(
                run_id=context.run_id,
                research_unit_id=context.research_unit_id,
                query=request.query,
                language=request.language,
                locale=request.locale,
                provider="bailian_web_search",
                target_source_type=request.target_source_type,
                rationale=request.rationale,
                date_from=request.date_from,
                date_to=request.date_to,
            )
        ]
    await context.repository.append_query_records(context.run_id, query_records)
    if sources:
        await context.repository.append_sources(context.run_id, sources)
        await context.repository.append_source_versions(context.run_id, source_versions)
        await context.repository.append_passages(context.run_id, passages)
    return sources, passages


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
        if request.target_source_type in {"bridge", "cross_language_bridge"}:
            return TavilySearchProvider()
        if selected_language.language.casefold().startswith("zh"):
            return BailianWebSearchProvider()
        return TavilySearchProvider()

    async def search(
        self, request: SearchRequest, plan: ResearchPlan, config: RunnableConfig
    ) -> str:
        """Execute a routed request and make any provider substitution explicit."""
        provider = self.route(request, plan)
        if not isinstance(provider, BailianWebSearchProvider):
            return await provider.search(request, config)
        try:
            return await provider.search(request, config)
        except Exception as error:
            # Preserve the failed Bailian attempt before using a non-equivalent
            # fallback. The succeeding Tavily record points back to this attempt.
            await _persist_bailian_query_record(
                request,
                config,
                failure=str(error),
            )
            return await TavilySearchProvider().search(
                request,
                config,
                fallback_from=provider.name,
            )


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
    locale: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    query_rationale: str | None = None,
    config: RunnableConfig = None,
) -> str:
    """Search only a language and source type selected by the multilingual plan."""
    if config is None:
        raise ToolException("Discovery requires runtime configuration.")
    request = SearchRequest(
        query=query,
        language=language,
        target_source_type=target_source_type,
        locale=locale,
        date_from=date_from,
        date_to=date_to,
        rationale=query_rationale,
    )
    return await SearchProviderRouter().search(
        request, _research_plan_from_config(config), config
    )
