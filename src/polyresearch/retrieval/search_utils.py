"""Tavily discovery and typed evidence-ingestion helpers."""

import asyncio
import hashlib
import json
import logging
import re
from typing import Annotated, Any, Literal
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import aiohttp
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool
from tavily import AsyncTavilyClient

from polyresearch.retrieval.deduplication import deduplicate_source_artifacts
from polyresearch.models import EvidencePassage, ProvenanceAttachment, QueryRecord, SourceRecord, SourceVersion
from polyresearch.runtime.model_utils import get_tavily_api_key
from polyresearch.repositories import RunContext
from polyresearch.retrieval.source_ingestion import extract_document, fetch_source_content, languages_match
from polyresearch.retrieval.source_quality import score_initial_source_quality

TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events."
)

_TRACKING_QUERY_PARAMETERS = {"fbclid", "gclid", "mc_cid", "mc_eid"}
_PASSAGE_SELECTION_LIMIT = 12


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
        JSON evidence payload containing selected, citable original passages
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
        if original_text:
            document = extract_document(
                original_text,
                content_type=result.get("content_type") or result.get("mime_type"),
            )
            http_metadata = {
                key: result[key]
                for key in ("status", "status_code", "content_type", "etag", "last_modified")
                if result.get(key) is not None
            }
            extraction_method = "provider_content"
        else:
            try:
                document = await fetch_source_content(result["canonical_url"])
            except (aiohttp.ClientError, asyncio.TimeoutError, UnicodeError) as error:
                logging.getLogger(__name__).info(
                    "Unable to fetch discovered source %s: %s", result["canonical_url"], error
                )
                continue
            original_text = document.raw_content
            http_metadata = document.http_metadata
            extraction_method = document.extraction_method

        content_hash = hashlib.sha256(original_text.encode("utf-8")).hexdigest()
        source_canonical_url = result["canonical_url"]
        if document.canonical_url:
            try:
                source_canonical_url = canonicalize_url(
                    urljoin(result["discovered_url"], document.canonical_url)
                )
            except ValueError:
                logging.getLogger(__name__).info(
                    "Ignoring invalid extracted canonical URL for %s", result["discovered_url"]
                )
        source = SourceRecord(
            canonical_url=source_canonical_url,
            title=result.get("title") or document.title or result["canonical_url"],
            publisher=result.get("publisher") or document.publisher,
            author=result.get("author") or document.author,
            language=document.language or query_language,
            content_language=document.content_language,
            metadata_language=document.metadata_language,
            language_detection_method=document.language_detection_method,
            planned_query_language=query_language,
            language_matches_planned_query=languages_match(document.language, query_language),
            source_type=target_source_type or "web",
            published_at=document.published_at,
            updated_at=document.updated_at,
            content_hash=content_hash,
            extraction_quality=document.extraction_quality,
            extraction_notes=document.extraction_notes,
            document_structure=document.document_structure,
            research_unit_id=_research_unit_id_from_config(config),
            discovered_url=result["discovered_url"],
            redirect_chain=result["redirect_chain"],
        )
        source = source.model_copy(
            update={
                "initial_quality_assessment": score_initial_source_quality(
                    source, document.content, query=result.get("query")
                )
            }
        )
        source_version = SourceVersion(
            source_id=source.id,
            version_number=1,
            content_hash=content_hash,
            raw_content=original_text,
            http_metadata=http_metadata,
            extraction_method=extraction_method,
            extraction_quality=document.extraction_quality,
        )
        sources.append(source)
        source_versions.append(source_version)
        passages.extend(
            _chunk_evidence_passages(
                source, original_text, document.passages, extracted_content=document.content
            )
        )

    sources, source_versions, passages = await _deduplicate_source_artifacts(
        config, sources, source_versions, passages
    )
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
    selected_passages = select_citable_passages(sources, passages, " ".join(queries))

    return json.dumps(
        {
            "type": "polyresearch_evidence",
            "sources": [source.model_dump(mode="json") for source in sources],
            "passages": [passage.model_dump(mode="json") for passage in selected_passages],
            "selection": {
                "method": "deterministic_query_passage_ranking_v1",
                "selected_passage_ids": [str(passage.id) for passage in selected_passages],
                "supplemental_summaries": [],
            },
        },
        ensure_ascii=False,
    )


def _chunk_evidence_passages(
    source: SourceRecord,
    original_text: str,
    extracted_passages: list[tuple[str, str]] | None = None,
    *,
    extracted_content: str | None = None,
) -> list[EvidencePassage]:
    """Split original text into citable passages with structural and offset anchors."""
    rows = extracted_passages or [
        (f"paragraph-{index}", paragraph.strip())
        for index, paragraph in enumerate(re.split(r"\n\s*\n", original_text), start=1)
        if paragraph.strip()
    ]
    if not rows:
        return []
    locator_content = extracted_content if extracted_content is not None else original_text
    search_start = 0
    passages: list[EvidencePassage] = []
    for locator, text in rows:
        character_start = locator_content.find(text, search_start)
        if character_start < 0:
            # A provider may normalize whitespace differently from extraction. The
            # structural locator remains usable, while we avoid inventing offsets.
            character_end = None
        else:
            character_end = character_start + len(text)
            search_start = character_end
        heading = None
        if " / paragraph-" in locator:
            heading = locator.rsplit(" / paragraph-", 1)[0]
        passages.append(
            EvidencePassage(
                source_id=source.id,
                text=text,
                locator=locator,
                heading=heading,
                character_start=character_start if character_start >= 0 else None,
                character_end=character_end,
                original_language=source.language,
            )
        )
    return passages


def select_citable_passages(
    sources: list[SourceRecord],
    passages: list[EvidencePassage],
    query: str,
    *,
    limit: int = _PASSAGE_SELECTION_LIMIT,
) -> list[EvidencePassage]:
    """Rank exact original passages for a query; never synthesize a summary."""
    if limit < 1:
        return []
    source_by_id = {source.id: source for source in sources}
    query_terms = set(re.findall(r"\w+", query.casefold(), re.UNICODE))

    def score(passage: EvidencePassage) -> tuple[float, str]:
        source = source_by_id.get(passage.source_id)
        text_terms = set(re.findall(r"\w+", passage.text.casefold(), re.UNICODE))
        title_terms = set(
            re.findall(r"\w+", source.title.casefold(), re.UNICODE) if source else []
        )
        query_coverage = len(query_terms & text_terms) / len(query_terms) if query_terms else 0.0
        title_coverage = len(query_terms & title_terms) / len(query_terms) if query_terms else 0.0
        quality = (
            source.initial_quality_assessment.score
            if source and source.initial_quality_assessment
            else 0.5
        )
        return (query_coverage * 0.75 + title_coverage * 0.15 + quality * 0.10, str(passage.id))

    return sorted(passages, key=score, reverse=True)[:limit]


async def _deduplicate_source_artifacts(
    config: RunnableConfig | None,
    sources: list[SourceRecord],
    versions: list[SourceVersion],
    passages: list[EvidencePassage],
) -> tuple[list[SourceRecord], list[SourceVersion], list[EvidencePassage]]:
    """Apply run-scoped canonical, hash, near-copy, and origin clustering."""
    if not config:
        return deduplicate_source_artifacts(sources, versions, passages)
    try:
        context = RunContext.from_runnable_config(config)
    except ValueError:
        return deduplicate_source_artifacts(sources, versions, passages)
    existing_sources, existing_versions = await asyncio.gather(
        context.repository.list_sources(context.run_id),
        context.repository.list_source_versions(context.run_id),
    )
    return deduplicate_source_artifacts(
        sources,
        versions,
        passages,
        existing_sources=existing_sources,
        existing_versions=existing_versions,
    )


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

