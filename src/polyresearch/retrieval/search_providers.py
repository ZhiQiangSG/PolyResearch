"""Provider-routed discovery constrained by the persisted multilingual plan."""

import asyncio
import hashlib
import json
import logging
import os
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field, replace
from threading import Lock
from time import perf_counter
from urllib.parse import urljoin

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
    TraceRecord,
)
from polyresearch.repositories import RunContext
from polyresearch.security import redacted_exception_info, redact_secrets

logger = logging.getLogger(__name__)

_GENERIC_DISCOVERY_OBSERVATION = (
    "Discovery did not return usable evidence. Try another planned query or source type."
)


def _sanitized_failure(error: BaseException) -> str:
    """Retain useful diagnostics without exposing credentials to durable artifacts."""
    return f"{type(error).__name__}: {redact_secrets(str(error))}"


@dataclass
class _BailianRateLimiter:
    """One in-process reservation queue for a Bailian provider endpoint."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    next_allowed_at: float = 0.0
    last_used_at: float = 0.0


_BAILIAN_RATE_LIMITERS: dict[str, _BailianRateLimiter] = {}
_BAILIAN_LIMITER_REGISTRY_LOCK = Lock()
_BAILIAN_LIMITER_IDLE_TTL_SECONDS = 60 * 60


def _bailian_limiter_key(bailian) -> str:
    """Identify a provider account without retaining its credential in memory."""
    api_key = bailian.authentication.api_key or os.getenv(
        bailian.authentication.api_key_env_var
    )
    if not api_key:
        # The MCP loader rejects this configuration before a request is made.
        # Keep the fallback deterministic for isolated tool tests.
        api_key = "missing-bailian-api-key"
    credential_fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return f"{bailian.server_url}|{credential_fingerprint}"


async def _reserve_bailian_request_slot(
    provider_key: str,
    minimum_interval: float,
    *,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], Awaitable[object]] | None = None,
) -> None:
    """Reserve one request slot without holding the lock during network I/O."""
    monotonic = clock or asyncio.get_running_loop().time
    sleeper = sleep or asyncio.sleep
    limiter = _get_bailian_rate_limiter(provider_key, monotonic())
    async with limiter.lock:
        wait_seconds = limiter.next_allowed_at - monotonic()
        if wait_seconds > 0:
            await sleeper(wait_seconds)
        # Record the actual reservation time after any scheduler delay, so later
        # callers cannot obtain the same slot.
        reserved_at = monotonic()
        limiter.next_allowed_at = reserved_at + minimum_interval
        limiter.last_used_at = reserved_at


def _get_bailian_rate_limiter(provider_key: str, now: float) -> _BailianRateLimiter:
    """Get a limiter while pruning idle provider accounts from long-lived workers."""
    with _BAILIAN_LIMITER_REGISTRY_LOCK:
        stale_keys = [
            key
            for key, limiter in _BAILIAN_RATE_LIMITERS.items()
            if not limiter.lock.locked()
            and now - limiter.last_used_at >= _BAILIAN_LIMITER_IDLE_TTL_SECONDS
            and now >= limiter.next_allowed_at
        ]
        for key in stale_keys:
            del _BAILIAN_RATE_LIMITERS[key]
        limiter = _BAILIAN_RATE_LIMITERS.get(provider_key)
        if limiter is None:
            limiter = _BailianRateLimiter(last_used_at=now)
            _BAILIAN_RATE_LIMITERS[provider_key] = limiter
        else:
            limiter.last_used_at = now
        return limiter


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
    source_budget: int = 5


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
        from polyresearch.retrieval.search_utils import tavily_search

        return await tavily_search.coroutine(
            [request.query],
            max_results=request.source_budget,
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
        from polyresearch.retrieval.mcp_utils import load_bailian_web_search_tool
        from polyresearch.retrieval.search_utils import select_citable_passages

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
        minimum_interval = 1 / bailian.max_requests_per_second
        await _reserve_bailian_request_slot(
            _bailian_limiter_key(bailian), minimum_interval
        )
        result = await asyncio.wait_for(
            tools[0].ainvoke({"query": request.query}, config=config),
            timeout=bailian.timeout_seconds,
        )
        sources, passages = await _persist_bailian_ingestion(request, config, result)
        return json.dumps(
            {
                "type": "polyresearch_evidence",
                "sources": [source.model_dump(mode="json") for source in sources],
                "passages": [
                    passage.model_dump(mode="json")
                    for passage in select_citable_passages(sources, passages, request.query)
                ],
                "selection": {
                    "method": "deterministic_query_passage_ranking_v1",
                    "supplemental_summaries": [],
                },
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
        logger.debug("Skipping Bailian query persistence without run context")
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
            logger.debug(
                "Ignoring malformed Bailian result payload",
                extra={"operation": "parse_bailian_result", "provider": "bailian_web_search"},
            )
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
        logger.debug("Skipping Bailian ingestion persistence without run context")
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
                raw_output=redact_secrets(raw_output),
            )
        ],
    )

    # Reuse the same URL and passage semantics as the direct Tavily path.
    from polyresearch.retrieval.search_utils import (
        _chunk_evidence_passages,
        _deduplicate_source_artifacts,
        _redirect_chain,
        canonicalize_url,
    )
    from polyresearch.retrieval.source_ingestion import extract_document, languages_match
    from polyresearch.retrieval.source_quality import score_initial_source_quality

    sources: list[SourceRecord] = []
    source_versions: list[SourceVersion] = []
    passages: list[EvidencePassage] = []
    query_records: list[QueryRecord] = []
    seen_urls: set[str] = set()
    for result_rank, row in enumerate(_bailian_result_rows(raw_result)[:request.source_budget], start=1):
        discovered_url = row.get("url") or row.get("link") or row.get("source_url")
        if not isinstance(discovered_url, str):
            continue
        try:
            canonical_url = canonicalize_url(discovered_url)
        except ValueError:
            logger.debug(
                "Skipping Bailian result with invalid URL",
                extra={"operation": "canonicalize_bailian_url", "provider": "bailian_web_search", "query_language": request.language},
            )
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
        document = extract_document(
            original_text,
            content_type=row.get("content_type") or row.get("mime_type"),
        )
        content_hash = hashlib.sha256(original_text.encode("utf-8")).hexdigest()
        source_canonical_url = canonical_url
        if document.canonical_url:
            try:
                source_canonical_url = canonicalize_url(
                    urljoin(discovered_url, document.canonical_url)
                )
            except ValueError:
                logger.debug(
                    "Ignoring invalid extracted Bailian canonical URL",
                    extra={"operation": "canonicalize_extracted_url", "provider": "bailian_web_search", "query_language": request.language},
                )
                pass
        source = SourceRecord(
            canonical_url=source_canonical_url,
            discovered_url=discovered_url,
            redirect_chain=_redirect_chain(row, discovered_url),
            title=row.get("title") or document.title or canonical_url,
            publisher=row.get("publisher") or document.publisher,
            author=row.get("author") or document.author,
            language=document.language or request.language,
            content_language=document.content_language,
            metadata_language=document.metadata_language,
            language_detection_method=document.language_detection_method,
            planned_query_language=request.language,
            language_matches_planned_query=languages_match(document.language, request.language),
            source_type=request.target_source_type,
            content_hash=content_hash,
            extraction_quality=document.extraction_quality,
            extraction_notes=document.extraction_notes,
            document_structure=document.document_structure,
            research_unit_id=context.research_unit_id,
        )
        source = source.model_copy(
            update={
                "initial_quality_assessment": score_initial_source_quality(
                    source, document.content, query=request.query
                )
            }
        )
        sources.append(source)
        source_versions.append(
            SourceVersion(
                source_id=source.id,
                version_number=1,
                content_hash=content_hash,
                raw_content=original_text,
                http_metadata={
                    key: row[key]
                    for key in ("status", "status_code", "content_type", "etag", "last_modified")
                    if row.get(key) is not None
                },
                extraction_method="provider_content",
                extraction_quality=document.extraction_quality,
            )
        )
        passages.extend(
            _chunk_evidence_passages(
                source, original_text, document.passages, extracted_content=document.content
            )
        )

    sources, source_versions, passages = await _deduplicate_source_artifacts(
        config, sources, source_versions, passages
    )
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
            logger.warning(
                "Bailian discovery failed; falling back to Tavily",
                extra={
                    "operation": "provider_routed_discovery",
                    "provider": provider.name,
                    "fallback_provider": "tavily",
                    "query_language": request.language,
                    "target_source_type": request.target_source_type,
                    "run_id": str(config.get("configurable", {}).get("run_id", "")),
                },
                exc_info=redacted_exception_info(error),
            )
            # Preserve the failed Bailian attempt before using a non-equivalent
            # fallback. The succeeding Tavily record points back to this attempt.
            await _persist_bailian_query_record(
                request,
                config,
                failure=_sanitized_failure(error),
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
    context = RunContext.from_runnable_config(config)
    configurable = Configuration.from_runnable_config(config)
    request = SearchRequest(
        query=query,
        language=language,
        target_source_type=target_source_type,
        locale=locale,
        date_from=date_from,
        date_to=date_to,
        rationale=query_rationale,
    )
    plan = _research_plan_from_config(config)
    router = SearchProviderRouter()
    provider = router.route(request, plan)
    try:
        reservation = await context.repository.reserve_discovery_budget(
            context.run_id,
            max_queries=configurable.max_queries_per_run,
            max_sources=configurable.max_source_fetches_per_run,
            requested_sources=5,
        )
    except ValueError as error:
        logger.warning(
            "Discovery budget reservation failed",
            extra={
                "operation": "reserve_discovery_budget",
                "provider": provider.name,
                "query_language": language,
                "target_source_type": target_source_type,
                "run_id": str(context.run_id),
                "failure": _sanitized_failure(error),
            },
            exc_info=redacted_exception_info(error),
        )
        raise ToolException(_GENERIC_DISCOVERY_OBSERVATION) from None
    request = replace(request, source_budget=reservation.source_slots)
    started_at = datetime.now(timezone.utc)
    started = perf_counter()
    try:
        result = await router.search(request, plan, config)
        failure = None
    except Exception as error:
        result = None
        failure = _sanitized_failure(error)
        logger.warning(
            "Provider-routed discovery failed",
            extra={
                "operation": "provider_routed_discovery",
                "provider": provider.name,
                "query_language": language,
                "target_source_type": target_source_type,
                "run_id": str(context.run_id),
            },
            exc_info=redacted_exception_info(error),
        )
    try:
        sources_used = len(json.loads(result).get("sources", [])) if result else 0
        await context.repository.finalize_discovery_budget(
            reservation, sources_used=sources_used
        )
    except Exception as error:
        logger.warning(
            "Discovery budget finalization failed",
            extra={"operation": "finalize_discovery_budget", "run_id": str(context.run_id)},
            exc_info=redacted_exception_info(error),
        )
        if failure is None:
            failure = _sanitized_failure(error)
    try:
        context = RunContext.from_runnable_config(config)
        records = await context.repository.list_query_records(context.run_id)
        matching = [
            record for record in records
            if record.query == query and record.language == language
        ]
        await context.repository.append_trace_records(context.run_id, [TraceRecord(
            run_id=context.run_id,
            operation="provider_routed_discovery",
            provider=provider.name,
            query_ids=[record.id for record in matching],
            graph_artifact_ids=[f"query:{record.id}" for record in matching],
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            latency_ms=(perf_counter() - started) * 1000,
            retry_count=0,
            cost_note="Search-provider cost is not supplied by the provider API.",
            provider_failure=failure or next((record.failure for record in matching if record.failure), None),
        )])
    except ValueError:
        logger.debug(
            "Skipping provider trace persistence without run context",
            extra={"operation": "persist_provider_trace", "provider": provider.name},
        )
        pass
    if failure is not None:
        raise ToolException(_GENERIC_DISCOVERY_OBSERVATION)
    return result
