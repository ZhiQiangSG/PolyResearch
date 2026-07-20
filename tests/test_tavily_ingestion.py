import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from polyresearch.models import (
    AtomicSubquestion,
    LanguageDecision,
    LanguageSelectionAssessment,
    ResearchLanguage,
    ResearchPlan,
    ResearchRun,
)
from polyresearch.repositories import SqliteEvidenceRepository
from polyresearch.retrieval import mcp_utils, search_utils as utils
from polyresearch.runtime import tool_registry
from polyresearch.configuration import Configuration
from polyresearch.retrieval.search_providers import (
    BailianWebSearchProvider,
    SearchProviderRouter,
    SearchRequest,
    TavilySearchProvider,
    planned_web_search,
)
from polyresearch.nodes.provenance import persist_non_tavily_tool_outputs as _persist_non_tavily_tool_outputs


def _routing_plan() -> ResearchPlan:
    assessment = LanguageSelectionAssessment(
        place_and_institutional_jurisdiction="Fixture jurisdiction.",
        primary_actors_and_official_records="Fixture official records.",
        scholarly_technical_and_media_ecosystems="Fixture ecosystem.",
        diasporic_or_regional_coverage="Not applicable.",
        primary_source_availability="Fixture primary sources.",
        marginal_information_gain="Fixture information gain.",
    )
    languages = [
        ResearchLanguage(
            language="zh",
            priority=1,
            query_budget=2,
            expected_unique_value="Chinese primary sources.",
            selection_rationale="Chinese discovery.",
            selection_assessment=assessment,
            expected_source_types=["official"],
        ),
        ResearchLanguage(
            language="en",
            priority=2,
            query_budget=1,
            expected_unique_value="Broad bridge coverage.",
            selection_rationale="English discovery.",
            selection_assessment=assessment,
            expected_source_types=["news"],
        ),
    ]
    return ResearchPlan(
        run_id=uuid4(),
        subquestions=[
            AtomicSubquestion(question="What happened?", answer_scope="Find evidence.")
        ],
        ranked_languages=languages,
        language_decisions=[
            LanguageDecision(language="zh", status="selected", rationale="Primary records."),
            LanguageDecision(language="en", status="selected", rationale="Bridge coverage."),
        ],
        language_rationale={"zh": "Primary records.", "en": "Bridge coverage."},
        query_variants={"zh": ["政策"], "en": ["policy"]},
    )


class TavilyIngestionTests(unittest.IsolatedAsyncioTestCase):
    def test_bailian_configuration_rejects_non_allowlisted_tools(self) -> None:
        with self.assertRaises(ValueError):
            Configuration(
                bailian_web_search={
                    "tool_name": "filesystem_read",
                    "authentication": {"api_key": "test"},
                }
            )

    async def test_planned_search_is_available_without_bailian(self) -> None:
        tools = await tool_registry.get_all_tools({"configurable": {}})
        self.assertIn("planned_web_search", [tool.name for tool in tools])

    def test_router_selects_bailian_for_chinese_and_tavily_otherwise(self) -> None:
        router = SearchProviderRouter()
        plan = _routing_plan()
        self.assertIsInstance(
            router.route(SearchRequest("政策", "zh", "official"), plan),
            BailianWebSearchProvider,
        )
        self.assertIsInstance(
            router.route(SearchRequest("policy", "en", "news"), plan),
            TavilySearchProvider,
        )
        with self.assertRaises(Exception):
            router.route(SearchRequest("policy", "en", "official"), plan)
        plan.ranked_languages[0].expected_source_types.append("bridge")
        self.assertIsInstance(
            router.route(SearchRequest("政策", "zh", "bridge"), plan),
            TavilySearchProvider,
        )

    async def test_chinese_bailian_failure_records_explicit_tavily_fallback(self) -> None:
        async def fake_search(*args, **kwargs):
            return [
                {
                    "query": "政策",
                    "results": [
                        {
                            "url": "https://example.test/policy",
                            "title": "Policy update",
                            "raw_content": "Primary policy evidence.",
                        }
                    ],
                }
            ]

        original_search = utils.tavily_search_async
        utils.tavily_search_async = fake_search
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteEvidenceRepository(Path(directory) / "research.db")
            run = ResearchRun(id=uuid4(), question="What changed?", output_language="en")
            plan = _routing_plan().model_copy(update={"run_id": run.id})
            try:
                await repository.create_run(run)
                payload = await planned_web_search.coroutine(
                    "政策",
                    "zh",
                    "official",
                    locale="zh-CN",
                    query_rationale="Seek Chinese official evidence.",
                    config={
                        "configurable": {
                            "run_id": str(run.id),
                            "evidence_repository": repository,
                            "research_plan": plan,
                        }
                    },
                )
                self.assertIn("polyresearch_evidence", payload)
                records = await repository.list_query_records(run.id)
                self.assertEqual([record.provider for record in records], ["bailian_web_search", "tavily"])
                self.assertIsNotNone(records[0].failure)
                self.assertEqual(records[1].fallback_from, "bailian_web_search")
                self.assertEqual(records[1].language, "zh")
                self.assertEqual(records[1].locale, "zh-CN")
            finally:
                repository.close()
                utils.tavily_search_async = original_search

    async def test_planned_search_routes_to_mocked_bailian_and_tavily_without_credentials(self) -> None:
        async def fake_tavily_search(*args, **kwargs):
            return [
                {
                    "query": "policy bridge coverage",
                    "results": [
                        {
                            "url": "https://example.test/bridge",
                            "title": "Bridge coverage",
                            "raw_content": "English bridge evidence.",
                        }
                    ],
                }
            ]

        class FakeBailianTool:
            name = "web_search"

            async def ainvoke(self, arguments, config=None):
                self.arguments = arguments
                return (
                    '{"results": [{"title": "中文官方来源", '
                    '"url": "https://example.test/chinese-policy", '
                    '"raw_content": "中文官方证据。"}]}'
                )

        fake_bailian_tool = FakeBailianTool()

        class FakeMcpClient:
            def __init__(self, config):
                self.config = config

            async def get_tools(self):
                return [fake_bailian_tool]

        original_tavily_search = utils.tavily_search_async
        original_mcp_client = mcp_utils.MultiServerMCPClient
        utils.tavily_search_async = fake_tavily_search
        mcp_utils.MultiServerMCPClient = FakeMcpClient
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteEvidenceRepository(Path(directory) / "research.db")
            run = ResearchRun(id=uuid4(), question="What changed?", output_language="en")
            plan = _routing_plan().model_copy(update={"run_id": run.id})
            config = {
                "configurable": {
                    "run_id": str(run.id),
                    "evidence_repository": repository,
                    "research_plan": plan,
                    "bailian_web_search": {
                        "authentication": {"api_key": "mock-bailian-key"}
                    },
                }
            }
            try:
                await repository.create_run(run)
                chinese_result = await planned_web_search.coroutine(
                    "政策", "zh", "official", locale="zh-CN", config=config
                )
                english_result = await planned_web_search.coroutine(
                    "policy bridge coverage", "en", "news", locale="en-US", config=config
                )
                records = await repository.list_query_records(run.id)
                self.assertIn("polyresearch_evidence", chinese_result)
                self.assertIn("中文官方证据", chinese_result)
                self.assertIn("polyresearch_evidence", english_result)
                self.assertEqual(fake_bailian_tool.arguments, {"query": "政策"})
                self.assertEqual(
                    [record.provider for record in records],
                    ["bailian_web_search", "tavily"],
                )
                self.assertEqual(records[0].language, "zh")
                self.assertEqual(records[1].language, "en")
                self.assertEqual(len(await repository.list_sources(run.id)), 2)
                self.assertEqual(len(await repository.list_source_versions(run.id)), 2)
                self.assertEqual(len(await repository.list_passages(run.id)), 2)
                attachments = await repository.list_provenance_attachments(run.id)
                self.assertTrue(
                    any(attachment.provider == "bailian_web_search" for attachment in attachments)
                )
            finally:
                repository.close()
                utils.tavily_search_async = original_tavily_search
                mcp_utils.MultiServerMCPClient = original_mcp_client

    async def test_bailian_loads_only_allowlisted_web_search_tool(self) -> None:
        captured_config = None

        class FakeMcpClient:
            def __init__(self, config):
                nonlocal captured_config
                captured_config = config

            async def get_tools(self):
                return [
                    SimpleNamespace(name="web_search"),
                    SimpleNamespace(name="unrelated_remote_tool"),
                ]

        original_client = mcp_utils.MultiServerMCPClient
        mcp_utils.MultiServerMCPClient = FakeMcpClient
        try:
            tools = await mcp_utils.load_bailian_web_search_tool(
                {
                    "configurable": {
                        "bailian_web_search": {
                            "authentication": {"api_key": "test-key"}
                        },
                    }
                },
                existing_tool_names=set(),
            )
            self.assertIn("web_search", [tool.name for tool in tools])
            self.assertNotIn("unrelated_remote_tool", [tool.name for tool in tools])
            self.assertEqual(
                captured_config["bailian_web_search"]["headers"],
                {"Authorization": "Bearer test-key"},
            )
        finally:
            mcp_utils.MultiServerMCPClient = original_client
    async def test_search_persists_evidence_before_returning_typed_payload(self) -> None:
        async def fake_search(*args, **kwargs):
            return [
                {
                    "query": "policy update",
                    "results": [
                        {
                            "url": "https://example.test/policy",
                            "title": "Policy update",
                            "raw_content": "First paragraph.\n\nSecond paragraph.",
                        }
                    ],
                }
            ]

        original_search = utils.tavily_search_async
        utils.tavily_search_async = fake_search
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteEvidenceRepository(Path(directory) / "research.db")
            run = ResearchRun(
                id=uuid4(), question="What changed?", output_language="en"
            )
            try:
                await repository.create_run(run)
                payload = json.loads(
                    await utils.tavily_search.coroutine(
                        ["policy update"],
                        config={
                        "configurable": {
                            "run_id": str(run.id),
                            "evidence_repository": repository,
                        },
                    },
                    query_language="fr",
                    locale="fr-FR",
                    start_date="2025-01-01",
                    end_date="2025-12-31",
                    target_source_type="official",
                    query_rationale="Find the official policy update.",
                )
                )

                self.assertEqual(payload["type"], "polyresearch_evidence")
                query_records = await repository.list_query_records(run.id)
                self.assertEqual(len(query_records), 1)
                self.assertEqual(query_records[0].run_id, run.id)
                self.assertEqual(query_records[0].language, "fr")
                self.assertEqual(query_records[0].locale, "fr-FR")
                self.assertEqual(query_records[0].target_source_type, "official")
                self.assertEqual(query_records[0].rationale, "Find the official policy update.")
                self.assertEqual(str(query_records[0].date_from), "2025-01-01")
                self.assertEqual(str(query_records[0].date_to), "2025-12-31")
                self.assertEqual(len(await repository.list_provenance_attachments(run.id)), 1)
                sources = await repository.list_sources(run.id)
                self.assertEqual(len(sources), 1)
                self.assertEqual(sources[0].planned_query_language, "fr")
                self.assertEqual(sources[0].content_language, "en")
                self.assertFalse(sources[0].language_matches_planned_query)
                self.assertEqual(len(await repository.list_source_versions(run.id)), 1)
                passages = await repository.list_passages(run.id)
                self.assertEqual([passage.locator for passage in passages], ["paragraph-1", "paragraph-2"])
                self.assertEqual([passage.text for passage in passages], ["First paragraph.", "Second paragraph."])
            finally:
                repository.close()
                utils.tavily_search_async = original_search

    async def test_search_canonicalizes_urls_preserves_redirects_and_deduplicates(self) -> None:
        async def fake_search(*args, **kwargs):
            return [
                {
                    "query": "policy update",
                    "results": [
                        {
                            "url": "HTTPS://Example.TEST/policy?id=1&utm_source=newsletter#top",
                            "title": "Policy update",
                            "raw_content": "First copy.",
                            "redirect_chain": [
                                "http://example.test/policy?id=1",
                                "https://example.test/policy?id=1",
                            ],
                        },
                        {
                            "url": "https://example.test/policy?id=1",
                            "title": "Duplicate policy update",
                            "raw_content": "Second copy should not be fetched.",
                        },
                    ],
                }
            ]

        original_search = utils.tavily_search_async
        utils.tavily_search_async = fake_search
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteEvidenceRepository(Path(directory) / "research.db")
            run = ResearchRun(id=uuid4(), question="What changed?", output_language="en")
            try:
                await repository.create_run(run)
                await utils.tavily_search.coroutine(
                    ["policy update"],
                    config={
                        "configurable": {
                            "run_id": str(run.id),
                            "evidence_repository": repository,
                        }
                    },
                )
                sources = await repository.list_sources(run.id)
                records = await repository.list_query_records(run.id)
                self.assertEqual(len(sources), 1)
                self.assertEqual(sources[0].canonical_url, "https://example.test/policy?id=1")
                self.assertEqual(
                    sources[0].discovered_url,
                    "HTTPS://Example.TEST/policy?id=1&utm_source=newsletter#top",
                )
                self.assertEqual(len(sources[0].redirect_chain), 2)
                self.assertEqual([record.result_rank for record in records], [1, 2])
                self.assertEqual(
                    {record.result_url for record in records},
                    {"https://example.test/policy?id=1"},
                )
            finally:
                repository.close()
                utils.tavily_search_async = original_search

    async def test_search_persists_extracted_metadata_and_structure(self) -> None:
        async def fake_search(*args, **kwargs):
            return [{
                "query": "official policy",
                "results": [{
                    "url": "https://example.test/discovered?utm_source=newsletter",
                    "raw_content": """
                        <html lang=\"en\"><head>
                        <link rel=\"canonical\" href=\"/official-policy\" />
                        <meta property=\"og:title\" content=\"Official policy\" />
                        <meta property=\"og:site_name\" content=\"Policy Office\" />
                        <meta name=\"author\" content=\"Policy Team\" />
                        <meta property=\"article:published_time\" content=\"2026-01-02T03:04:05Z\" />
                        <meta property=\"article:modified_time\" content=\"2026-01-03T03:04:05Z\" />
                        </head><body><h1>Policy</h1><p>Original evidence.</p></body></html>
                    """,
                    "content_type": "text/html",
                }],
            }]

        original_search = utils.tavily_search_async
        utils.tavily_search_async = fake_search
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteEvidenceRepository(Path(directory) / "research.db")
            run = ResearchRun(id=uuid4(), question="What changed?", output_language="en")
            try:
                await repository.create_run(run)
                await utils.tavily_search.coroutine(
                    ["official policy"],
                    config={"configurable": {"run_id": str(run.id), "evidence_repository": repository}},
                )
                source = (await repository.list_sources(run.id))[0]
                self.assertEqual(source.canonical_url, "https://example.test/official-policy")
                self.assertEqual(source.title, "Official policy")
                self.assertEqual(source.publisher, "Policy Office")
                self.assertEqual(source.author, "Policy Team")
                self.assertEqual(source.published_at.isoformat(), "2026-01-02T03:04:05+00:00")
                self.assertEqual(source.updated_at.isoformat(), "2026-01-03T03:04:05+00:00")
                self.assertEqual(source.document_structure[0].heading, "Policy")
                self.assertEqual(source.document_structure[0].first_passage_locator, "Policy / paragraph-1")
            finally:
                repository.close()
                utils.tavily_search_async = original_search

    async def test_non_tavily_tool_output_is_kept_as_an_audit_attachment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteEvidenceRepository(Path(directory) / "research.db")
            run = ResearchRun(
                id=uuid4(), question="What changed?", output_language="en"
            )
            try:
                await repository.create_run(run)
                await _persist_non_tavily_tool_outputs(
                    {
                        "configurable": {
                            "run_id": str(run.id),
                            "evidence_repository": repository,
                        }
                    },
                    [{"name": "mcp_lookup"}],
                    ["untrusted remote payload"],
                )

                attachments = await repository.list_provenance_attachments(run.id)
                self.assertEqual(len(attachments), 1)
                self.assertEqual(attachments[0].tool_name, "mcp_lookup")
                self.assertEqual(attachments[0].raw_output, "untrusted remote payload")
            finally:
                repository.close()


if __name__ == "__main__":
    unittest.main()
