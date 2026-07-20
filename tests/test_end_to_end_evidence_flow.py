import importlib
import json
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import HumanMessage

from polyresearch import utils
from polyresearch.models import (
    Claim,
    ClaimExtractionResult,
    ReportDraft,
    ReportStatementDraft,
)
from polyresearch.repositories import SqliteEvidenceRepository

graph_module = importlib.import_module("polyresearch.graph")


class _ClaimExtractorStub:
    def __init__(self, claim: Claim) -> None:
        self.claim = claim

    def with_structured_output(self, schema):
        return self

    def with_retry(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        return ClaimExtractionResult(claims=[self.claim])


class _ReportWriterStub:
    def __init__(self, claim_id) -> None:
        self.claim_id = claim_id

    def with_structured_output(self, schema):
        return self

    def with_retry(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        return ReportDraft(
            title="Policy update",
            statements=[
                ReportStatementDraft(
                    rendered_text=(
                        "Available evidence indicates that the policy changed on 1 January."
                    ),
                    claim_ids=[self.claim_id],
                )
            ],
        )


class EndToEndEvidenceFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_retrieval_to_restart_to_cited_report(self) -> None:
        async def fake_search(*args, **kwargs):
            return [
                {
                    "query": "policy update",
                    "results": [
                        {
                            "url": "https://example.test/policy",
                            "title": "Official policy update",
                            "raw_content": "The policy changed on 1 January.",
                        }
                    ],
                }
            ]

        original_search = utils.tavily_search_async
        original_factory = graph_module.create_qwen_chat_model
        utils.tavily_search_async = fake_search
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "research.db"
            repository = SqliteEvidenceRepository(database_path)
            run_id = uuid4()
            config = {
                "configurable": {
                    "run_id": str(run_id),
                    "evidence_repository": repository,
                    "output_language": "en",
                }
            }
            try:
                await graph_module.initialize_research_run(
                    {"messages": [HumanMessage(content="What changed in the policy?")]},
                    config,
                )
                search_payload = json.loads(
                    await utils.tavily_search.coroutine(["policy update"], config=config)
                )
                persisted_passage_id = search_payload["passages"][0]["id"]
                claim = Claim(
                    statement="The policy changed on 1 January.",
                    evidence_passage_ids=[persisted_passage_id],
                    extraction_confidence=0.9,
                )

                graph_module.create_qwen_chat_model = (
                    lambda *args, **kwargs: _ClaimExtractorStub(claim)
                )
                await graph_module.extract_claims({}, config)

                graph_module.create_qwen_chat_model = (
                    lambda *args, **kwargs: _ReportWriterStub(claim.id)
                )
                report_result = await graph_module.final_report_generation(
                    {
                        "messages": [HumanMessage(content="What changed in the policy?")],
                        "research_brief": "What changed in the policy?",
                    },
                    config,
                )
                self.assertIn(f"[P:{persisted_passage_id}]", report_result["final_report"])

                # A new repository instance resumes from the durable evidence ledger.
                repository.close()
                repository = SqliteEvidenceRepository(database_path)

                run = await repository.get_run(run_id)
                queries = await repository.list_query_records(run_id)
                sources = await repository.list_sources(run_id)
                versions = await repository.list_source_versions(run_id)
                passages = await repository.list_passages(run_id)
                claims = await repository.list_claims(run_id)
                links = await repository.list_evidence_links(run_id)
                statements = await repository.list_report_statements(run_id)
                bundles = await repository.list_report_bundles(run_id)

                self.assertEqual(run.id, run_id)
                self.assertEqual(len(queries), 1)
                self.assertEqual(len(sources), 1)
                self.assertEqual(len(versions), 1)
                self.assertEqual(len(passages), 1)
                self.assertEqual(len(claims), 1)
                self.assertEqual(len(links), 1)
                self.assertEqual(len(statements), 1)
                self.assertEqual(statements[0].citation_ids, [passages[0].id])
                self.assertEqual(links[0].claim_id, statements[0].claim_ids[0])
                self.assertEqual(links[0].passage_id, passages[0].id)
                self.assertEqual(passages[0].source_id, sources[0].id)
                self.assertIn(f"[P:{passages[0].id}]", bundles[0].markdown or "")
            finally:
                graph_module.create_qwen_chat_model = original_factory
                utils.tavily_search_async = original_search
                repository.close()
