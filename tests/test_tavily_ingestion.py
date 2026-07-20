import json
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from polyresearch.models import ResearchRun
from polyresearch.repositories import SqliteEvidenceRepository
from polyresearch import utils
from polyresearch.graph import _persist_non_tavily_tool_outputs


class TavilyIngestionTests(unittest.IsolatedAsyncioTestCase):
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
                            }
                        },
                    )
                )

                self.assertEqual(payload["type"], "polyresearch_evidence")
                self.assertEqual(len(await repository.list_query_records(run.id)), 1)
                self.assertEqual(len(await repository.list_provenance_attachments(run.id)), 1)
                self.assertEqual(len(await repository.list_sources(run.id)), 1)
                self.assertEqual(len(await repository.list_source_versions(run.id)), 1)
                passages = await repository.list_passages(run.id)
                self.assertEqual([passage.locator for passage in passages], ["paragraph-1", "paragraph-2"])
                self.assertEqual([passage.text for passage in passages], ["First paragraph.", "Second paragraph."])
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
