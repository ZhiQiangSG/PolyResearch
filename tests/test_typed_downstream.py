import importlib
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from polyresearch.models import (
    Claim,
    ClaimExtractionResult,
    EvidencePassage,
    ResearchRun,
    SourceRecord,
)
from polyresearch.repositories import SqliteEvidenceRepository

graph_module = importlib.import_module("polyresearch.graph")


class _ClaimExtractorStub:
    def __init__(self, claim: Claim) -> None:
        self.claim = claim
        self.messages = None

    def with_structured_output(self, schema):
        return self

    def with_retry(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        return ClaimExtractionResult(claims=[self.claim])


class TypedDownstreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_claim_extraction_reads_and_writes_the_durable_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteEvidenceRepository(Path(directory) / "research.db")
            run = ResearchRun(
                id=uuid4(), question="What changed?", output_language="en"
            )
            source = SourceRecord(
                canonical_url="https://example.test/policy", title="Policy update"
            )
            passage = EvidencePassage(
                source_id=source.id,
                text="The policy changed on 1 January.",
                locator="paragraph-1",
                original_language="en",
            )
            claim = Claim(
                statement="The policy changed on 1 January.",
                evidence_passage_ids=[passage.id],
                extraction_confidence=0.9,
            )
            extractor = _ClaimExtractorStub(claim)
            original_factory = graph_module.create_qwen_chat_model
            graph_module.create_qwen_chat_model = lambda *args, **kwargs: extractor
            try:
                await repository.create_run(run)
                await repository.append_sources(run.id, [source])
                await repository.append_passages(run.id, [passage])

                result = await graph_module.extract_claims(
                    {},
                    {
                        "configurable": {
                            "run_id": str(run.id),
                            "evidence_repository": repository,
                        }
                    },
                )

                self.assertEqual(result["claims"], [claim])
                self.assertEqual(await repository.list_claims(run.id), [claim])
                links = await repository.list_evidence_links(run.id)
                self.assertEqual(len(links), 1)
                self.assertEqual(links[0].claim_id, claim.id)
                self.assertEqual(links[0].passage_id, passage.id)
                self.assertEqual(len(extractor.messages), 2)
                self.assertIn("EvidenceLedger", extractor.messages[1].content)
                self.assertNotIn("ToolMessage", extractor.messages[1].content)
            finally:
                graph_module.create_qwen_chat_model = original_factory
                repository.close()
