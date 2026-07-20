import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from polyresearch.models import (
    Claim,
    EvidenceLink,
    EvidencePassage,
    QueryRecord,
    ReportStatement,
    ResearchRun,
    SourceRecord,
    TranslationRecord,
    VerificationResult,
    VerificationStatus,
)
from polyresearch.provenance_graph import (
    build_provenance_graph,
    trace_report_statements_to_evidence,
)
from polyresearch.repositories import SqliteEvidenceRepository


class ProvenanceGraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_projects_all_required_ledger_artifacts_as_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteEvidenceRepository(Path(directory) / "research.db")
            run = ResearchRun(id=uuid4(), question="What changed?", output_language="en")
            source = SourceRecord(canonical_url="https://example.test/policy", title="Policy")
            passage = EvidencePassage(source_id=source.id, text="Policy changed.", locator="p-1")
            translation = TranslationRecord(
                passage_id=passage.id, translated_text="La política cambió.",
                target_language="es", source_original_text_hash=passage.original_text_hash,
            )
            claim = Claim(statement="Policy changed.", evidence_passage_ids=[passage.id], extraction_confidence=0.9)
            link = EvidenceLink(claim_id=claim.id, passage_id=passage.id, relationship="supports")
            verification = VerificationResult(
                claim_id=claim.id, status=VerificationStatus.SUPPORTED, confidence=0.9,
                rationale="Direct evidence.", evidence_link_ids=[link.id],
            )
            statement = ReportStatement(
                run_id=run.id, rendered_text="Policy changed.", claim_ids=[claim.id],
                citation_ids=[passage.id], verification_status=VerificationStatus.SUPPORTED,
            )
            second_statement = ReportStatement(
                run_id=run.id, rendered_text="The policy changed, according to the source.",
                claim_ids=[claim.id], citation_ids=[passage.id],
                verification_status=VerificationStatus.SUPPORTED,
            )
            query = QueryRecord(
                run_id=run.id, query="policy", language="en", provider="tavily",
                result_url=source.canonical_url,
            )
            try:
                await repository.create_run(run)
                await repository.append_query_records(run.id, [query])
                await repository.append_sources(run.id, [source])
                await repository.append_passages(run.id, [passage])
                await repository.append_translations(run.id, [translation])
                await repository.append_claims(run.id, [claim])
                await repository.append_evidence_links(run.id, [link])
                await repository.append_verification_results(run.id, [verification])
                await repository.append_report_statements(run.id, [statement, second_statement])

                graph = await build_provenance_graph(repository, run.id)

                self.assertEqual(graph.run_id, run.id)
                self.assertEqual(
                    {node.kind for node in graph.nodes},
                    {
                        "research_run", "query", "source", "passage", "translation", "claim",
                        "evidence_link", "verification_result", "report_statement",
                    },
                )
                self.assertEqual(len({node.node_id for node in graph.nodes}), len(graph.nodes))
                self.assertEqual(
                    {edge.kind for edge in graph.edges},
                    {
                        "FOUND_BY", "CONTAINS", "TRANSLATED_AS", "ASSERTS", "SUPPORTS",
                        "VERIFIED_BY", "RENDERED_AS",
                    },
                )
                found_by = next(edge for edge in graph.edges if edge.kind == "FOUND_BY")
                self.assertEqual(found_by.from_node_id, f"query:{query.id}")
                self.assertEqual(found_by.to_node_id, f"source:{source.id}")
                paths_by_statement = trace_report_statements_to_evidence(graph)
                self.assertEqual(set(paths_by_statement), {statement.id, second_statement.id})
                for statement_id, paths in paths_by_statement.items():
                    self.assertTrue(paths, f"Report statement {statement_id} has no evidence path")
                    self.assertEqual(paths[0].claim_id, claim.id)
                    self.assertEqual(paths[0].passage_id, passage.id)
            finally:
                repository.close()
