import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from polyresearch.models import (
    Claim,
    EvidenceLink,
    EvidencePassage,
    QueryRecord,
    ProvenanceGraph,
    ProvenanceGraphEdge,
    ProvenanceGraphNode,
    ReportEvidenceTrace,
    ReportStatement,
    ResearchRun,
    SourceRecord,
    TranslationRecord,
    VerificationResult,
    VerificationStatus,
)
from polyresearch.provenance_graph import (
    build_provenance_graph,
    diagnose_incomplete_report_provenance,
    trace_report_statements_to_discovery,
    trace_report_statements_to_evidence,
)
from polyresearch.repositories import ReportProvenanceError, SqliteEvidenceRepository


class ProvenanceGraphTests(unittest.IsolatedAsyncioTestCase):
    def test_full_trace_contract_keeps_translation_optional(self) -> None:
        trace = ReportEvidenceTrace(
            report_statement_id=uuid4(), claim_id=uuid4(), evidence_passage_id=uuid4(),
            source_id=uuid4(), query_id=uuid4(),
        )

        self.assertIsNone(trace.translation_id)

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
                traces_by_statement = trace_report_statements_to_discovery(graph)
                self.assertEqual(set(traces_by_statement), {statement.id, second_statement.id})
                for statement_id, traces in traces_by_statement.items():
                    self.assertTrue(traces, f"Report statement {statement_id} has no discovery trace")
                    trace = traces[0]
                    self.assertEqual(trace.claim_id, claim.id)
                    self.assertEqual(trace.evidence_passage_id, passage.id)
                    self.assertEqual(trace.source_id, source.id)
                    self.assertEqual(trace.query_id, query.id)
                    self.assertEqual(trace.translation_id, translation.id)
            finally:
                repository.close()

    async def test_reverse_traversal_returns_every_valid_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteEvidenceRepository(Path(directory) / "research.db")
            run = ResearchRun(id=uuid4(), question="What changed?", output_language="en")
            source = SourceRecord(canonical_url="https://example.test/policy", title="Policy")
            first_passage = EvidencePassage(source_id=source.id, text="Policy began Monday.", locator="p-1")
            second_passage = EvidencePassage(source_id=source.id, text="The policy is national.", locator="p-2")
            translations = [
                TranslationRecord(
                    passage_id=first_passage.id, translated_text="La política comenzó el lunes.",
                    target_language="es", source_original_text_hash=first_passage.original_text_hash,
                ),
                TranslationRecord(
                    passage_id=first_passage.id, translated_text="该政策于周一开始。",
                    target_language="zh", source_original_text_hash=first_passage.original_text_hash,
                ),
            ]
            first_claim = Claim(
                statement="Policy began Monday.",
                evidence_passage_ids=[first_passage.id, second_passage.id], extraction_confidence=0.9,
            )
            second_claim = Claim(
                statement="Policy began Monday nationally.",
                evidence_passage_ids=[first_passage.id], extraction_confidence=0.8,
            )
            statement = ReportStatement(
                run_id=run.id, rendered_text="The policy began Monday nationally.",
                claim_ids=[first_claim.id, second_claim.id],
                citation_ids=[first_passage.id, second_passage.id],
                verification_status=VerificationStatus.SUPPORTED,
            )
            queries = [
                QueryRecord(run_id=run.id, query="policy Monday", language="en", provider="tavily", result_url=source.canonical_url),
                QueryRecord(run_id=run.id, query="national policy", language="en", provider="tavily", result_url=source.canonical_url),
            ]
            try:
                await repository.create_run(run)
                await repository.append_query_records(run.id, queries)
                await repository.append_sources(run.id, [source])
                await repository.append_passages(run.id, [first_passage, second_passage])
                await repository.append_translations(run.id, translations)
                await repository.append_claims(run.id, [first_claim, second_claim])
                await repository.append_report_statements(run.id, [statement])

                graph = await build_provenance_graph(repository, run.id)
                traces = trace_report_statements_to_discovery(graph)[statement.id]

                # c1→p1 has 2 queries × 2 translations; c1→p2 has 2 queries × no
                # translation; c2→p1 has 2 queries × 2 translations.
                self.assertEqual(len(traces), 10)
                self.assertEqual({trace.query_id for trace in traces}, {query.id for query in queries})
                self.assertEqual(
                    {trace.translation_id for trace in traces if trace.translation_id},
                    {translation.id for translation in translations},
                )
                self.assertTrue(any(trace.translation_id is None for trace in traces))
            finally:
                repository.close()

    async def test_complete_path_without_required_translation_is_not_a_gap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteEvidenceRepository(Path(directory) / "research.db")
            run = ResearchRun(id=uuid4(), question="What changed?", output_language="en")
            source = SourceRecord(canonical_url="https://example.test/policy", title="Policy")
            passage = EvidencePassage(
                source_id=source.id, text="The policy changed.", locator="p-1", original_language="en"
            )
            claim = Claim(statement="The policy changed.", evidence_passage_ids=[passage.id], extraction_confidence=0.9)
            statement = ReportStatement(
                run_id=run.id, rendered_text="The policy changed.", claim_ids=[claim.id],
                citation_ids=[passage.id], verification_status=VerificationStatus.SUPPORTED,
            )
            query = QueryRecord(
                run_id=run.id, query="policy changed", language="en", provider="tavily",
                result_url=source.canonical_url,
            )
            try:
                await repository.create_run(run)
                await repository.append_query_records(run.id, [query])
                await repository.append_sources(run.id, [source])
                await repository.append_passages(run.id, [passage])
                await repository.append_claims(run.id, [claim])
                await repository.append_report_statements(run.id, [statement])

                graph = await build_provenance_graph(repository, run.id)
                traces = trace_report_statements_to_discovery(graph)[statement.id]
                diagnostics = diagnose_incomplete_report_provenance(graph)[statement.id]

                self.assertEqual(len(traces), 1)
                self.assertIsNone(traces[0].translation_id)
                self.assertEqual(diagnostics, [])
            finally:
                repository.close()

    async def test_persistence_gate_rejects_missing_query_linkage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = SqliteEvidenceRepository(Path(directory) / "research.db")
            run = ResearchRun(id=uuid4(), question="What changed?", output_language="en")
            source = SourceRecord(canonical_url="https://example.test/policy", title="Policy")
            passage = EvidencePassage(
                source_id=source.id, text="政策已变更。", locator="p-1", original_language="zh"
            )
            claim = Claim(statement="The policy changed.", evidence_passage_ids=[passage.id], extraction_confidence=0.8)
            statement = ReportStatement(
                run_id=run.id, rendered_text="The policy changed.", claim_ids=[claim.id],
                citation_ids=[passage.id], verification_status=VerificationStatus.SUPPORTED,
            )
            try:
                await repository.create_run(run)
                await repository.append_sources(run.id, [source])
                await repository.append_passages(run.id, [passage])
                await repository.append_claims(run.id, [claim])
                with self.assertRaisesRegex(ReportProvenanceError, "claim → original passage"):
                    await repository.append_report_statements(run.id, [statement])
                self.assertEqual(await repository.list_report_statements(run.id), [])
            finally:
                repository.close()

    def test_diagnostics_flag_missing_source_edge(self) -> None:
        statement_id, claim_id, passage_id = uuid4(), uuid4(), uuid4()
        graph = ProvenanceGraph(
            run_id=uuid4(),
            nodes=[
                ProvenanceGraphNode(node_id=f"report_statement:{statement_id}", artifact_id=statement_id, kind="report_statement"),
                ProvenanceGraphNode(node_id=f"claim:{claim_id}", artifact_id=claim_id, kind="claim"),
                ProvenanceGraphNode(node_id=f"passage:{passage_id}", artifact_id=passage_id, kind="passage"),
            ],
            edges=[
                ProvenanceGraphEdge(
                    edge_id="rendered", from_node_id=f"claim:{claim_id}",
                    to_node_id=f"report_statement:{statement_id}", kind="RENDERED_AS",
                ),
                ProvenanceGraphEdge(
                    edge_id="asserts", from_node_id=f"passage:{passage_id}",
                    to_node_id=f"claim:{claim_id}", kind="ASSERTS",
                ),
            ],
        )

        diagnostics = diagnose_incomplete_report_provenance(graph)[statement_id]

        self.assertEqual([diagnostic.code for diagnostic in diagnostics], ["missing_source"])
