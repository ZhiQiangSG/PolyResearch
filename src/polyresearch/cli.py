"""Command-line entry point for PolyResearch."""

import argparse
import asyncio
import json
import os
from uuid import UUID, uuid4

from langchain_core.messages import HumanMessage


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser without importing the graph eagerly."""
    parser = argparse.ArgumentParser(
        prog="polyresearch",
        description="Run a PolyResearch deep-research query.",
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Research question to investigate.",
    )
    parser.add_argument(
        "--inspect-ledger",
        metavar="RUN_ID",
        help="Print sources, passages, translations, discovery queries, and claims for a run.",
    )
    parser.add_argument(
        "--source-id",
        metavar="SOURCE_ID",
        help="Limit --inspect-ledger output to one source UUID.",
    )
    parser.add_argument(
        "--inspect-trace",
        metavar="RUN_ID",
        help="Print complete provenance traces and gaps for one report statement.",
    )
    parser.add_argument(
        "--report-statement-id",
        metavar="STATEMENT_ID",
        help="Required with --inspect-trace; identifies the report statement UUID.",
    )
    return parser


async def run_query(query: str) -> str:
    """Run one research query and return the generated report."""
    from polyresearch.graph import graph
    from polyresearch.repositories import SqliteEvidenceRepository

    repository = SqliteEvidenceRepository(
        os.environ.get("POLYRESEARCH_DB_PATH", "polyresearch.db")
    )
    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            {
                "configurable": {
                    "run_id": str(uuid4()),
                    "evidence_repository": repository,
                    "output_language": os.environ.get(
                        "POLYRESEARCH_OUTPUT_LANGUAGE", "en"
                    ),
                }
            },
        )
        return result.get("final_report", "")
    finally:
        repository.close()


async def build_ledger_inspection(
    repository, run_id: UUID, source_id: UUID | None = None
) -> dict:
    """Build a read-only source-to-claim view from the typed evidence ledger."""
    sources, passages, translations, queries, claims = await asyncio.gather(
        repository.list_sources(run_id),
        repository.list_passages(run_id),
        repository.list_translations(run_id),
        repository.list_query_records(run_id),
        repository.list_claims(run_id),
    )
    if source_id is not None:
        sources = [source for source in sources if source.id == source_id]
        if not sources:
            raise ValueError(f"Source {source_id} does not exist in run {run_id}")

    inspected_sources = []
    for source in sources:
        source_passages = [passage for passage in passages if passage.source_id == source.id]
        passage_ids = {passage.id for passage in source_passages}
        source_translations = [
            translation for translation in translations if translation.passage_id in passage_ids
        ]
        downstream_claims = [
            claim for claim in claims if passage_ids.intersection(claim.evidence_passage_ids)
        ]
        associated_urls = {source.canonical_url, source.discovered_url}
        source_queries = [
            query
            for query in queries
            if query.result_url is None or query.result_url in associated_urls
        ]
        inspected_sources.append(
            {
                "source": source.model_dump(mode="json"),
                "passages": [passage.model_dump(mode="json") for passage in source_passages],
                "translations": [
                    translation.model_dump(mode="json") for translation in source_translations
                ],
                "queries": [query.model_dump(mode="json") for query in source_queries],
                "downstream_claims": [claim.model_dump(mode="json") for claim in downstream_claims],
            }
        )
    return {"run_id": str(run_id), "sources": inspected_sources}


async def inspect_ledger(run_id: str, source_id: str | None = None) -> str:
    """Open the configured local ledger and render a JSON inspection view."""
    from polyresearch.repositories import SqliteEvidenceRepository

    repository = SqliteEvidenceRepository(
        os.environ.get("POLYRESEARCH_DB_PATH", "polyresearch.db")
    )
    try:
        inspection = await build_ledger_inspection(
            repository,
            UUID(run_id),
            UUID(source_id) if source_id else None,
        )
        return json.dumps(inspection, ensure_ascii=False, indent=2)
    finally:
        repository.close()


async def build_report_trace_inspection(
    repository, run_id: UUID, report_statement_id: UUID
) -> dict:
    """Build a read-only complete-trace and diagnostic view for one statement."""
    from polyresearch.evidence.provenance_graph import (
        build_provenance_graph,
        diagnose_incomplete_report_provenance,
        trace_report_statements_to_discovery,
    )

    graph = await build_provenance_graph(repository, run_id)
    statement_node = next(
        (
            node
            for node in graph.nodes
            if node.kind == "report_statement" and node.artifact_id == report_statement_id
        ),
        None,
    )
    if statement_node is None:
        raise ValueError(f"Report statement {report_statement_id} does not exist in run {run_id}")
    traces = trace_report_statements_to_discovery(graph)[report_statement_id]
    diagnostics = diagnose_incomplete_report_provenance(graph)[report_statement_id]
    return {
        "run_id": str(run_id),
        "report_statement": statement_node.attributes,
        "traces": [trace.model_dump(mode="json") for trace in traces],
        "diagnostics": [diagnostic.model_dump(mode="json") for diagnostic in diagnostics],
        "complete": bool(traces) and not diagnostics,
    }


async def inspect_report_trace(run_id: str, report_statement_id: str) -> str:
    """Open the configured SQLite ledger and render a statement trace as JSON."""
    from polyresearch.repositories import SqliteEvidenceRepository

    repository = SqliteEvidenceRepository(
        os.environ.get("POLYRESEARCH_DB_PATH", "polyresearch.db")
    )
    try:
        inspection = await build_report_trace_inspection(
            repository, UUID(run_id), UUID(report_statement_id)
        )
        return json.dumps(inspection, ensure_ascii=False, indent=2)
    finally:
        repository.close()


def main() -> None:
    """Run the CLI, displaying help when no research question is provided."""
    parser = build_parser()
    args = parser.parse_args()
    if args.inspect_trace:
        if not args.report_statement_id:
            parser.error("--inspect-trace requires --report-statement-id")
        try:
            print(asyncio.run(inspect_report_trace(args.inspect_trace, args.report_statement_id)))
        except (ValueError, LookupError) as error:
            parser.error(str(error))
        return
    if args.inspect_ledger:
        try:
            print(asyncio.run(inspect_ledger(args.inspect_ledger, args.source_id)))
        except (ValueError, LookupError) as error:
            parser.error(str(error))
        return
    query = " ".join(args.query).strip()
    if not query:
        parser.print_help()
        return

    report = asyncio.run(run_query(query))
    print(report)
