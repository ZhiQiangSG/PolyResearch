"""Command-line entry point for PolyResearch."""

import argparse
import asyncio
import os
from uuid import uuid4

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


def main() -> None:
    """Run the CLI, displaying help when no research question is provided."""
    parser = build_parser()
    args = parser.parse_args()
    query = " ".join(args.query).strip()
    if not query:
        parser.print_help()
        return

    report = asyncio.run(run_query(query))
    print(report)
