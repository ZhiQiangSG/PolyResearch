"""Backward-compatible public entry point for the assembled workflow."""

from polyresearch.workflows.orchestrator import (
    clarify_with_user,
    graph,
    initialize_research_run,
    language_gap_analysis,
    multilingual_planner,
    write_research_brief,
)
from polyresearch.workflows.report_generator import (
    _build_report_statements,
    _render_statement_markdown,
    final_report_generation,
)
from polyresearch.workflows.researcher import (
    execute_tool_safely,
    extract_claims,
    researcher,
    researcher_subgraph,
    researcher_tools,
    resolve_conflicts,
    translate_claim_evidence,
    unknown_tool_observation,
    verify_claim_clusters,
    verify_claims,
)
from polyresearch.workflows.supervisor import supervisor, supervisor_subgraph, supervisor_tools

__all__ = [
    "graph", "initialize_research_run", "clarify_with_user", "write_research_brief",
    "multilingual_planner", "language_gap_analysis", "supervisor", "supervisor_tools",
    "supervisor_subgraph", "researcher", "researcher_tools", "researcher_subgraph",
    "extract_claims", "translate_claim_evidence", "verify_claim_clusters", "verify_claims", "resolve_conflicts",
    "execute_tool_safely", "unknown_tool_observation", "final_report_generation",
    "_build_report_statements", "_render_statement_markdown",
]
