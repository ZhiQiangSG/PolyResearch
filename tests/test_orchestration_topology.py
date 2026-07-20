"""Regression tests for the explicit evidence-first top-level graph stages."""

from polyresearch.workflows.orchestrator import build_graph


def test_top_level_graph_exposes_evidence_first_stage_boundaries() -> None:
    graph = build_graph()
    node_names = set(graph.get_graph().nodes)

    assert {
        "initialize_research_run",
        "clarify_with_user",
        "write_research_brief",
        "multilingual_planner",
        "provider_routed_discovery",
        "fetch_extract",
        "evidence_ledger",
        "claim_extraction",
        "verification_conflict_loop",
        "report_composition",
        "report_qa",
    }.issubset(node_names)
