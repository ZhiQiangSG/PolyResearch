"""Report generation, QA, and rendering from typed evidence artifacts."""

import json
from typing import cast

from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig

from polyresearch.configuration import Configuration
from polyresearch.models import (
    AgentState, Claim, EvidencePassage, ReportBundle, ReportDraft, ReportQaIssue,
    ReportStatement, SourceRecord, VerificationResult, VerificationStatus,
)
from polyresearch.nodes.provenance import (
    load_evidence_ledger as _load_evidence_ledger,
    serialize_artifacts as _serialize_artifacts,
)
from polyresearch.prompts import final_report_generation_prompt
from polyresearch.evidence.report_qa import validate_report_statements
from polyresearch.runtime.model_utils import create_qwen_chat_model, get_model_token_limit, is_token_limit_exceeded
from polyresearch.runtime.text_utils import get_today_str

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.
    
    This function takes all collected research findings and synthesizes them into a 
    well-structured, comprehensive final report using the configured report generation model.
    
    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys
        
    Returns:
        Dictionary containing the final report and cleared state
    """
    # Step 1: Load report inputs from the durable typed evidence ledger.
    context, sources, passages, claims, _, verification_results = await _load_evidence_ledger(
        config
    )
    queries = await context.repository.list_query_records(context.run_id)
    findings = json.dumps(
        {
            "sources": _serialize_artifacts(sources, SourceRecord),
            "passages": _serialize_artifacts(passages, EvidencePassage),
            "claims": _serialize_artifacts(claims, Claim),
            "verification_results": _serialize_artifacts(
                verification_results, VerificationResult
            ),
        },
        ensure_ascii=False,
    )
    
    # Step 2: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    writer_model = create_qwen_chat_model(
        configurable,
        configurable.final_report_model,
        configurable.final_report_model_max_tokens,
        config,
    ).with_structured_output(ReportDraft).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    )
    
    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    
    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt with all research context
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str()
            )
            
            # Generate the final report
            report_draft = cast(ReportDraft, await writer_model.ainvoke([
                HumanMessage(content=final_report_prompt)
            ]))
            statements = _build_report_statements(
                run_id=context.run_id,
                report_draft=report_draft,
                claims=claims,
                verification_results=verification_results,
            )
            qa_issues = validate_report_statements(
                statements=statements,
                claims=claims,
                passages=passages,
                sources=sources,
                queries=queries,
            )
            if not statements:
                qa_issues.append(
                    ReportQaIssue(
                        code="no_report_statements",
                        severity="error",
                        message="Report draft produced no resolvable factual statements.",
                    )
                )
            qa_errors = [issue for issue in qa_issues if issue.severity == "error"]
            if qa_errors:
                return {
                    "final_report": "Report QA failed:\n"
                    + "\n".join(f"- {issue.message}" for issue in qa_errors),
                    "report_qa_issues": qa_errors,
                    "messages": [AIMessage(content="Report QA failed")],
                }
            markdown = _render_statement_markdown(
                title=report_draft.title,
                statements=statements,
                passages=passages,
                sources=sources,
                qa_issues=qa_issues,
            )
            await context.repository.append_report_statements(context.run_id, statements)
            bundle = ReportBundle(
                run_id=context.run_id,
                markdown=markdown,
                provenance_json={
                    "statement_ids": [str(statement.id) for statement in statements],
                    "claim_ids": [
                        str(claim_id)
                        for statement in statements
                        for claim_id in statement.claim_ids
                    ],
                },
                qa_issues=qa_issues,
                qa_passed=True,
            )
            await context.repository.append_report_bundles(context.run_id, [bundle])
            
            # Return successful report generation
            return {
                "final_report": markdown,
                "messages": [AIMessage(content=markdown)],
            }
            
        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length: {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                        }
                    # Use 4x token limit as character approximation for truncation
                    findings_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # Truncate findings and retry
                findings = findings[:findings_token_limit]
                continue
            else:
                # Non-token-limit error: return error immediately
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                }
    
    # Step 4: Return failure result if all retries exhausted
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
    }


def _build_report_statements(
    *,
    run_id,
    report_draft: ReportDraft,
    claims: list[Claim],
    verification_results: list[VerificationResult],
) -> list[ReportStatement]:
    """Resolve model-selected claim IDs into durable, cited report statements."""
    claims_by_id = {claim.id: claim for claim in claims}
    statuses_by_claim_id = {
        result.claim_id: result.status for result in verification_results
    }
    statements: list[ReportStatement] = []
    for draft in report_draft.statements:
        citation_ids = list(
            dict.fromkeys(
                passage_id
                for claim_id in draft.claim_ids
                if claim_id in claims_by_id
                for passage_id in claims_by_id[claim_id].evidence_passage_ids
            )
        )
        statuses = [statuses_by_claim_id.get(claim_id) for claim_id in draft.claim_ids]
        status = (
            statuses[0]
            if statuses and all(candidate == statuses[0] for candidate in statuses)
            and statuses[0] is not None
            else VerificationStatus.INSUFFICIENT_EVIDENCE
        )
        statements.append(
            ReportStatement(
                run_id=run_id,
                rendered_text=draft.rendered_text,
                claim_ids=draft.claim_ids,
                citation_ids=citation_ids,
                verification_status=status,
            )
        )
    return statements


def _render_statement_markdown(
    *,
    title: str,
    statements: list[ReportStatement],
    passages: list[EvidencePassage],
    sources: list[SourceRecord],
    qa_issues,
) -> str:
    """Render persisted statements and stable passage citations as Markdown."""
    passages_by_id = {passage.id: passage for passage in passages}
    sources_by_id = {source.id: source for source in sources}
    rendered_statements = []
    used_citation_ids = []
    for statement in statements:
        citations = " ".join(f"[P:{citation_id}]" for citation_id in statement.citation_ids)
        rendered_statements.append(f"{statement.rendered_text} {citations}".rstrip())
        used_citation_ids.extend(statement.citation_ids)

    lines = [f"# {title}", "", *rendered_statements]
    if used_citation_ids:
        lines.extend(["", "## Sources", ""])
        for citation_id in dict.fromkeys(used_citation_ids):
            passage = passages_by_id.get(citation_id)
            if not passage:
                continue
            source = sources_by_id.get(passage.source_id)
            source_label = source.title if source else "Unknown source"
            source_url = source.canonical_url if source else ""
            lines.append(
                f"- [P:{citation_id}] {source_label} — {source_url} ({passage.locator})"
            )
    warnings = [issue for issue in qa_issues if issue.severity == "warning"]
    if warnings:
        lines.extend(["", "## QA warnings", ""])
        lines.extend(f"- {warning.message}" for warning in warnings)
    return "\n".join(lines)

