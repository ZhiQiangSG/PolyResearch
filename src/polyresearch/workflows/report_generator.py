"""Report generation, QA, and rendering from typed evidence artifacts."""

import json
from collections import defaultdict
from typing import cast
from uuid import UUID

from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig

from polyresearch.configuration import Configuration
from polyresearch.models import (
    AgentState, Claim, DisagreementAssessment, DisagreementDimension,
    EvidencePassage, ReportBundle, ReportDraft, ReportQaIssue, ReportStatement,
    SourceRecord, UnresolvedDisagreement, VerificationResult, VerificationStatus,
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
    unresolved_disagreements = _build_unresolved_disagreements(
        claims=claims, verification_results=verification_results
    )
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
                unresolved_disagreements=unresolved_disagreements,
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
                    "unresolved_disagreement_cluster_ids": [
                        str(disagreement.cluster_id)
                        for disagreement in unresolved_disagreements
                    ],
                },
                unresolved_disagreements=unresolved_disagreements,
                qa_issues=qa_issues,
                qa_passed=True,
            )
            await context.repository.append_report_bundles(context.run_id, [bundle])
            
            # Return successful report generation
            return {
                "final_report": markdown,
                "unresolved_disagreements": unresolved_disagreements,
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
        claim_id: result.status
        for claim_id, result in _latest_results_by_claim_id(verification_results).items()
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


def _build_unresolved_disagreements(
    *,
    claims: list[Claim],
    verification_results: list[VerificationResult],
) -> list[UnresolvedDisagreement]:
    """Promote unresolved verified disagreements into durable report outputs."""
    claims_by_cluster: dict[UUID, list[Claim]] = defaultdict(list)
    for claim in claims:
        claims_by_cluster[claim.claim_cluster_id or claim.id].append(claim)
    results_by_claim_id = _latest_results_by_claim_id(verification_results)
    unresolved: list[UnresolvedDisagreement] = []
    unresolved_statuses = {
        VerificationStatus.CONTRADICTED,
        VerificationStatus.NOT_COMPARABLE,
    }

    for cluster_id, cluster_claims in claims_by_cluster.items():
        results = [
            result for claim in cluster_claims
            if (result := results_by_claim_id.get(claim.id)) is not None
        ]
        assessments = _unique_assessments(results)
        has_genuine_conflict = any(
            assessment.dimension == DisagreementDimension.GENUINE_CONFLICT
            and assessment.present
            for assessment in assessments
        )
        has_conflicting_status = any(result.status in unresolved_statuses for result in results)
        # Insufficient evidence is a limitation, but is not itself a disagreement.
        if not has_genuine_conflict and not has_conflicting_status:
            continue

        causes = [assessment for assessment in assessments if assessment.present]
        why_it_may_conflict = [assessment.explanation for assessment in causes]
        if not why_it_may_conflict:
            why_it_may_conflict = [
                "The available evidence reaches incompatible conclusions without "
                "enough shared context to determine the cause."
            ]
        unresolved.append(
            UnresolvedDisagreement(
                cluster_id=cluster_id,
                claim_ids=[claim.id for claim in cluster_claims],
                conflicting_claims=[claim.statement for claim in cluster_claims],
                verification_statuses={
                    str(result.claim_id): result.status for result in results
                },
                disagreement_assessments=causes,
                why_it_may_conflict=why_it_may_conflict,
                evidence_needed=_resolution_evidence_needed(causes),
            )
        )
    return unresolved


def _latest_results_by_claim_id(
    results: list[VerificationResult],
) -> dict[UUID, VerificationResult]:
    """Use only the newest immutable verification attempt in report output."""
    latest: dict[UUID, VerificationResult] = {}
    for result in results:
        current = latest.get(result.claim_id)
        if current is None or (result.attempt_number, result.created_at, str(result.id)) > (
            current.attempt_number,
            current.created_at,
            str(current.id),
        ):
            latest[result.claim_id] = result
    return latest


def _unique_assessments(
    results: list[VerificationResult],
) -> list[DisagreementAssessment]:
    """Deduplicate equivalent verifier assessments emitted for cluster members."""
    assessments: dict[tuple[DisagreementDimension, bool, str], DisagreementAssessment] = {}
    for result in results:
        for assessment in result.disagreement_assessments:
            assessments[(assessment.dimension, assessment.present, assessment.explanation)] = assessment
    return list(assessments.values())


def _resolution_evidence_needed(
    causes: list[DisagreementAssessment],
) -> list[str]:
    """Name the evidence that would make each identified disagreement comparable."""
    requirements = {
        DisagreementDimension.TIME_PERIOD: (
            "Primary or official records covering the same time period for each claim."
        ),
        DisagreementDimension.GEOGRAPHIC_SCOPE: (
            "Primary or official records with explicitly matched geographic scope."
        ),
        DisagreementDimension.DEFINITION_OR_METHOD: (
            "Methodology notes or official definitions that make the measurements comparable."
        ),
        DisagreementDimension.POPULATION_OR_SAMPLE: (
            "Primary data with a matched population or sampling frame."
        ),
        DisagreementDimension.TRANSLATION_AMBIGUITY: (
            "Original-language passages and independently reviewed translations."
        ),
        DisagreementDimension.GENUINE_CONFLICT: (
            "Additional independent primary or official records that directly address the conflicting proposition."
        ),
    }
    needed = list(dict.fromkeys(requirements[cause.dimension] for cause in causes))
    return needed or [
        "Primary or official evidence that directly addresses the claim within the same scope."
    ]


def _render_statement_markdown(
    *,
    title: str,
    statements: list[ReportStatement],
    passages: list[EvidencePassage],
    sources: list[SourceRecord],
    qa_issues,
    unresolved_disagreements: list[UnresolvedDisagreement],
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
    if unresolved_disagreements:
        lines.extend(["", "## Unresolved disagreements", ""])
        for disagreement in unresolved_disagreements:
            lines.extend([f"### Claim cluster {disagreement.cluster_id}", "", "**What conflicts:**"])
            for claim_id, statement in zip(
                disagreement.claim_ids, disagreement.conflicting_claims, strict=True
            ):
                status = disagreement.verification_statuses.get(str(claim_id), "unverified")
                lines.append(f"- {statement} ({status})")
            lines.extend(["", "**Why it may conflict:**"])
            lines.extend(f"- {reason}" for reason in disagreement.why_it_may_conflict)
            lines.extend(["", "**Evidence needed to resolve it:**"])
            lines.extend(f"- {evidence}" for evidence in disagreement.evidence_needed)
            lines.append("")
    warnings = [issue for issue in qa_issues if issue.severity == "warning"]
    if warnings:
        lines.extend(["", "## QA warnings", ""])
        lines.extend(f"- {warning.message}" for warning in warnings)
    return "\n".join(lines)
