"""Report generation, QA, and rendering from typed evidence artifacts."""

import hashlib
import json
import logging
import re
from html import escape
from datetime import datetime, timezone
from time import perf_counter
from collections import defaultdict
from typing import cast
from uuid import UUID

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from polyresearch.configuration import Configuration
from polyresearch.models import (
    AgentState, Claim, DisagreementAssessment, DisagreementDimension,
    EvidenceLink, EvidencePassage, QueryRecord, ReportBundle, ReportDraft, ReportOutline,
    ReportQaIssue, ReportStatement, ResearchPlan, ResearchRun, SourceRecord, TranslationRecord, UnresolvedDisagreement,
    TraceRecord, VerificationResult, VerificationStatus,
)
from polyresearch.nodes.provenance import (
    load_evidence_ledger as _load_evidence_ledger,
    serialize_artifacts as _serialize_artifacts,
)
from polyresearch.prompts import report_outline_generation_prompt, report_prose_generation_prompt
from polyresearch.evidence.report_qa import validate_report_statements
from polyresearch.runtime.model_utils import create_qwen_chat_model, get_model_token_limit, is_token_limit_exceeded
from polyresearch.security import redacted_exception_info

logger = logging.getLogger(__name__)


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
    context, sources, passages, claims, evidence_links, verification_results = await _load_evidence_ledger(
        config
    )
    evidence_fingerprint = _report_evidence_fingerprint(
        claims=claims,
        verification_results=verification_results,
        sources=sources,
    )
    existing_bundles = await context.repository.list_report_bundles(context.run_id)
    if existing_bundles:
        latest_bundle = max(existing_bundles, key=lambda item: (item.created_at, str(item.id)))
        if (
            latest_bundle.qa_passed
            and latest_bundle.markdown
            and latest_bundle.evidence_fingerprint == evidence_fingerprint
        ):
            return {
                "final_report": latest_bundle.markdown,
                "unresolved_disagreements": latest_bundle.unresolved_disagreements,
                "report_qa_issues": latest_bundle.qa_issues,
                "messages": [AIMessage(content=latest_bundle.markdown)],
            }
    queries = await context.repository.list_query_records(context.run_id)
    run = await context.repository.get_run(context.run_id)
    plans = await context.repository.list_research_plans(context.run_id)
    translations = await context.repository.list_translations(context.run_id)
    unresolved_disagreements = _build_unresolved_disagreements(
        claims=claims, verification_results=verification_results
    )
    approved_artifacts = json.dumps(
        {
            "claims": _serialize_artifacts(claims, Claim),
            "verification_results": _serialize_artifacts(
                verification_results, VerificationResult
            ),
        },
        ensure_ascii=False,
    )
    
    # Step 2: Qwen first selects a claim-bound outline, then writes only from it.
    configurable = Configuration.from_runnable_config(config)
    base_model = create_qwen_chat_model(
        configurable,
        configurable.final_report_model,
        configurable.final_report_model_max_tokens,
        config,
    )
    outline_model = base_model.with_structured_output(ReportOutline).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    )
    writer_model = base_model.with_structured_output(ReportDraft).with_retry(
        stop_after_attempt=configurable.max_structured_output_retries
    )
    
    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    artifacts_token_limit = None
    
    while current_retry <= max_retries:
        try:
            outline_prompt = report_outline_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                approved_artifacts=approved_artifacts,
            )
            outline = cast(ReportOutline, await outline_model.ainvoke([
                HumanMessage(content=outline_prompt)
            ]))
            # A legacy structured writer stub can return a draft during the outline
            # call. Production Qwen output is validated as ReportOutline.
            if isinstance(outline, ReportDraft):
                report_draft = outline
                outline_claim_ids = {
                    claim_id for statement in report_draft.statements for claim_id in statement.claim_ids
                }
            else:
                outline_claim_ids = {
                    claim_id for section in outline.sections for claim_id in section.claim_ids
                }
                outline_issues = _validate_outline_claim_ids(outline, claims)
                if outline_issues:
                    return _report_qa_failure(outline_issues)
                prose_prompt = report_prose_generation_prompt.format(
                    research_brief=state.get("research_brief", ""),
                    report_outline=outline.model_dump_json(),
                    approved_artifacts=approved_artifacts,
                    output_language=run.output_language,
                )
                report_draft = cast(ReportDraft, await writer_model.ainvoke([
                    HumanMessage(content=prose_prompt)
                ]))
            draft_issues = _validate_draft_claim_ids(report_draft, outline_claim_ids)
            if draft_issues:
                return _report_qa_failure(draft_issues)
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
                evidence_links=evidence_links,
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
            render_started_at = datetime.now(timezone.utc)
            render_started = perf_counter()
            markdown = _render_statement_markdown(
                title=report_draft.title,
                statements=statements,
                passages=passages,
                sources=sources,
                qa_issues=qa_issues,
                unresolved_disagreements=unresolved_disagreements,
            )
            html = _render_statement_html(
                title=report_draft.title,
                statements=statements,
                claims=claims,
                passages=passages,
                sources=sources,
                translations=translations,
                evidence_links=evidence_links,
                verification_results=verification_results,
            )
            render_latency_ms = (perf_counter() - render_started) * 1000
            await context.repository.append_report_statements(context.run_id, statements)
            await context.repository.append_trace_records(
                context.run_id,
                _build_report_trace_records(
                    run_id=context.run_id,
                    statements=statements,
                    claims=claims,
                    passages=passages,
                    sources=sources,
                    queries=queries,
                    started_at=render_started_at,
                    latency_ms=render_latency_ms,
                ),
            )
            bundle = ReportBundle(
                run_id=context.run_id,
                markdown=markdown,
                html=html,
                provenance_json=_build_markdown_provenance_bundle(
                    run=run,
                    research_plan=plans[-1] if plans else None,
                    statements=statements,
                    claims=claims,
                    passages=passages,
                    sources=sources,
                    translations=translations,
                    evidence_links=evidence_links,
                    verification_results=verification_results,
                    queries=queries,
                    unresolved_disagreements=unresolved_disagreements,
                ),
                unresolved_disagreements=unresolved_disagreements,
                qa_issues=qa_issues,
                qa_passed=True,
                evidence_fingerprint=evidence_fingerprint,
            )
            await context.repository.append_report_bundles(context.run_id, [bundle])
            
            # Return successful report generation
            return {
                "final_report": markdown,
                "unresolved_disagreements": unresolved_disagreements,
                "report_qa_issues": qa_issues,
                "messages": [AIMessage(content=markdown)],
            }
            
        except Exception as e:
            logger.warning(
                "Report generation attempt failed",
                extra={
                    "operation": "final_report_generation",
                    "run_id": str(context.run_id),
                    "attempt": current_retry + 1,
                },
                exc_info=redacted_exception_info(e),
            )
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
                    artifacts_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    artifacts_token_limit = int(artifacts_token_limit * 0.9)
                
                # Truncate only the artifact ledger; raw source content never enters
                # either report-generation stage.
                approved_artifacts = approved_artifacts[:artifacts_token_limit]
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


def _report_evidence_fingerprint(
    *,
    claims: list[Claim],
    verification_results: list[VerificationResult],
    sources: list[SourceRecord],
) -> str:
    """Return a deterministic identity for the report's mutable input ledger.

    Report bundles are immutable output snapshots.  IDs are intentionally part of
    the digest so newly appended evidence, including a later verification pass,
    invalidates a previously QA-passed report instead of reusing stale prose.
    """
    payload = {
        "claims": sorted(
            (item.model_dump(mode="json") for item in claims), key=lambda item: item["id"]
        ),
        "verification_results": sorted(
            (item.model_dump(mode="json") for item in verification_results),
            key=lambda item: item["id"],
        ),
        "sources": sorted(
            (item.model_dump(mode="json") for item in sources), key=lambda item: item["id"]
        ),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_outline_claim_ids(
    outline: ReportOutline, claims: list[Claim]
) -> list[ReportQaIssue]:
    """Reject an outline that selects missing claims before prose generation."""
    known_claim_ids = {claim.id for claim in claims}
    selected_claim_ids = {
        claim_id for section in outline.sections for claim_id in section.claim_ids
    }
    if selected_claim_ids and selected_claim_ids.issubset(known_claim_ids):
        return []
    if not selected_claim_ids:
        message = "Report outline did not select any approved claim IDs."
    else:
        message = "Report outline references claim IDs that are absent from the approved ledger."
    return [ReportQaIssue(code="invalid_report_outline", severity="error", message=message)]


def _validate_draft_claim_ids(
    report_draft: ReportDraft, outline_claim_ids: set[UUID]
) -> list[ReportQaIssue]:
    """Ensure prose cannot expand beyond the claims selected in the outline."""
    draft_claim_ids = {
        claim_id for statement in report_draft.statements for claim_id in statement.claim_ids
    }
    if draft_claim_ids.issubset(outline_claim_ids):
        return []
    return [
        ReportQaIssue(
            code="prose_uses_unapproved_claim",
            severity="error",
            message="Report prose references claims not approved by the report outline.",
        )
    ]


def _report_qa_failure(issues: list[ReportQaIssue]) -> dict:
    """Return a consistent blocking response for either report stage."""
    return {
        "final_report": "Report QA failed:\n" + "\n".join(
            f"- {issue.message}" for issue in issues
        ),
        "report_qa_issues": issues,
        "messages": [AIMessage(content="Report QA failed")],
    }


def _build_report_statements(
    *,
    run_id,
    report_draft: ReportDraft,
    claims: list[Claim],
    verification_results: list[VerificationResult],
) -> list[ReportStatement]:
    """Bind every sentence/displayable clause to claims before report rendering."""
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
        for clause in _split_displayable_clauses(draft.rendered_text):
            statements.append(
                ReportStatement(
                    run_id=run_id,
                    rendered_text=clause,
                    claim_ids=draft.claim_ids,
                    citation_ids=citation_ids,
                    verification_status=status,
                )
            )
    return statements


def _split_displayable_clauses(text: str) -> list[str]:
    """Create independently auditable records for sentences and semicolon clauses."""
    sentences = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    return [
        clause.strip()
        for sentence in sentences
        for clause in re.split(r"(?<=;)\s+", sentence)
        if clause.strip()
    ]


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
    supported, uncertain = _partition_report_statements(statements)
    lines = [f"# {title}", "", "## Findings supported by evidence", ""]
    lines.extend(_render_markdown_statements(supported) or ["- No claim reached supported status."])
    lines.extend(["", "## Conflicting or uncertain claims", ""])
    lines.extend(_render_markdown_statements(uncertain, include_status=True) or [
        "- No conflicting or uncertain claim statements were rendered."
    ])
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

    language_counts, source_mix, retrieval_dates = _report_metadata(sources)
    lines.extend(["", "## Language coverage and source mix", ""])
    lines.append("- Evidence languages: " + _format_counts(language_counts))
    lines.append("- Source types: " + _format_counts(source_mix))
    lines.extend(["", "## Method, retrieval date, and limitations", ""])
    lines.extend([
        "- Method: Qwen selected a claim-bound outline, then wrote prose only from approved claim and verification artifacts.",
        "- Retrieval date(s): " + (", ".join(retrieval_dates) if retrieval_dates else "not recorded"),
    ])
    limitations = _report_limitations(sources, uncertain, qa_issues)
    lines.extend(f"- Limitation: {limitation}" for limitation in limitations)

    lines.extend(["", "## Complete sources", ""])
    for source in sources:
        source_passages = [
            f"P:{passage.id}" for passage in passages_by_id.values()
            if passage.source_id == source.id
        ]
        source_language = source.content_language or source.language or "unknown"
        lines.append(
            f"- {source.title} — {source.canonical_url} "
            f"(publisher: {source.publisher or 'unknown'}; language: {source_language}; "
            f"citations: {', '.join(source_passages) or 'none'})"
        )
    if not sources:
        lines.append("- No sources were retained in the evidence ledger.")
    lines.extend([
        "", "## Citation provenance", "",
        "The stable `P:<passage-id>` citations in this Markdown report resolve "
        "through the companion JSON provenance bundle.",
    ])
    warnings = [issue for issue in qa_issues if issue.severity == "warning"]
    if warnings:
        lines.extend(["", "## QA warnings", ""])
        lines.extend(f"- {warning.message}" for warning in warnings)
    return "\n".join(lines)


def _partition_report_statements(
    statements: list[ReportStatement],
) -> tuple[list[ReportStatement], list[ReportStatement]]:
    """Keep qualified/conflicting claims out of the supported-findings section."""
    return (
        [item for item in statements if item.verification_status is VerificationStatus.SUPPORTED],
        [item for item in statements if item.verification_status is not VerificationStatus.SUPPORTED],
    )


def _render_markdown_statements(
    statements: list[ReportStatement], *, include_status: bool = False
) -> list[str]:
    lines = []
    for statement in statements:
        citations = " ".join(f"[P:{citation_id}]" for citation_id in statement.citation_ids)
        status = f" ({statement.verification_status.value})" if include_status else ""
        lines.append(f"- {statement.rendered_text}{status} {citations}".rstrip())
    return lines


def _report_metadata(sources: list[SourceRecord]) -> tuple[dict[str, int], dict[str, int], list[str]]:
    language_counts: dict[str, int] = defaultdict(int)
    source_mix: dict[str, int] = defaultdict(int)
    retrieval_dates: set[str] = set()
    for source in sources:
        language_counts[source.content_language or source.language or "unknown"] += 1
        source_mix[source.source_type] += 1
        retrieval_dates.add(source.retrieved_at.date().isoformat())
    return dict(language_counts), dict(source_mix), sorted(retrieval_dates)


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}: {value}" for key, value in sorted(counts.items())) or "none"


def _report_limitations(
    sources: list[SourceRecord], uncertain: list[ReportStatement], qa_issues: list[ReportQaIssue],
) -> list[str]:
    limitations = []
    if not sources:
        limitations.append("No retained sources are available for review.")
    if uncertain:
        limitations.append("Some claims remain conflicting, incomplete, outdated, or not comparable.")
    if not any(source.content_language or source.language for source in sources):
        limitations.append("Source-language metadata is incomplete.")
    limitations.extend(issue.message for issue in qa_issues if issue.severity == "warning")
    return limitations or ["No additional report-level limitations were recorded."]


def _build_markdown_provenance_bundle(
    *,
    run: ResearchRun,
    research_plan: ResearchPlan | None,
    statements: list[ReportStatement],
    claims: list[Claim],
    passages: list[EvidencePassage],
    sources: list[SourceRecord],
    translations: list[TranslationRecord],
    evidence_links: list[EvidenceLink],
    verification_results: list[VerificationResult],
    queries: list[QueryRecord],
    unresolved_disagreements: list[UnresolvedDisagreement],
) -> dict:
    """Create the durable JSON companion required to inspect Markdown citations."""
    cited_passage_ids = {
        passage_id for statement in statements for passage_id in statement.citation_ids
    }
    cited_claim_ids = {
        claim_id for statement in statements for claim_id in statement.claim_ids
    }
    cited_source_ids = {
        passage.source_id for passage in passages if passage.id in cited_passage_ids
    }
    return {
        "format": "polyresearch-markdown-provenance-v1",
        "run_id": str(run.id),
        "run_configuration": run.model_dump(mode="json"),
        "selected_language_configuration": research_plan.metadata.get("run_configuration", {}) if research_plan else {},
        "retrieval_timestamps": {"queries": [query.executed_at.isoformat() for query in queries], "sources": [source.retrieved_at.isoformat() for source in sources if source.id in cited_source_ids]},
        "statements": [statement.model_dump(mode="json") for statement in statements],
        "citations": {f"P:{passage.id}": passage.model_dump(mode="json") for passage in passages if passage.id in cited_passage_ids},
        "sources": [source.model_dump(mode="json") for source in sources if source.id in cited_source_ids],
        "translations": [translation.model_dump(mode="json") for translation in translations if translation.passage_id in cited_passage_ids],
        "claims": [claim.model_dump(mode="json") for claim in claims if claim.id in cited_claim_ids],
        "evidence_links": [link.model_dump(mode="json") for link in evidence_links if link.claim_id in cited_claim_ids],
        "verification_history": [result.model_dump(mode="json") for result in verification_results if result.claim_id in cited_claim_ids],
        "unresolved_disagreements": [item.model_dump(mode="json") for item in unresolved_disagreements],
    }


def _build_report_trace_records(
    *,
    run_id: UUID,
    statements: list[ReportStatement],
    claims: list[Claim],
    passages: list[EvidencePassage],
    sources: list[SourceRecord],
    queries: list[QueryRecord],
    started_at: datetime,
    latency_ms: float,
) -> list[TraceRecord]:
    """Link each rendered statement to its query/source/claim graph artifacts."""
    passages_by_id = {passage.id: passage for passage in passages}
    sources_by_id = {source.id: source for source in sources}
    traces = []
    for statement in statements:
        source_urls = {
            sources_by_id[passages_by_id[passage_id].source_id].canonical_url
            for passage_id in statement.citation_ids
            if passage_id in passages_by_id
            and passages_by_id[passage_id].source_id in sources_by_id
        }
        statement_queries = [
            query for query in queries if query.result_url in source_urls
        ]
        graph_ids = [f"report_statement:{statement.id}"]
        graph_ids.extend(f"claim:{claim_id}" for claim_id in statement.claim_ids)
        graph_ids.extend(f"passage:{passage_id}" for passage_id in statement.citation_ids)
        graph_ids.extend(f"query:{query.id}" for query in statement_queries)
        traces.append(
            TraceRecord(
                run_id=run_id,
                operation="report_render",
                query_ids=[query.id for query in statement_queries],
                report_statement_ids=[statement.id],
                graph_artifact_ids=graph_ids,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                latency_ms=latency_ms,
                cost_note="Report-model cost is not supplied by the configured Qwen transport.",
                provider_failure=next(
                    (query.failure for query in statement_queries if query.failure), None
                ),
            )
        )
    return traces


def _render_statement_html(
    *,
    title: str,
    statements: list[ReportStatement],
    claims: list[Claim],
    passages: list[EvidencePassage],
    sources: list[SourceRecord],
    translations: list[TranslationRecord],
    evidence_links: list[EvidenceLink],
    verification_results: list[VerificationResult],
) -> str:
    """Render claim-bound statements with a safe, inspectable evidence panel."""
    evidence_by_statement = _build_statement_evidence_panels(
        statements=statements,
        claims=claims,
        passages=passages,
        sources=sources,
        translations=translations,
        evidence_links=evidence_links,
        verification_results=verification_results,
    )
    supported, uncertain = _partition_report_statements(statements)
    language_counts, source_mix, retrieval_dates = _report_metadata(sources)
    statement_html = "\n".join(
        "<p><a class=\"report-statement\" href=\"#evidence-panel\" "
        f"data-evidence-id=\"{statement.id}\" aria-controls=\"evidence-panel\">"
        f"{escape(statement.rendered_text)} <span class=\"citation-anchor\">"
        f"[evidence]</span></a></p>"
        for statement in supported
    )
    uncertain_html = "\n".join(
        "<p><a class=\"report-statement\" href=\"#evidence-panel\" "
        f"data-evidence-id=\"{statement.id}\" aria-controls=\"evidence-panel\">"
        f"{escape(statement.rendered_text)} <span class=\"citation-anchor\">"
        f"[{statement.verification_status.value}; evidence]</span></a></p>"
        for statement in uncertain
    )
    source_html = "".join(
        f"<li>{escape(source.title)} — {escape(source.canonical_url)} "
        f"(publisher: {escape(source.publisher or 'unknown')}; language: "
        f"{escape(source.content_language or source.language or 'unknown')})</li>"
        for source in sources
    ) or "<li>No sources were retained in the evidence ledger.</li>"
    limitations = _report_limitations(sources, uncertain, [])
    evidence_json = json.dumps(evidence_by_statement, ensure_ascii=False).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{escape(title)}</title>
<style>
body {{ font-family: system-ui, sans-serif; line-height: 1.5; margin: 2rem; max-width: 72rem; }}
.report-statement {{ color: inherit; text-decoration: none; border-bottom: 1px dotted #2563eb; cursor: pointer; }}
.citation-anchor {{ color: #2563eb; font-size: .85em; }}
#evidence-panel {{ border: 1px solid #cbd5e1; border-radius: .5rem; padding: 1rem; margin-top: 2rem; }}
#evidence-panel h3 {{ margin-bottom: .25rem; }} #evidence-panel pre {{ white-space: pre-wrap; }}
.evidence-item {{ border-top: 1px solid #e2e8f0; margin-top: 1rem; padding-top: 1rem; }}
</style></head><body>
<main><h1>{escape(title)}</h1>
<section><h2>Findings supported by evidence</h2>{statement_html or '<p>No claim reached supported status.</p>'}</section>
<section><h2>Conflicting or uncertain claims</h2>{uncertain_html or '<p>No conflicting or uncertain claim statements were rendered.</p>'}</section>
<section><h2>Language coverage and source mix</h2><ul><li>Evidence languages: {escape(_format_counts(language_counts))}</li><li>Source types: {escape(_format_counts(source_mix))}</li></ul></section>
<section><h2>Method, retrieval date, and limitations</h2><ul><li>Method: Qwen selected a claim-bound outline, then wrote prose only from approved claim and verification artifacts.</li><li>Retrieval date(s): {escape(', '.join(retrieval_dates) or 'not recorded')}</li>{''.join(f'<li>Limitation: {escape(item)}</li>' for item in limitations)}</ul></section>
<section><h2>Complete sources</h2><ul>{source_html}</ul></section></main>
<aside id="evidence-panel" aria-live="polite"><p>Select a cited statement to inspect its evidence.</p></aside>
<script id="statement-evidence" type="application/json">{evidence_json}</script>
<script>
(() => {{
  const records = JSON.parse(document.getElementById('statement-evidence').textContent);
  const panel = document.getElementById('evidence-panel');
  const add = (parent, tag, text) => {{ const node = document.createElement(tag); node.textContent = text; parent.append(node); return node; }};
  const list = (parent, label, values) => {{
    if (!values.length) return;
    add(parent, 'h4', label); const ul = document.createElement('ul');
    values.forEach(value => add(ul, 'li', value)); parent.append(ul);
  }};
  document.querySelectorAll('.report-statement').forEach(anchor => anchor.addEventListener('click', event => {{
    event.preventDefault(); const record = records[anchor.dataset.evidenceId]; if (!record) return;
    panel.replaceChildren(); add(panel, 'h2', 'Evidence for this statement');
    add(panel, 'p', record.rendered_text); add(panel, 'p', `Verification status: ${{record.verification_status}}`);
    record.verification_history.forEach(item => {{
      const section = document.createElement('section'); section.className = 'evidence-item';
      add(section, 'h3', `Verification attempt ${{item.attempt_number}}: ${{item.status}}`);
      add(section, 'p', `Claim confidence: ${{item.confidence}}`);
      add(section, 'p', item.rationale); add(section, 'p', `Model: ${{item.model_id}}; prompt: ${{item.prompt_version}}; verified: ${{item.verified_at}}`);
      list(section, 'Confidence factors', Object.entries(item.confidence_factors).map(([name, score]) => `${{name}}: ${{score}}`)); panel.append(section);
    }});
    record.evidence.forEach(item => {{
      const section = document.createElement('section'); section.className = 'evidence-item';
      add(section, 'h3', item.source.title); add(section, 'p', `Publisher: ${{item.source.publisher || 'Unknown'}} | URL: ${{item.source.url}}`);
      add(section, 'p', `Published: ${{item.source.published_at || 'Unknown'}} | Retrieved: ${{item.source.retrieved_at}}`);
      add(section, 'p', `Source language: ${{item.source.language || 'Unknown'}} | Locator: ${{item.passage.locator}}`);
      add(section, 'p', `Source quality: ${{item.source.quality.score ?? 'Unscored'}} — ${{item.source.quality.rationale || 'No assessment recorded.'}}`);
      add(section, 'h4', 'Original-language passage'); add(section, 'pre', item.passage.text);
      item.translations.forEach(translation => {{ add(section, 'h4', `Translation (${{translation.target_language}})`); add(section, 'pre', translation.text); }});
      panel.append(section);
    }});
    list(panel, 'Corroborating evidence', record.related_evidence.supports);
    list(panel, 'Conflicting evidence', record.related_evidence.contradicts);
    panel.scrollIntoView({{behavior: 'smooth', block: 'start'}});
  }}));
}})();
</script></body></html>"""


def _build_statement_evidence_panels(
    *,
    statements: list[ReportStatement], claims: list[Claim], passages: list[EvidencePassage],
    sources: list[SourceRecord], translations: list[TranslationRecord],
    evidence_links: list[EvidenceLink], verification_results: list[VerificationResult],
) -> dict[str, dict]:
    """Project immutable provenance into the panel data for each report statement."""
    claims_by_id = {claim.id: claim for claim in claims}
    passages_by_id = {passage.id: passage for passage in passages}
    sources_by_id = {source.id: source for source in sources}
    translations_by_passage: dict[UUID, list[TranslationRecord]] = defaultdict(list)
    for translation in translations:
        translations_by_passage[translation.passage_id].append(translation)
    results_by_claim: dict[UUID, list[VerificationResult]] = defaultdict(list)
    for result in verification_results:
        results_by_claim[result.claim_id].append(result)
    links_by_claim: dict[UUID, list[EvidenceLink]] = defaultdict(list)
    for link in evidence_links:
        links_by_claim[link.claim_id].append(link)

    panels: dict[str, dict] = {}
    for statement in statements:
        evidence = []
        related = {"supports": [], "contradicts": []}
        for passage_id in statement.citation_ids:
            passage = passages_by_id.get(passage_id)
            if passage is None:
                continue
            source = sources_by_id.get(passage.source_id)
            quality = source.initial_quality_assessment if source else None
            evidence.append({
                "passage": {"id": str(passage.id), "text": passage.text, "locator": passage.locator},
                "source": {
                    "title": source.title if source else "Unknown source",
                    "publisher": source.publisher if source else None,
                    "url": source.canonical_url if source else None,
                    "published_at": source.published_at.isoformat() if source and source.published_at else None,
                    "retrieved_at": source.retrieved_at.isoformat() if source else None,
                    "language": (source.content_language or source.language) if source else passage.original_language,
                    "quality": {"score": quality.score if quality else None, "rationale": "; ".join(quality.rationale) if quality else None},
                },
                "translations": [{"target_language": item.target_language, "text": item.translated_text} for item in translations_by_passage[passage.id]],
            })
        for claim_id in statement.claim_ids:
            for link in links_by_claim[claim_id]:
                if link.relationship in related:
                    linked_passage = passages_by_id.get(link.passage_id)
                    linked_source = sources_by_id.get(linked_passage.source_id) if linked_passage else None
                    evidence_label = (
                        f"{linked_source.title if linked_source else 'Unknown source'} "
                        f"({linked_passage.locator if linked_passage else link.passage_id}): "
                        f"{linked_passage.text if linked_passage else ''}"
                    )
                    related[link.relationship].append(
                        evidence_label + (f" — {link.rationale}" if link.rationale else "")
                    )
        history = [
            {"attempt_number": item.attempt_number, "status": item.status.value, "confidence": item.confidence,
             "rationale": item.rationale, "model_id": item.verifier_model_id,
             "prompt_version": item.verifier_prompt_version, "verified_at": item.verified_at.isoformat(),
             "confidence_factors": item.confidence_factors}
            for claim_id in statement.claim_ids
            for item in sorted(results_by_claim[claim_id], key=lambda result: (result.attempt_number, result.verified_at))
        ]
        panels[str(statement.id)] = {"rendered_text": statement.rendered_text, "verification_status": statement.verification_status.value, "verification_history": history, "evidence": evidence, "related_evidence": related}
    return panels
