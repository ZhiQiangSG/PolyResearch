"""Deterministic QA checks for auditable report statements."""

import re

from polyresearch.models import (
    Claim,
    EvidencePassage,
    QueryRecord,
    ReportQaIssue,
    ReportStatement,
    SourceRecord,
    VerificationStatus,
)

_HEDGING_PATTERN = re.compile(
    r"\b(may|might|could|appears?|suggests?|indicates?|uncertain|insufficient|"
    r"not comparable|conflict|disputed|according to)\b",
    re.IGNORECASE,
)


def validate_report_statements(
    *,
    statements: list[ReportStatement],
    claims: list[Claim],
    passages: list[EvidencePassage],
    sources: list[SourceRecord],
    queries: list[QueryRecord] | None = None,
) -> list[ReportQaIssue]:
    """Return blocking integrity errors and conservative wording warnings."""
    claim_ids = {claim.id for claim in claims}
    claims_by_id = {claim.id: claim for claim in claims}
    passages_by_id = {passage.id: passage for passage in passages}
    sources_by_id = {source.id: source for source in sources}
    source_ids = {source.id for source in sources}
    query_urls = (
        {query.result_url for query in queries if query.result_url}
        if queries is not None
        else None
    )
    issues: list[ReportQaIssue] = []
    for statement in statements:
        unknown_claims = set(statement.claim_ids) - claim_ids
        if unknown_claims:
            issues.append(
                ReportQaIssue(
                    code="unknown_claim_id",
                    severity="error",
                    statement_id=statement.id,
                    message="Statement references claim IDs that are absent from the ledger.",
                )
            )
        if not statement.citation_ids:
            issues.append(
                ReportQaIssue(
                    code="missing_citation",
                    severity="error",
                    statement_id=statement.id,
                    message="Statement has no passage-level citation IDs.",
                )
            )
        for citation_id in statement.citation_ids:
            passage = passages_by_id.get(citation_id)
            if passage is None:
                issues.append(
                    ReportQaIssue(
                        code="unresolvable_citation",
                        severity="error",
                        statement_id=statement.id,
                        message=f"Citation {citation_id} does not resolve to an evidence passage.",
                    )
                )
            elif passage.source_id not in source_ids:
                issues.append(
                    ReportQaIssue(
                        code="orphaned_citation_source",
                        severity="error",
                        statement_id=statement.id,
                        message=f"Citation {citation_id} resolves to a passage without a source.",
                    )
                )
        if query_urls is not None:
            has_discovery_trace = any(
                (
                    source := sources_by_id.get(passages_by_id[passage_id].source_id)
                )
                and {source.canonical_url, source.discovered_url} & query_urls
                for claim_id in statement.claim_ids
                if (claim := claims_by_id.get(claim_id))
                for passage_id in claim.evidence_passage_ids
                if passage_id in passages_by_id
            )
            if not has_discovery_trace:
                issues.append(
                    ReportQaIssue(
                        code="incomplete_discovery_trace",
                        severity="error",
                        statement_id=statement.id,
                        message=(
                            "Statement lacks a claim → passage → source → query "
                            "discovery trace."
                        ),
                    )
                )
        if statement.verification_status in {
            VerificationStatus.PARTIALLY_SUPPORTED,
            VerificationStatus.CONTRADICTED,
            VerificationStatus.INSUFFICIENT_EVIDENCE,
            VerificationStatus.OUTDATED,
            VerificationStatus.NOT_COMPARABLE,
        } and not _HEDGING_PATTERN.search(statement.rendered_text):
            issues.append(
                ReportQaIssue(
                    code="wording_exceeds_verification_status",
                    severity="warning",
                    statement_id=statement.id,
                    message=(
                        f"Statement is assertive despite verification status "
                        f"{statement.verification_status.value}."
                    ),
                )
            )
    return issues
