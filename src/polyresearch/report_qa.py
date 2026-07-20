"""Deterministic QA checks for auditable report statements."""

import re

from polyresearch.models import (
    Claim,
    EvidencePassage,
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
) -> list[ReportQaIssue]:
    """Return blocking integrity errors and conservative wording warnings."""
    claim_ids = {claim.id for claim in claims}
    passages_by_id = {passage.id: passage for passage in passages}
    source_ids = {source.id for source in sources}
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
