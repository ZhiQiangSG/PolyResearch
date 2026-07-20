"""Selection helpers for immutable claim-verification attempts."""

from uuid import UUID

from polyresearch.models import VerificationResult


def latest_results_by_claim_id(
    results: list[VerificationResult],
) -> dict[UUID, VerificationResult]:
    """Select the newest immutable verification attempt for each claim."""
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
