"""Deterministic, conservative confidence scoring for verified claims."""

from datetime import datetime, timezone

from polyresearch.models import (
    Claim,
    DisagreementDimension,
    EvidenceLink,
    EvidencePassage,
    SourceRecord,
    TranslationRecord,
    VerificationStatus,
)


def _source_origin(source: SourceRecord) -> str:
    """Return the provenance cluster used to count independent corroboration."""
    return (
        source.near_duplicate_cluster_id
        or source.shared_origin_cluster_id
        or source.publisher_family
        or str(source.id)
    )


def verification_confidence(
    *,
    claim: Claim,
    status: VerificationStatus,
    model_confidence: float,
    evidence_links: list[EvidenceLink],
    passages: list[EvidencePassage],
    sources: list[SourceRecord],
    translations: list[TranslationRecord],
    output_language: str,
    disagreement_assessments,
) -> tuple[float, dict[str, float], int]:
    """Cap model confidence with reproducible evidence-quality factors.

    Pages in the same origin or near-duplicate cluster count as one corroborating
    source, preventing wire copies and reposts from inflating confidence.
    """
    passage_by_id = {passage.id: passage for passage in passages}
    source_by_id = {source.id: source for source in sources}
    linked_passages = [
        passage_by_id[link.passage_id]
        for link in evidence_links
        if link.passage_id in passage_by_id
    ]
    linked_sources = [
        source_by_id[passage.source_id]
        for passage in linked_passages
        if passage.source_id in source_by_id
    ]
    relationship_scores = {"supports": 1.0, "contradicts": 0.9, "contextualizes": 0.55}
    directness = (
        sum(relationship_scores[link.relationship] for link in evidence_links)
        / len(evidence_links)
        if evidence_links
        else 0.0
    )
    source_quality = (
        sum(
            source.initial_quality_assessment.score
            if source.initial_quality_assessment
            else 0.3
            for source in linked_sources
        )
        / len(linked_sources)
        if linked_sources
        else 0.0
    )
    independent_source_count = len({_source_origin(source) for source in linked_sources})
    independence = min(1.0, 0.45 + 0.2 * independent_source_count)
    scope_dimensions = {
        DisagreementDimension.TIME_PERIOD,
        DisagreementDimension.GEOGRAPHIC_SCOPE,
        DisagreementDimension.DEFINITION_OR_METHOD,
        DisagreementDimension.POPULATION_OR_SAMPLE,
    }
    scope_mismatches = sum(
        assessment.present
        for assessment in disagreement_assessments
        if assessment.dimension in scope_dimensions
    )
    scope_fit = max(0.0, 1.0 - 0.2 * scope_mismatches)
    dated_sources = [source.updated_at or source.published_at for source in linked_sources]
    dated_sources = [date for date in dated_sources if date is not None]
    if dated_sources:
        newest = max(dated_sources)
        age_years = max(0.0, (datetime.now(timezone.utc) - newest).days / 365.25)
        recency = max(0.2, 1.0 - age_years / 10)
    else:
        recency = 0.4
    genuine_conflict = any(
        assessment.present
        and assessment.dimension is DisagreementDimension.GENUINE_CONFLICT
        for assessment in disagreement_assessments
    )
    agreement = {
        VerificationStatus.SUPPORTED: 0.9,
        VerificationStatus.PARTIALLY_SUPPORTED: 0.6,
        VerificationStatus.CONTRADICTED: 0.35,
        VerificationStatus.INSUFFICIENT_EVIDENCE: 0.2,
        VerificationStatus.OUTDATED: 0.3,
        VerificationStatus.NOT_COMPARABLE: 0.25,
    }[status]
    if genuine_conflict:
        agreement = min(agreement, 0.4)
    translation_by_passage = {
        translation.passage_id: translation
        for translation in translations
        if translation.target_language.casefold() == output_language.casefold()
    }
    translation_scores = []
    for passage in linked_passages:
        if not passage.original_language or passage.original_language.casefold() == output_language.casefold():
            translation_scores.append(1.0)
        else:
            translation = translation_by_passage.get(passage.id)
            translation_scores.append(translation.confidence if translation and translation.confidence is not None else 0.35)
    translation_certainty = (
        sum(translation_scores) / len(translation_scores) if translation_scores else 0.35
    )
    factors = {
        "directness": directness,
        "source_quality": source_quality,
        "independence": independence,
        "scope_fit": scope_fit,
        "recency": recency,
        "agreement": agreement,
        "translation_certainty": translation_certainty,
    }
    weighted = sum(
        factors[name] * weight
        for name, weight in {
            "directness": 0.2,
            "source_quality": 0.2,
            "independence": 0.15,
            "scope_fit": 0.15,
            "recency": 0.1,
            "agreement": 0.1,
            "translation_certainty": 0.1,
        }.items()
    )
    # Confidence describes how safely the claim can be used in a current report,
    # not merely how certain the model is about its classification.  A stale,
    # contradicted, or non-comparable claim must therefore remain visibly bounded
    # even when its underlying source is direct and high quality.
    return round(min(model_confidence, weighted, agreement), 4), {
        name: round(value, 4) for name, value in factors.items()
    }, independent_source_count
