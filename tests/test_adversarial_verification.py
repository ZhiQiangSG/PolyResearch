"""Adversarial fixtures for multilingual claim verification and conflict reporting."""

from datetime import datetime, timezone
from uuid import uuid4

from polyresearch.evidence.verification_confidence import verification_confidence
from polyresearch.models import (
    Claim,
    DisagreementAssessment,
    DisagreementDimension,
    EvidenceLink,
    EvidencePassage,
    SourceRecord,
    SourceQualityAssessment,
    TranslationRecord,
    VerificationResult,
    VerificationStatus,
)
from polyresearch.workflows.report_generator import _build_unresolved_disagreements


def _assessment(
    dimension: DisagreementDimension, present: bool, explanation: str
) -> DisagreementAssessment:
    return DisagreementAssessment(
        dimension=dimension, present=present, explanation=explanation
    )


def _source(url: str, title: str, **updates) -> SourceRecord:
    return SourceRecord(
        canonical_url=url,
        title=title,
        initial_quality_assessment=SourceQualityAssessment(
            score=0.9, scoring_version="adversarial-fixture-v1"
        ),
        **updates,
    )


def _confidence(
    claim: Claim,
    links: list[EvidenceLink],
    passages: list[EvidencePassage],
    sources: list[SourceRecord],
    *,
    status: VerificationStatus = VerificationStatus.SUPPORTED,
    translations: list[TranslationRecord] | None = None,
    assessments: list[DisagreementAssessment] | None = None,
) -> tuple[float, dict[str, float], int]:
    return verification_confidence(
        claim=claim,
        status=status,
        model_confidence=0.95,
        evidence_links=links,
        passages=passages,
        sources=sources,
        translations=translations or [],
        output_language="en",
        disagreement_assessments=assessments or [],
    )


def test_conflicting_chinese_and_english_sources_remain_an_unresolved_output() -> None:
    cluster_id = uuid4()
    chinese_source = _source("https://gov.example.cn/notice", "官方通报", language="zh")
    english_source = _source("https://agency.example.org/report", "Agency report", language="en")
    chinese_passage = EvidencePassage(
        source_id=chinese_source.id,
        text="该项目完成率为百分之八十。",
        locator="第3段",
        original_language="zh",
    )
    english_passage = EvidencePassage(
        source_id=english_source.id,
        text="The programme achieved 55 percent completion.",
        locator="paragraph 4",
        original_language="en",
    )
    chinese_claim = Claim(
        statement="The programme achieved 80 percent completion.",
        original_wording="该项目完成率为百分之八十。",
        evidence_passage_ids=[chinese_passage.id],
        extraction_confidence=0.9,
        claim_cluster_id=cluster_id,
    )
    english_claim = Claim(
        statement="The programme achieved 55 percent completion.",
        evidence_passage_ids=[english_passage.id],
        extraction_confidence=0.9,
        claim_cluster_id=cluster_id,
    )
    conflict = _assessment(
        DisagreementDimension.GENUINE_CONFLICT,
        True,
        "The Chinese and English records make incompatible claims for the same programme.",
    )

    disagreements = _build_unresolved_disagreements(
        claims=[chinese_claim, english_claim],
        verification_results=[
            VerificationResult(
                claim_id=chinese_claim.id,
                status=VerificationStatus.CONTRADICTED,
                confidence=0.4,
                rationale="The opposing official records conflict.",
                disagreement_assessments=[conflict],
            ),
            VerificationResult(
                claim_id=english_claim.id,
                status=VerificationStatus.CONTRADICTED,
                confidence=0.4,
                rationale="The opposing official records conflict.",
                disagreement_assessments=[conflict],
            ),
        ],
    )

    assert len(disagreements) == 1
    assert disagreements[0].conflicting_claims == [
        chinese_claim.statement,
        english_claim.statement,
    ]
    assert disagreements[0].evidence_needed == [
        "Additional independent primary or official records that directly address the conflicting proposition."
    ]


def test_reposted_sources_do_not_inflate_independent_corroboration() -> None:
    original = _source(
        "https://wire.example/original", "Wire original", shared_origin_cluster_id="wire:42"
    )
    repost = _source(
        "https://copy.example/repost", "Wire repost", near_duplicate_cluster_id="wire:42"
    )
    independent = _source("https://ministry.example/record", "Ministry record")
    passages = [
        EvidencePassage(source_id=source.id, text="The rule changed.", locator="p1")
        for source in (original, repost, independent)
    ]
    claim = Claim(
        statement="The rule changed.",
        evidence_passage_ids=[passage.id for passage in passages],
        extraction_confidence=0.9,
    )
    links = [
        EvidenceLink(claim_id=claim.id, passage_id=passage.id, relationship="supports")
        for passage in passages
    ]

    _, factors, independent_source_count = _confidence(
        claim, links, passages, [original, repost, independent]
    )

    assert independent_source_count == 2
    assert factors["independence"] == 0.85


def test_stale_claim_is_scored_conservatively_as_outdated() -> None:
    source = _source(
        "https://archive.example/2010", "2010 archive",
        published_at=datetime(2010, 1, 1, tzinfo=timezone.utc),
    )
    passage = EvidencePassage(source_id=source.id, text="The rule applies.", locator="p1")
    claim = Claim(
        statement="The rule currently applies.",
        evidence_passage_ids=[passage.id],
        extraction_confidence=0.9,
    )
    link = EvidenceLink(claim_id=claim.id, passage_id=passage.id, relationship="supports")

    confidence, factors, _ = _confidence(
        claim, [link], [passage], [source], status=VerificationStatus.OUTDATED
    )

    assert factors["recency"] == 0.2
    assert factors["agreement"] == 0.3
    assert confidence <= 0.3


def test_incompatible_measurements_are_not_comparable_and_name_needed_methodology() -> None:
    cluster_id = uuid4()
    first = Claim(
        statement="Unemployment was 5 percent under the survey definition.",
        evidence_passage_ids=[uuid4()],
        extraction_confidence=0.8,
        claim_cluster_id=cluster_id,
    )
    second = Claim(
        statement="Unemployment was 8 percent under the registered-jobseeker definition.",
        evidence_passage_ids=[uuid4()],
        extraction_confidence=0.8,
        claim_cluster_id=cluster_id,
    )
    incompatible_method = _assessment(
        DisagreementDimension.DEFINITION_OR_METHOD,
        True,
        "The sources use different unemployment definitions and measurement methods.",
    )

    disagreements = _build_unresolved_disagreements(
        claims=[first, second],
        verification_results=[
            VerificationResult(
                claim_id=first.id,
                status=VerificationStatus.NOT_COMPARABLE,
                confidence=0.25,
                rationale="The measures are not comparable.",
                disagreement_assessments=[incompatible_method],
            ),
            VerificationResult(
                claim_id=second.id,
                status=VerificationStatus.NOT_COMPARABLE,
                confidence=0.25,
                rationale="The measures are not comparable.",
                disagreement_assessments=[incompatible_method],
            ),
        ],
    )

    assert len(disagreements) == 1
    assert disagreements[0].why_it_may_conflict == [incompatible_method.explanation]
    assert disagreements[0].evidence_needed == [
        "Methodology notes or official definitions that make the measurements comparable."
    ]


def test_ambiguous_chinese_translation_reduces_verification_confidence() -> None:
    source = _source("https://law.example.cn/article", "条例", language="zh")
    passage = EvidencePassage(
        source_id=source.id,
        text="有关部门可以采取必要措施。",
        locator="第12条",
        original_language="zh",
    )
    claim = Claim(
        statement="Authorities must take necessary measures.",
        original_wording="有关部门可以采取必要措施。",
        evidence_passage_ids=[passage.id],
        extraction_confidence=0.8,
    )
    link = EvidenceLink(claim_id=claim.id, passage_id=passage.id, relationship="supports")
    ambiguous_translation = TranslationRecord(
        passage_id=passage.id,
        translated_text="Relevant departments may take necessary measures.",
        target_language="en",
        source_original_text_hash=passage.original_text_hash,
        confidence=0.15,
    )
    translation_ambiguity = _assessment(
        DisagreementDimension.TRANSLATION_AMBIGUITY,
        True,
        "The Chinese term 可以 indicates permission, while the English claim asserts an obligation.",
    )

    confidence, factors, _ = _confidence(
        claim,
        [link],
        [passage],
        [source],
        translations=[ambiguous_translation],
        assessments=[translation_ambiguity],
    )

    assert factors["translation_certainty"] == 0.15
    assert confidence < 0.8
