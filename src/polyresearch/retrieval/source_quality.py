"""Deterministic, explainable initial source-quality scoring."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from polyresearch.models import SourceQualityAssessment, SourceRecord


_TOKEN = re.compile(r"\w+", re.UNICODE)
_TYPE_SCORES = {
    "official": 1.0,
    "primary": 0.95,
    "peer_reviewed": 0.9,
    "news": 0.65,
    "commentary": 0.35,
    "unverified": 0.15,
    "web": 0.45,
}


def _tokens(value: str) -> set[str]:
    return set(_TOKEN.findall(value.casefold()))


def _relevance(query: str | None, title: str, content: str) -> float:
    if not query:
        return 0.5
    query_tokens = _tokens(query)
    document_tokens = _tokens(f"{title} {content}")
    if not query_tokens or not document_tokens:
        return 0.0
    # Query coverage is more interpretable than a document-length-sensitive score.
    return len(query_tokens & document_tokens) / len(query_tokens)


def score_initial_source_quality(
    source: SourceRecord, content: str, *, query: str | None = None
) -> SourceQualityAssessment:
    """Score only observable initial signals; later verification may supersede it."""
    hostname = (urlsplit(source.canonical_url).hostname or "").casefold()
    source_type = source.source_type.casefold()
    source_type_score = _TYPE_SCORES.get(source_type, _TYPE_SCORES["web"])
    transparent_publisher = bool(source.publisher)
    publisher_score = 1.0 if transparent_publisher else 0.75 if hostname else 0.25
    attribution_score = (0.55 if source.author else 0.0) + (
        0.45 if source.published_at or source.updated_at else 0.0
    )
    primary_signal = 1.0 if source_type in {"official", "primary"} else (
        0.9 if hostname.endswith(".gov") or ".gov." in hostname else 0.2
    )
    relevance_score = _relevance(query, source.title, content)
    factors = {
        "source_type": source_type_score,
        "publisher_transparency": publisher_score,
        "author_and_date_presence": attribution_score,
        "primary_source_signals": primary_signal,
        "relevance": relevance_score,
    }
    score = sum(
        factors[name] * weight
        for name, weight in {
            "source_type": 0.25,
            "publisher_transparency": 0.15,
            "author_and_date_presence": 0.15,
            "primary_source_signals": 0.2,
            "relevance": 0.25,
        }.items()
    )
    rationale = [
        f"source_type={source_type}",
        "publisher_metadata_present" if transparent_publisher else "publisher_inferred_from_domain",
        "author_present" if source.author else "author_missing",
        "date_present" if source.published_at or source.updated_at else "date_missing",
        "primary_source_signal" if primary_signal >= 0.9 else "no_primary_source_signal",
        f"query_coverage={relevance_score:.2f}",
    ]
    return SourceQualityAssessment(
        score=round(score, 4),
        scoring_version="initial-source-quality-v1",
        factor_scores={name: round(value, 4) for name, value in factors.items()},
        rationale=rationale,
    )
