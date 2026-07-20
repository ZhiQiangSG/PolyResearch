"""Explainable source de-duplication and independence clustering."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from urllib.parse import urlsplit

from polyresearch.models import EvidencePassage, SourceRecord, SourceVersion


_TOKEN = re.compile(r"\w+", re.UNICODE)


def _normalized_tokens(value: str) -> set[str]:
    return set(_TOKEN.findall(value.casefold()))


def _key(prefix: str, value: str) -> str:
    return f"{prefix}:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]}"


def publisher_family(source: SourceRecord) -> str:
    """Return a conservative publisher-family label from metadata or hostname."""
    if source.publisher:
        return "publisher:" + " ".join(source.publisher.casefold().split())
    hostname = (urlsplit(source.canonical_url).hostname or "").casefold()
    labels = hostname.removeprefix("www.").split(".")
    return "domain:" + ".".join(labels[-2:]) if len(labels) >= 2 else "domain:" + hostname


def shared_origin_cluster_id(source: SourceRecord, content: str) -> str:
    """Cluster likely shared-origin copies without claiming that they are identical."""
    title = " ".join(source.title.casefold().split())
    opening = " ".join(content.casefold().split())[:240]
    return _key("origin", f"{title}\n{opening}")


def _similar(left: str, right: str) -> bool:
    left_tokens, right_tokens = _normalized_tokens(left), _normalized_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens) >= 0.82


def deduplicate_source_artifacts(
    sources: list[SourceRecord],
    versions: list[SourceVersion],
    passages: list[EvidencePassage],
    *,
    existing_sources: Iterable[SourceRecord] = (),
    existing_versions: Iterable[SourceVersion] = (),
) -> tuple[list[SourceRecord], list[SourceVersion], list[EvidencePassage]]:
    """Suppress canonical/exact copies and annotate retained independence clusters."""
    version_by_source = {version.source_id: version for version in versions}
    existing_version_by_source = {version.source_id: version for version in existing_versions}
    known_urls = {source.canonical_url for source in existing_sources}
    known_hashes = {source.content_hash for source in existing_sources if source.content_hash}
    prior_contents = [
        (source, existing_version_by_source[source.id].raw_content)
        for source in existing_sources
        if source.id in existing_version_by_source
    ]
    kept: list[SourceRecord] = []
    kept_contents: list[tuple[SourceRecord, str]] = []
    kept_ids = set()

    for source in sources:
        version = version_by_source[source.id]
        if source.canonical_url in known_urls or source.content_hash in known_hashes:
            continue
        family = publisher_family(source)
        origin_cluster = shared_origin_cluster_id(source, version.raw_content)
        near_cluster = None
        for prior, prior_content in [*prior_contents, *kept_contents]:
            if _similar(version.raw_content, prior_content):
                near_cluster = prior.near_duplicate_cluster_id or prior.shared_origin_cluster_id
                if near_cluster is None:
                    near_cluster = shared_origin_cluster_id(prior, prior_content)
                break
        annotated = source.model_copy(
            update={
                "publisher_family": family,
                "shared_origin_cluster_id": origin_cluster,
                "near_duplicate_cluster_id": near_cluster,
            }
        )
        kept.append(annotated)
        kept_contents.append((annotated, version.raw_content))
        kept_ids.add(source.id)
        known_urls.add(source.canonical_url)
        if source.content_hash:
            known_hashes.add(source.content_hash)

    return (
        kept,
        [version for version in versions if version.source_id in kept_ids],
        [passage for passage in passages if passage.source_id in kept_ids],
    )
