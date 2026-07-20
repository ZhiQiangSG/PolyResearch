"""Conservative proposition clustering for claim-level corroboration."""

from __future__ import annotations

import re
import unicodedata
from uuid import NAMESPACE_URL, uuid5

from polyresearch.models.evidence import Claim


_TOKEN = re.compile(r"\w+", re.UNICODE)


def _terms(value: str) -> set[str]:
    return set(_TOKEN.findall(unicodedata.normalize("NFKC", value).casefold()))


def _similarity(left: Claim, right: Claim) -> float:
    left_terms, right_terms = _terms(left.atomic_proposition or left.statement), _terms(
        right.atomic_proposition or right.statement
    )
    if not left_terms or not right_terms:
        return 0.0
    lexical = len(left_terms & right_terms) / len(left_terms | right_terms)
    left_entities = {entity.resolved_entity_id for entity in left.entities if entity.resolved_entity_id}
    right_entities = {entity.resolved_entity_id for entity in right.entities if entity.resolved_entity_id}
    if left_entities and right_entities and left_entities & right_entities:
        return min(1.0, lexical + 0.15)
    return lexical


def cluster_claims(claims: list[Claim]) -> list[Claim]:
    """Group only high-similarity propositions; leave uncertain claims unclustered."""
    parents = list(range(len(claims)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    similarities: dict[tuple[int, int], float] = {}
    for left in range(len(claims)):
        for right in range(left + 1, len(claims)):
            similarity = _similarity(claims[left], claims[right])
            similarities[left, right] = similarity
            if similarity >= 0.8:
                union(left, right)

    groups: dict[int, list[int]] = {}
    for index in range(len(claims)):
        groups.setdefault(find(index), []).append(index)

    clustered = list(claims)
    for members in groups.values():
        if len(members) < 2:
            continue
        propositions = sorted(
            unicodedata.normalize("NFKC", claims[index].atomic_proposition or claims[index].statement)
            for index in members
        )
        cluster_id = uuid5(NAMESPACE_URL, "polyresearch/claim-cluster/" + "|".join(propositions))
        pair_scores = [
            similarities[min(left, right), max(left, right)]
            for position, left in enumerate(members)
            for right in members[position + 1 :]
        ]
        confidence = sum(pair_scores) / len(pair_scores)
        for index in members:
            clustered[index] = claims[index].model_copy(
                update={
                    "claim_cluster_id": cluster_id,
                    "claim_cluster_method": "lexical_proposition_similarity_v1",
                    "claim_cluster_confidence": round(confidence, 4),
                }
            )
    return clustered
