"""Conservative, multilingual entity resolution for extracted claim artifacts."""

from __future__ import annotations

import unicodedata
from uuid import NAMESPACE_URL, uuid5

from polyresearch.models.evidence import Claim, ClaimEntity


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _identity_forms(entity: ClaimEntity) -> set[str]:
    """Include every recorded form without translating or discarding original scripts."""
    forms = [
        entity.original_name,
        entity.normalized_name or "",
        *entity.aliases,
        *entity.script_variants,
        *entity.transliterations,
        *entity.historical_names,
    ]
    return {_normalize(form) for form in forms if form.strip()}


def resolve_claim_entities(claims: list[Claim]) -> list[Claim]:
    """Link exact/declared multilingual aliases; preserve ambiguity rather than guess."""
    flattened = [
        (claim_index, entity_index, entity)
        for claim_index, claim in enumerate(claims)
        for entity_index, entity in enumerate(claim.entities)
    ]
    parents = list(range(len(flattened)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    forms = [_identity_forms(entity) for _, _, entity in flattened]
    for left in range(len(flattened)):
        left_type = flattened[left][2].entity_type
        for right in range(left + 1, len(flattened)):
            right_type = flattened[right][2].entity_type
            if left_type and right_type and left_type != right_type:
                continue
            if forms[left] & forms[right]:
                union(left, right)

    groups: dict[int, list[int]] = {}
    for index in range(len(flattened)):
        groups.setdefault(find(index), []).append(index)

    resolved_entities: dict[int, ClaimEntity] = {}
    for members in groups.values():
        canonical_names = {
            _normalize(flattened[index][2].normalized_name)
            for index in members
            if flattened[index][2].normalized_name
        }
        group_forms = sorted({form for index in members for form in forms[index]})
        if len(canonical_names) == 1:
            entity_id = uuid5(NAMESPACE_URL, "polyresearch/entity/" + "|".join(group_forms))
            status, confidence, rationale = (
                "resolved",
                0.9 if len(members) > 1 else 0.75,
                "Matched by normalized name or declared alias/script/transliteration/historical form.",
            )
        elif len(canonical_names) > 1:
            entity_id = None
            status, confidence, rationale = (
                "ambiguous",
                0.4,
                "Overlapping forms map to multiple normalized names; no canonical entity was chosen.",
            )
        else:
            entity_id = None
            status, confidence, rationale = (
                "unresolved",
                None,
                "No normalized entity mapping was supplied; original forms are preserved.",
            )
        for index in members:
            resolved_entities[index] = flattened[index][2].model_copy(
                update={
                    "resolved_entity_id": entity_id,
                    "resolution_status": status,
                    "resolution_confidence": confidence,
                    "resolution_rationale": rationale,
                }
            )

    resolved_claims = list(claims)
    for claim_index, claim in enumerate(claims):
        entities = [
            resolved_entities[index]
            for index, (member_claim_index, _, _) in enumerate(flattened)
            if member_claim_index == claim_index
        ]
        resolved_claims[claim_index] = claim.model_copy(update={"entities": entities})
    return resolved_claims
