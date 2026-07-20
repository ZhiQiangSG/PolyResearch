import unittest
from uuid import uuid4

from polyresearch.evidence.entity_resolution import resolve_claim_entities
from polyresearch.models import Claim, ClaimEntity


class EntityResolutionTests(unittest.TestCase):
    def test_resolves_aliases_native_scripts_transliterations_and_historical_names(self) -> None:
        first = Claim(
            statement="Beijing issued the notice.",
            evidence_passage_ids=[uuid4()],
            extraction_confidence=0.9,
            entities=[
                ClaimEntity(
                    original_name="北京",
                    normalized_name="Beijing",
                    entity_type="place",
                    aliases=["Peking"],
                    script_variants=["北京"],
                    transliterations=["Beijing"],
                    historical_names=["Peking"],
                )
            ],
        )
        second = Claim(
            statement="Peking published the notice.",
            evidence_passage_ids=[uuid4()],
            extraction_confidence=0.9,
            entities=[
                ClaimEntity(
                    original_name="Peking",
                    normalized_name="Beijing",
                    entity_type="place",
                    aliases=["北京"],
                )
            ],
        )

        resolved = resolve_claim_entities([first, second])

        first_entity, second_entity = resolved[0].entities[0], resolved[1].entities[0]
        self.assertEqual(first_entity.resolution_status, "resolved")
        self.assertEqual(first_entity.resolved_entity_id, second_entity.resolved_entity_id)
        self.assertIn("transliteration", first_entity.resolution_rationale)

    def test_preserves_uncertain_mapping_without_forcing_a_canonical_entity(self) -> None:
        claim = Claim(
            statement="Alexandria issued a statement.",
            evidence_passage_ids=[uuid4()],
            extraction_confidence=0.6,
            entities=[ClaimEntity(original_name="Alexandria", entity_type="place")],
        )

        resolved = resolve_claim_entities([claim])

        entity = resolved[0].entities[0]
        self.assertEqual(entity.resolution_status, "unresolved")
        self.assertIsNone(entity.resolved_entity_id)
