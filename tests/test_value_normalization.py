import unittest
from uuid import uuid4

from polyresearch.models import Claim, ClaimDate, ClaimQuantity
from polyresearch.evidence.value_normalization import normalize_claim_values


class ValueNormalizationTests(unittest.TestCase):
    def test_normalizes_dates_numbers_currencies_and_units_without_replacing_originals(self) -> None:
        claim = Claim(
            statement="The budget was US$1,234.50 on 2026年7月20日.",
            evidence_passage_ids=[uuid4()], extraction_confidence=0.9,
            quantities=[
                ClaimQuantity(original_value="US$1,234.50"),
                ClaimQuantity(original_value="2,500 km", unit="km"),
                ClaimQuantity(original_value="75%"),
            ], dates=[ClaimDate(original_value="2026年7月20日")],
        )

        normalized = normalize_claim_values([claim])[0]

        self.assertEqual(normalized.quantities[0].original_value, "US$1,234.50")
        self.assertEqual(normalized.quantities[0].normalized_value, "1234.5")
        self.assertEqual(normalized.quantities[0].currency_code, "USD")
        self.assertEqual(normalized.quantities[1].normalized_unit, "kilometre")
        self.assertEqual(normalized.quantities[2].normalized_unit, "percent")
        self.assertEqual(normalized.dates[0].original_value, "2026年7月20日")
        self.assertEqual(normalized.dates[0].normalized_value, "2026-07-20")

    def test_retains_ambiguous_values_without_forcing_normalization(self) -> None:
        claim = Claim(
            statement="The event was in spring.", evidence_passage_ids=[uuid4()],
            extraction_confidence=0.6, quantities=[ClaimQuantity(original_value="many")],
            dates=[ClaimDate(original_value="spring 2026")],
        )

        normalized = normalize_claim_values([claim])[0]

        self.assertEqual(normalized.quantities[0].normalization_status, "original_retained")
        self.assertEqual(normalized.dates[0].normalization_status, "original_retained")
