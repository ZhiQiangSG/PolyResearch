"""Conservative normalization of claim dates, numbers, currencies, and units."""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from decimal import Decimal, InvalidOperation

from polyresearch.models.evidence import Claim, ClaimDate, ClaimQuantity


_NUMBER = re.compile(r"(?<!\w)([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?!\w)")
_CHINESE_DATE = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_SLASH_DATE = re.compile(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})")
_CURRENCIES = {
    "US$": "USD", "USD": "USD", "美元": "USD", "€": "EUR", "EUR": "EUR",
    "£": "GBP", "GBP": "GBP", "人民币": "CNY", "元": "CNY", "CNY": "CNY",
    "JPY": "JPY", "日元": "JPY",
}
_UNITS = {
    "%": "percent", "percent": "percent", "km": "kilometre",
    "kilometers": "kilometre", "kilometres": "kilometre", "kg": "kilogram",
    "kilograms": "kilogram", "m": "metre", "meters": "metre", "metres": "metre",
}


def _decimal_string(value: str) -> str | None:
    try:
        number = Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None
    normalized = format(number.normalize(), "f")
    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


def normalize_quantity(quantity: ClaimQuantity) -> ClaimQuantity:
    """Derive machine-comparable values while retaining the exact source string."""
    original = unicodedata.normalize("NFKC", quantity.original_value)
    number_match = _NUMBER.search(original)
    normalized_value = _decimal_string(number_match.group(1)) if number_match else None
    currency = next((code for marker, code in _CURRENCIES.items() if marker.casefold() in original.casefold()), None)
    explicit_unit = quantity.unit or next(
        (
            unit
            for marker, unit in _UNITS.items()
            if (marker == "%" and marker in original)
            or (marker != "%" and re.search(rf"\b{re.escape(marker)}\b", original, re.I))
        ),
        None,
    )
    normalized_unit = _UNITS.get((explicit_unit or "").casefold(), explicit_unit)
    if normalized_value is None:
        return quantity.model_copy(update={
            "normalization_status": "original_retained",
            "normalization_notes": ["No unambiguous decimal number was parsed."],
        })
    notes = ["Decimal separators and Unicode digits normalized."]
    if currency:
        notes.append(f"Currency identified as {currency}.")
    if normalized_unit:
        notes.append(f"Unit normalized to {normalized_unit}.")
    return quantity.model_copy(update={
        "normalized_value": normalized_value,
        "normalized_unit": normalized_unit,
        "currency_code": currency,
        "normalization_status": "normalized",
        "normalization_notes": notes,
    })


def normalize_date(claim_date: ClaimDate) -> ClaimDate:
    """Normalize only complete, unambiguous calendar dates to ISO 8601."""
    original = unicodedata.normalize("NFKC", claim_date.original_value).strip()
    match = _CHINESE_DATE.search(original) or _SLASH_DATE.search(original)
    if match:
        year, month, day = map(int, match.groups())
        try:
            normalized = date(year, month, day).isoformat()
        except ValueError:
            normalized = None
    else:
        try:
            normalized = date.fromisoformat(original).isoformat()
        except ValueError:
            normalized = None
    if normalized is None:
        return claim_date.model_copy(update={
            "normalization_status": "original_retained",
            "normalization_notes": ["Date was incomplete or ambiguous."],
        })
    return claim_date.model_copy(update={
        "normalized_value": normalized,
        "normalization_status": "normalized",
        "normalization_notes": ["Complete calendar date normalized to ISO 8601."],
    })


def normalize_claim_values(claims: list[Claim]) -> list[Claim]:
    """Attach normalization derivatives without changing original claim evidence."""
    return [claim.model_copy(update={
        "quantities": [normalize_quantity(quantity) for quantity in claim.quantities],
        "dates": [normalize_date(claim_date) for claim_date in claim.dates],
    }) for claim in claims]
