# app/extractor.py

import re
import logging
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Common receipt keywords that usually indicate the final payable amount
TOTAL_KEYWORDS = [
    "total",
    "grand total",
    "amount due",
    "balance due",
    "net total",
    "total amount",
    "cash due",
    "payable",
]
SUMMARY_LINE_KEYWORDS = TOTAL_KEYWORDS + [
    "cash paid",
    "change",
    "vat",
    "subtotal",
    "tax",
    "balance",
]

# Currency symbols/prefixes
CURRENCY_PATTERN = r"(?:KSh|KES|USD|\$|€|£)?"

# Improved amount regex
AMOUNT_REGEX = re.compile(
    rf"{CURRENCY_PATTERN}\s*(-?\d{{1,3}}(?:[,\s]\d{{3}})*(?:[.,]\d{{2}})?|-?\d+(?:[.,]\d{{2}})?)",
    re.IGNORECASE,
)


def normalize_amount(value: str) -> Optional[float]:
    """
    Convert amount string into a float safely.

    Handles:
    - 1,234.56
    - 1234.56
    - 1234,56
    - 1 234.56
    """

    if not value:
        return None

    value = value.strip().replace(" ", "")

    try:
        # Case: European decimal comma (1234,56)
        if "," in value and "." not in value:
            value = value.replace(",", ".")

        # Case: thousand separators (1,234.56)
        elif "," in value and "." in value:
            value = value.replace(",", "")

        amount = Decimal(value)
        return float(amount)

    except (InvalidOperation, ValueError) as e:
        logger.warning(f"Failed to parse amount '{value}': {e}")
        return None


def extract_all_amounts(text: str) -> List[float]:
    """
    Extract all numeric monetary values from OCR text.
    """

    matches = AMOUNT_REGEX.findall(text)

    amounts = []

    for match in matches:
        amount = normalize_amount(match)

        if amount is not None:
            amounts.append(amount)

    return amounts


def extract_total_amount(text: str) -> Optional[float]:
    """
    Attempt to determine the most likely receipt total.

    Strategy:
    1. Prefer lines containing total keywords.
    2. Otherwise use the largest amount.
    """

    lines = text.splitlines()

    candidate_amounts = []

    # Search lines with total keywords
    for line in lines:
        lower = line.lower()

        if any(keyword in lower for keyword in TOTAL_KEYWORDS):
            amounts = extract_all_amounts(line)

            if amounts:
                candidate_amounts.extend(amounts)

    # Prefer the largest keyword-associated amount
    if candidate_amounts:
        return max(candidate_amounts)

    # Fallback: largest amount in document
    all_amounts = extract_all_amounts(text)

    if all_amounts:
        return max(all_amounts)

    return None


def extract_date(text: str) -> Optional[str]:
    """
    Extract a date from OCR text.

    Supported formats:
    - DD/MM/YYYY
    - YYYY-MM-DD
    - DD-MM-YYYY
    """

    date_patterns = [
        r"\b\d{2}/\d{2}/\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{2}-\d{2}-\d{4}\b",
    ]

    for pattern in date_patterns:
        match = re.search(pattern, text)

        if match:
            return match.group(0)

    return None


def extract_merchant(text: str) -> Optional[str]:
    """
    Attempt to identify merchant/store name.

    Strategy:
    - Use the first non-empty line
    - Ignore lines that are mostly numeric
    """

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines[:5]:
        if not re.fullmatch(r"[\d\W]+", line):
            return line

    return None


def extract_line_items(text: str) -> List[Dict]:
    """
    Extract potential line items from receipt text.

    Example:
    Bread        120.00
    Milk         80.00
    """

    items: List[Dict] = []

    lines = text.splitlines()

    # Patterns to capture: "<desc> 2 x 120.00", "<desc> 2 120.00", "<desc> 120.00"
    patterns = [
        re.compile(rf"^(.+?)\s+(\d+)\s*[xX]\s*{CURRENCY_PATTERN}\s*([\d,\.]+)\s*$", re.IGNORECASE),
        re.compile(rf"^(.+?)\s+(\d+)\s+{CURRENCY_PATTERN}\s*([\d,\.]+)\s*$", re.IGNORECASE),
        re.compile(rf"^(.+?)\s+{CURRENCY_PATTERN}\s*([\d,\.]+)\s*$", re.IGNORECASE),
    ]

    for raw in lines:
        line = raw.strip()

        if not line:
            continue

        low = line.lower()
        if any(keyword in low for keyword in SUMMARY_LINE_KEYWORDS):
            # skip summary lines
            continue

        matched = False

        # Try verbose patterns first (qty x unit)
        m = patterns[0].search(line)
        if m:
            description = m.group(1).strip()
            qty_raw = m.group(2)
            unit_raw = m.group(3)
            qty = int(qty_raw) if qty_raw.isdigit() else 1
            unit = normalize_amount(unit_raw)
            if unit is None:
                continue
            total = round(qty * unit, 2)
            items.append({
                "description": description,
                "quantity": qty,
                "unit_price": unit,
                "total": total,
            })
            matched = True

        if matched:
            continue

        # Try second pattern (desc qty unit)
        m = patterns[1].search(line)
        if m:
            description = m.group(1).strip()
            qty_raw = m.group(2)
            unit_raw = m.group(3)
            qty = int(qty_raw) if qty_raw.isdigit() else 1
            unit = normalize_amount(unit_raw)
            if unit is None:
                continue
            total = round(qty * unit, 2)
            items.append({
                "description": description,
                "quantity": qty,
                "unit_price": unit,
                "total": total,
            })
            continue

        # Fallback: description + amount
        m = patterns[2].search(line)
        if m:
            description = m.group(1).strip()
            amount_raw = m.group(2)
            amount = normalize_amount(amount_raw)
            if amount is None:
                continue
            items.append({
                "description": description,
                "quantity": 1,
                "unit_price": amount,
                "total": amount,
            })

    return items


def extract_field_amount(text: str, keywords: List[str]) -> Optional[float]:
    """Find an amount on a line containing any of the provided keywords."""
    if not text:
        return None
    lines = text.splitlines()

    for line in lines:
        lower = line.lower()
        if any(k in lower for k in keywords):
            matches = AMOUNT_REGEX.findall(line)
            if matches:
                # take last amount on the line
                val = normalize_amount(matches[-1])
                if val is not None:
                    return val
    return None


def extract_fields(text: str) -> Dict:
    """
    Extract structured receipt information from OCR text.

    Returns:
    {
        "merchant": str | None,
        "date": str | None,
        "amount": float | None,
        "amounts_found": list,
        "line_items": list,
        "raw_text": str
    }
    """

    if not text or not isinstance(text, str):
        logger.warning("Invalid OCR text input")
        return {
            "merchant": None,
            "date": None,
            "amount": None,
            "amounts_found": [],
            "line_items": [],
            "raw_text": "",
        }

    try:
        merchant = extract_merchant(text)
        date = extract_date(text)
        total_amount = extract_total_amount(text)
        all_amounts = extract_all_amounts(text)
        line_items = extract_line_items(text)

        # common fields
        vat = extract_field_amount(text, ["vat"])
        cash_paid = extract_field_amount(text, ["cash paid", "cash"])
        change = extract_field_amount(text, ["change"]) 

        return {
            "merchant": merchant,
            "date": date,
            "amount": total_amount,
            "amounts_found": all_amounts,
            "line_items": line_items,
            "vat": vat,
            "cash_paid": cash_paid,
            "change": change,
            "raw_text": text.strip(),
        }

    except Exception as e:
        logger.exception(f"Failed to extract receipt fields: {e}")

        return {
            "merchant": None,
            "date": None,
            "amount": None,
            "amounts_found": [],
            "line_items": [],
            "raw_text": text.strip(),
            "error": str(e),
        }