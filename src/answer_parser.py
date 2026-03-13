from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


NUMBER_PATTERN = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")


@dataclass
class ParseResult:
    value: str | None
    success: bool
    method: str
    notes: str = ""


def normalize_answer(raw_value: str | None) -> str | None:
    if raw_value is None:
        return None

    cleaned = raw_value.strip()
    if not cleaned:
        return None

    cleaned = cleaned.replace(",", "").replace("$", "")
    cleaned = cleaned.rstrip(". ")

    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return cleaned.lower()

    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def _extract_last_number(text: str) -> str | None:
    matches = NUMBER_PATTERN.findall(text)
    return matches[-1] if matches else None


def extract_gold_answer(answer_text: str) -> ParseResult:
    marker_match = re.search(r"####\s*(.+)$", answer_text, flags=re.MULTILINE)
    if marker_match:
        normalized = normalize_answer(marker_match.group(1))
        return ParseResult(value=normalized, success=normalized is not None, method="gsm8k_marker")

    last_number = _extract_last_number(answer_text)
    normalized = normalize_answer(last_number)
    return ParseResult(
        value=normalized,
        success=normalized is not None,
        method="last_number",
        notes="Fell back to the last numeric expression in the GSM8K answer.",
    )


def extract_model_answer(response_text: str, final_answer_tag: str = "Final Answer:") -> ParseResult:
    pattern = rf"^{re.escape(final_answer_tag)}\s*(.+)$"
    explicit_match = re.search(pattern, response_text, flags=re.MULTILINE | re.IGNORECASE)
    if explicit_match:
        normalized = normalize_answer(explicit_match.group(1))
        return ParseResult(value=normalized, success=normalized is not None, method="final_answer_tag")

    boxed_match = re.search(r"\\boxed\{([^}]+)\}", response_text)
    if boxed_match:
        normalized = normalize_answer(boxed_match.group(1))
        return ParseResult(value=normalized, success=normalized is not None, method="boxed_answer")

    last_number = _extract_last_number(response_text)
    normalized = normalize_answer(last_number)
    return ParseResult(
        value=normalized,
        success=normalized is not None,
        method="last_number",
        notes="Fell back to the last numeric expression in the model response.",
    )