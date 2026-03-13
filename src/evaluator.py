from __future__ import annotations


def evaluate_exact_match(parsed_answer: str | None, gold_answer: str | None) -> str:
    if parsed_answer is None:
        return "unparseable"
    if gold_answer is None:
        return "unscorable"
    if parsed_answer == gold_answer:
        return "correct"
    return "incorrect"