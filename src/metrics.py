from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from src.io_utils import write_csv, write_json


def _to_bool(value: object) -> bool | None:
    if value in (True, False):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _latest_rows(results_rows: list[dict]) -> list[dict]:
    latest_by_key: dict[tuple[str, str, str], dict] = {}
    ordered_keys: list[tuple[str, str, str]] = []

    for row in results_rows:
        key = (str(row.get("item_id", "")), str(row.get("model_id", "")), str(row.get("prompt_strategy", "")))
        if not all(key):
            continue
        if key not in latest_by_key:
            ordered_keys.append(key)
        latest_by_key[key] = row

    return [latest_by_key[key] for key in ordered_keys]


def compute_metrics(results_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    if not results_rows:
        return [], []

    latest_results = _latest_rows(results_rows)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in latest_results:
        grouped[(row["model_id"], row["prompt_strategy"])].append(row)

    summary_rows: list[dict] = []
    confusion_rows: list[dict] = []
    for (model_id, prompt_strategy), group in grouped.items():
        total_rows = len(group)
        successful = [row for row in group if row.get("status") == "success"]
        error_rows = [row for row in group if row.get("status") != "success"]
        judge_missing_rows = [row for row in successful if _to_bool(row.get("judge_correctness")) is None]
        correct_rows = sum(1 for row in successful if row.get("exact_match_label") == "correct")
        incorrect_rows = sum(1 for row in successful if row.get("exact_match_label") == "incorrect")
        unparseable_rows = sum(1 for row in successful if row.get("exact_match_label") == "unparseable")
        latency_values = [int(row.get("latency_ms", 0) or 0) for row in successful]
        judgeable_rows = [row for row in successful if _to_bool(row.get("judge_correctness")) is not None]

        summary_rows.append(
            {
                "model_id": model_id,
                "prompt_strategy": prompt_strategy,
                "total_rows": total_rows,
                "success_rows": len(successful),
                "error_rows": len(error_rows),
                "judgeable_rows": len(judgeable_rows),
                "judge_missing_rows": len(judge_missing_rows),
                "correct_rows": correct_rows,
                "incorrect_rows": incorrect_rows,
                "unparseable_rows": unparseable_rows,
                "accuracy": _safe_divide(correct_rows, len(successful)),
                "parse_failure_rate": _safe_divide(unparseable_rows, len(successful)),
                "avg_latency_ms": _safe_divide(sum(latency_values), len(latency_values)),
            }
        )

        tp = fp = tn = fn = 0
        for row in judgeable_rows:
            actual = row.get("exact_match_label") == "correct"
            predicted = bool(_to_bool(row.get("judge_correctness")))
            if actual and predicted:
                tp += 1
            elif actual and not predicted:
                fn += 1
            elif not actual and predicted:
                fp += 1
            else:
                tn += 1

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall) if precision or recall else 0.0
        no_data_reason = None
        if not judgeable_rows:
            if successful and judge_missing_rows:
                no_data_reason = "No judgeable rows because the judge did not return a valid correctness value for successful generations."
            elif error_rows:
                no_data_reason = "No judgeable rows because generation failed before judging."
            else:
                no_data_reason = "No judgeable rows were available for this model and strategy."

        confusion_rows.append(
            {
                "model_id": model_id,
                "prompt_strategy": prompt_strategy,
                "total_rows": total_rows,
                "success_rows": len(successful),
                "error_rows": len(error_rows),
                "judgeable_rows": len(judgeable_rows),
                "judge_missing_rows": len(judge_missing_rows),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "no_data_reason": no_data_reason,
            }
        )

    return summary_rows, confusion_rows


def export_metrics(results_rows: list[dict], metrics_path: str | Path, confusion_path: str | Path) -> None:
    summary_rows, confusion_rows = compute_metrics(results_rows)
    if not summary_rows and not confusion_rows:
        write_csv(metrics_path, [])
        write_json(confusion_path, [])
        return

    write_csv(metrics_path, summary_rows)
    write_json(confusion_path, confusion_rows)