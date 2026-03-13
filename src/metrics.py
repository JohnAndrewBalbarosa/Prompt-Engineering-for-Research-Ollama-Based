from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import numpy as np

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


def _compute_confusion_stats(tp: int, fp: int, fn: int, tn: int) -> dict:
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    npv = _safe_divide(tn, tn + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall) if precision or recall else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)

    matrix = np.array([[tn, fp], [fn, tp]], dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "npv": npv,
        "balanced_accuracy": balanced_accuracy,
        "confusion_matrix": matrix,
        "confusion_matrix_normalized": normalized,
        "norm_tn": float(normalized[0, 0]),
        "norm_fp": float(normalized[0, 1]),
        "norm_fn": float(normalized[1, 0]),
        "norm_tp": float(normalized[1, 1]),
    }


def _aggregate_quant_rows(quantitative_rows: list[dict]) -> list[dict]:
    scored = [row for row in quantitative_rows if row.get("aggregate_type") == "group" and int(row.get("judgeable_rows", 0)) > 0]
    if not scored:
        return []

    macro = {
        "model_id": "__all__",
        "prompt_strategy": "__all__",
        "aggregate_type": "macro",
        "judgeable_rows": sum(int(row.get("judgeable_rows", 0)) for row in scored),
        "tn": float(np.mean([float(row["tn"]) for row in scored])),
        "fp": float(np.mean([float(row["fp"]) for row in scored])),
        "fn": float(np.mean([float(row["fn"]) for row in scored])),
        "tp": float(np.mean([float(row["tp"]) for row in scored])),
        "precision": float(np.mean([float(row["precision"]) for row in scored])),
        "recall": float(np.mean([float(row["recall"]) for row in scored])),
        "f1": float(np.mean([float(row["f1"]) for row in scored])),
        "specificity": float(np.mean([float(row["specificity"]) for row in scored])),
        "npv": float(np.mean([float(row["npv"]) for row in scored])),
        "balanced_accuracy": float(np.mean([float(row["balanced_accuracy"]) for row in scored])),
        "norm_tn": float(np.mean([float(row["norm_tn"]) for row in scored])),
        "norm_fp": float(np.mean([float(row["norm_fp"]) for row in scored])),
        "norm_fn": float(np.mean([float(row["norm_fn"]) for row in scored])),
        "norm_tp": float(np.mean([float(row["norm_tp"]) for row in scored])),
    }

    tp = int(sum(int(row["tp"]) for row in scored))
    fp = int(sum(int(row["fp"]) for row in scored))
    fn = int(sum(int(row["fn"]) for row in scored))
    tn = int(sum(int(row["tn"]) for row in scored))
    micro_stats = _compute_confusion_stats(tp=tp, fp=fp, fn=fn, tn=tn)
    micro = {
        "model_id": "__all__",
        "prompt_strategy": "__all__",
        "aggregate_type": "micro",
        "judgeable_rows": int(sum(int(row.get("judgeable_rows", 0)) for row in scored)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": micro_stats["precision"],
        "recall": micro_stats["recall"],
        "f1": micro_stats["f1"],
        "specificity": micro_stats["specificity"],
        "npv": micro_stats["npv"],
        "balanced_accuracy": micro_stats["balanced_accuracy"],
        "norm_tn": micro_stats["norm_tn"],
        "norm_fp": micro_stats["norm_fp"],
        "norm_fn": micro_stats["norm_fn"],
        "norm_tp": micro_stats["norm_tp"],
    }
    return [macro, micro]


def compute_metrics(results_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    if not results_rows:
        return [], [], []

    latest_results = _latest_rows(results_rows)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in latest_results:
        grouped[(row["model_id"], row["prompt_strategy"])].append(row)

    summary_rows: list[dict] = []
    confusion_rows: list[dict] = []
    quantitative_rows: list[dict] = []
    for (model_id, prompt_strategy), group in grouped.items():
        total_rows = len(group)
        successful = [row for row in group if row.get("status") == "success"]
        error_rows = [row for row in group if row.get("status") != "success"]
        judge_missing_rows = [
            row
            for row in successful
            if _to_bool(row.get("checker_correctness")) is None and _to_bool(row.get("judge_correctness")) is None
        ]
        correct_rows = sum(1 for row in successful if row.get("exact_match_label") == "correct")
        incorrect_rows = sum(1 for row in successful if row.get("exact_match_label") == "incorrect")
        unparseable_rows = sum(1 for row in successful if row.get("exact_match_label") == "unparseable")
        latency_values = [int(row.get("latency_ms", 0) or 0) for row in successful]
        judgeable_rows = [
            row
            for row in successful
            if _to_bool(row.get("checker_correctness")) is not None or _to_bool(row.get("judge_correctness")) is not None
        ]

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
            checker_value = _to_bool(row.get("checker_correctness"))
            predicted = bool(checker_value) if checker_value is not None else bool(_to_bool(row.get("judge_correctness")))
            if actual and predicted:
                tp += 1
            elif actual and not predicted:
                fn += 1
            elif not actual and predicted:
                fp += 1
            else:
                tn += 1

        stats = _compute_confusion_stats(tp=tp, fp=fp, fn=fn, tn=tn)
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
                "precision": stats["precision"],
                "recall": stats["recall"],
                "f1": stats["f1"],
                "no_data_reason": no_data_reason,
            }
        )

        quantitative_rows.append(
            {
                "model_id": model_id,
                "prompt_strategy": prompt_strategy,
                "aggregate_type": "group",
                "judgeable_rows": len(judgeable_rows),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "precision": stats["precision"],
                "recall": stats["recall"],
                "f1": stats["f1"],
                "specificity": stats["specificity"],
                "npv": stats["npv"],
                "balanced_accuracy": stats["balanced_accuracy"],
                "norm_tn": stats["norm_tn"],
                "norm_fp": stats["norm_fp"],
                "norm_fn": stats["norm_fn"],
                "norm_tp": stats["norm_tp"],
            }
        )

    quantitative_rows.extend(_aggregate_quant_rows(quantitative_rows))

    return summary_rows, confusion_rows, quantitative_rows


def export_metrics(
    results_rows: list[dict],
    metrics_path: str | Path,
    confusion_path: str | Path,
    quantitative_summary_path: str | Path,
    quantitative_details_path: str | Path,
) -> None:
    summary_rows, confusion_rows, quantitative_rows = compute_metrics(results_rows)
    if not summary_rows and not confusion_rows and not quantitative_rows:
        write_csv(metrics_path, [])
        write_json(confusion_path, [])
        write_csv(quantitative_summary_path, [])
        write_json(quantitative_details_path, [])
        return

    write_csv(metrics_path, summary_rows)
    write_json(confusion_path, confusion_rows)
    write_csv(quantitative_summary_path, quantitative_rows)
    write_json(quantitative_details_path, quantitative_rows)