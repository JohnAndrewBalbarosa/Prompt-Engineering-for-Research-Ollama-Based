from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List

from src.answer_parser import extract_gold_answer
from src.config import DatasetConfig


def _normalize_snapshot_row(row: dict, index: int, split: str) -> dict:
    if "question" in row and "gold_final_answer" in row:
        return {
            "item_id": str(row.get("item_id") or f"{split}-{index}"),
            "question": row["question"],
            "raw_answer": row.get("raw_answer", ""),
            "gold_final_answer": str(row["gold_final_answer"]),
            "gold_parse_success": bool(row.get("gold_parse_success", True)),
            "split": str(row.get("split") or split),
        }

    if "question" in row and "answer" in row:
        gold_answer = extract_gold_answer(row["answer"])
        return {
            "item_id": str(row.get("item_id") or f"{split}-{index}"),
            "question": row["question"],
            "raw_answer": row["answer"],
            "gold_final_answer": gold_answer.value,
            "gold_parse_success": gold_answer.success,
            "split": str(row.get("split") or split),
        }

    raise ValueError("Snapshot row is missing required fields. Expected question + gold_final_answer (or answer).")


def load_gsm8k_records(dataset_config: DatasetConfig, snapshot_path: str | Path) -> List[dict]:
    snapshot_file = Path(snapshot_path)
    if not snapshot_file.exists():
        raise FileNotFoundError(
            f"Snapshot file not found at {snapshot_file}. Local-only mode requires a local dataset snapshot."
        )

    raw_rows: List[dict] = []
    with snapshot_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw_rows.append(json.loads(line))

    if dataset_config.sample_size is not None and len(raw_rows) > dataset_config.sample_size:
        random.Random(dataset_config.seed).shuffle(raw_rows)
        raw_rows = raw_rows[: dataset_config.sample_size]

    records: List[dict] = []
    for index, row in enumerate(raw_rows):
        records.append(_normalize_snapshot_row(row, index=index, split=dataset_config.split))

    return records