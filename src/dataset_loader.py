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


def _read_local_snapshot_rows(snapshot_path: Path, split: str) -> List[dict]:
    if not snapshot_path.exists():
        return []

    rows: List[dict] = []
    with snapshot_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row_split = str(row.get("split") or split)
            if split and row.get("split") and row_split != split:
                continue
            rows.append(row)
    return rows


def _fetch_remote_rows(dataset_config: DatasetConfig) -> List[dict]:
    try:
        from datasets import load_dataset
    except Exception as error:
        raise RuntimeError(
            "Automatic dataset retrieval requires the 'datasets' package. Install dependencies and retry."
        ) from error

    cache_dir = Path(dataset_config.hf_cache_dir) if dataset_config.hf_cache_dir else None
    dataset = load_dataset(
        dataset_config.name,
        dataset_config.subset,
        split=dataset_config.split,
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    rows: List[dict] = []
    for index, row in enumerate(dataset):
        rows.append(
            {
                "item_id": f"{dataset_config.split}-{index}",
                "question": row["question"],
                "answer": row["answer"],
                "split": dataset_config.split,
            }
        )
    return rows


def _deduplicate_rows(rows: List[dict]) -> List[dict]:
    deduped: List[dict] = []
    seen_questions: set[str] = set()
    for row in rows:
        question = str(row.get("question", "")).strip()
        if not question or question in seen_questions:
            continue
        seen_questions.add(question)
        deduped.append(row)
    return deduped


def _apply_sampling(rows: List[dict], sample_size: int | None, seed: int) -> List[dict]:
    if sample_size is None or len(rows) <= sample_size:
        return rows
    sampled = list(rows)
    random.Random(seed).shuffle(sampled)
    return sampled[:sample_size]


def _persist_snapshot(snapshot_path: Path, rows: List[dict]) -> None:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with snapshot_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_gsm8k_records(dataset_config: DatasetConfig, snapshot_path: str | Path) -> List[dict]:
    snapshot_file = Path(snapshot_path)
    retrieval_mode = (dataset_config.retrieval_mode or "auto").strip().lower()
    if retrieval_mode not in {"local", "auto", "remote"}:
        raise ValueError("dataset.retrieval_mode must be one of: local, auto, remote")

    local_rows = _read_local_snapshot_rows(snapshot_file, dataset_config.split)
    raw_rows: List[dict]

    if retrieval_mode == "local":
        if not local_rows:
            raise FileNotFoundError(
                f"Snapshot file not found or empty at {snapshot_file}. Local mode requires a local dataset snapshot."
            )
        raw_rows = local_rows
    elif retrieval_mode == "remote":
        raw_rows = _fetch_remote_rows(dataset_config)
    else:
        needs_more_rows = dataset_config.sample_size is not None and len(local_rows) < dataset_config.sample_size
        if local_rows and not needs_more_rows:
            raw_rows = local_rows
        else:
            try:
                remote_rows = _fetch_remote_rows(dataset_config)
                raw_rows = _deduplicate_rows(local_rows + remote_rows)
            except Exception:
                if not local_rows:
                    raise
                raw_rows = local_rows

    raw_rows = _deduplicate_rows(raw_rows)
    raw_rows = _apply_sampling(raw_rows, dataset_config.sample_size, dataset_config.seed)

    # Persist the effective working snapshot so local snapshot size matches configured sampling.
    if dataset_config.persist_downloaded_snapshot:
        _persist_snapshot(snapshot_file, raw_rows)

    records: List[dict] = []
    for index, row in enumerate(raw_rows):
        records.append(_normalize_snapshot_row(row, index=index, split=dataset_config.split))

    return records