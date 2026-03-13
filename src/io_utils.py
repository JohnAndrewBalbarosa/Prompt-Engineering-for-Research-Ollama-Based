from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def ensure_parent_dir(file_path: str | Path) -> Path:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def append_jsonl(file_path: str | Path, record: dict) -> None:
    path = ensure_parent_dir(file_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_json(file_path: str | Path, data: dict | list) -> None:
    path = ensure_parent_dir(file_path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)


def write_csv(file_path: str | Path, rows: Iterable[dict]) -> None:
    path = ensure_parent_dir(file_path)
    rows = list(rows)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_csv_if_exists(file_path: str | Path) -> list[dict]:
    path = Path(file_path)
    if not path.exists() or path.stat().st_size == 0:
        return []

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))