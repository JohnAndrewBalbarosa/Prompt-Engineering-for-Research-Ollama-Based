from __future__ import annotations

import json
from pathlib import Path


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


def load_json_if_exists(file_path: str | Path) -> list[dict]:
    path = ensure_parent_dir(file_path)
    if not path.exists() or path.stat().st_size == 0:
        return []

    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
        if isinstance(loaded, list):
            return loaded
    return []