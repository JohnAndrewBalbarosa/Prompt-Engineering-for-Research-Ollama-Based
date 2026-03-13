from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from src.io_utils import ensure_parent_dir


def init_db(db_path: str | Path) -> sqlite3.Connection:
    path = ensure_parent_dir(db_path)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    _create_schema(connection)
    return connection


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            config_path TEXT NOT NULL,
            record_count INTEGER NOT NULL,
            strategies_json TEXT NOT NULL,
            models_json TEXT NOT NULL,
            mode TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS raw_generations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            question TEXT NOT NULL,
            model_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            prompt_strategy TEXT NOT NULL,
            prompt_text TEXT NOT NULL,
            raw_response TEXT,
            status TEXT NOT NULL,
            error_message TEXT,
            latency_ms INTEGER,
            timestamp TEXT NOT NULL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            UNIQUE (run_id, item_id, model_id, prompt_strategy)
        );

        CREATE TABLE IF NOT EXISTS parsed_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            provider TEXT NOT NULL,
            prompt_strategy TEXT NOT NULL,
            status TEXT NOT NULL,
            error_message TEXT,
            latency_ms INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            gold_final_answer TEXT,
            parsed_answer TEXT,
            parse_success INTEGER NOT NULL,
            parse_method TEXT,
            parse_notes TEXT,
            exact_match_label TEXT,
            judge_correctness INTEGER,
            reasoning_score REAL,
            arithmetic_score REAL,
            format_following_score REAL,
            judge_explanation TEXT,
            UNIQUE (run_id, item_id, model_id, prompt_strategy)
        );

        CREATE TABLE IF NOT EXISTS metrics_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            prompt_strategy TEXT NOT NULL,
            total_rows INTEGER NOT NULL,
            success_rows INTEGER NOT NULL,
            error_rows INTEGER NOT NULL,
            judgeable_rows INTEGER NOT NULL,
            judge_missing_rows INTEGER NOT NULL,
            correct_rows INTEGER NOT NULL,
            incorrect_rows INTEGER NOT NULL,
            unparseable_rows INTEGER NOT NULL,
            accuracy REAL NOT NULL,
            parse_failure_rate REAL NOT NULL,
            avg_latency_ms REAL NOT NULL,
            UNIQUE (run_id, model_id, prompt_strategy)
        );

        CREATE TABLE IF NOT EXISTS confusion_matrices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            prompt_strategy TEXT NOT NULL,
            total_rows INTEGER NOT NULL,
            success_rows INTEGER NOT NULL,
            error_rows INTEGER NOT NULL,
            judgeable_rows INTEGER NOT NULL,
            judge_missing_rows INTEGER NOT NULL,
            tn INTEGER NOT NULL,
            fp INTEGER NOT NULL,
            fn INTEGER NOT NULL,
            tp INTEGER NOT NULL,
            precision REAL NOT NULL,
            recall REAL NOT NULL,
            f1 REAL NOT NULL,
            no_data_reason TEXT,
            UNIQUE (run_id, model_id, prompt_strategy)
        );

        CREATE INDEX IF NOT EXISTS idx_parsed_lookup
        ON parsed_results (item_id, model_id, prompt_strategy, status);

        CREATE VIEW IF NOT EXISTS v_results_by_strategy AS
        SELECT
            run_id,
            prompt_strategy,
            model_id,
            item_id,
            status,
            exact_match_label,
            judge_correctness,
            latency_ms,
            parsed_answer
        FROM parsed_results;

        CREATE VIEW IF NOT EXISTS v_confusion_by_strategy AS
        SELECT
            run_id,
            prompt_strategy,
            model_id,
            judgeable_rows,
            judge_missing_rows,
            tn,
            fp,
            fn,
            tp,
            precision,
            recall,
            f1,
            no_data_reason
        FROM confusion_matrices;
        """
    )


def upsert_run(connection: sqlite3.Connection, run_row: dict) -> None:
    connection.execute(
        """
        INSERT INTO runs (run_id, config_path, record_count, strategies_json, models_json, mode)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            config_path = excluded.config_path,
            record_count = excluded.record_count,
            strategies_json = excluded.strategies_json,
            models_json = excluded.models_json,
            mode = excluded.mode
        """,
        (
            run_row["run_id"],
            run_row["config_path"],
            int(run_row["record_count"]),
            json.dumps(run_row["strategies"], ensure_ascii=True),
            json.dumps(run_row["models"], ensure_ascii=True),
            run_row.get("mode", "local-only"),
        ),
    )


def upsert_raw_generation(connection: sqlite3.Connection, row: dict) -> None:
    connection.execute(
        """
        INSERT INTO raw_generations (
            run_id, item_id, question, model_id, provider, prompt_strategy,
            prompt_text, raw_response, status, error_message, latency_ms, timestamp,
            prompt_tokens, completion_tokens
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, item_id, model_id, prompt_strategy) DO UPDATE SET
            question = excluded.question,
            provider = excluded.provider,
            prompt_text = excluded.prompt_text,
            raw_response = excluded.raw_response,
            status = excluded.status,
            error_message = excluded.error_message,
            latency_ms = excluded.latency_ms,
            timestamp = excluded.timestamp,
            prompt_tokens = excluded.prompt_tokens,
            completion_tokens = excluded.completion_tokens
        """,
        (
            row["run_id"],
            row["item_id"],
            row["question"],
            row["model_id"],
            row["provider"],
            row["prompt_strategy"],
            row["prompt_text"],
            row.get("raw_response", ""),
            row["status"],
            row.get("error_message"),
            int(row.get("latency_ms") or 0),
            row["timestamp"],
            row.get("prompt_tokens"),
            row.get("completion_tokens"),
        ),
    )


def _to_db_bool(value: object) -> int | None:
    if value is True:
        return 1
    if value is False:
        return 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return 1
        if lowered == "false":
            return 0
    return None


def upsert_parsed_result(connection: sqlite3.Connection, row: dict) -> None:
    connection.execute(
        """
        INSERT INTO parsed_results (
            run_id, item_id, model_id, provider, prompt_strategy, status,
            error_message, latency_ms, prompt_tokens, completion_tokens,
            gold_final_answer, parsed_answer, parse_success, parse_method, parse_notes,
            exact_match_label, judge_correctness, reasoning_score, arithmetic_score,
            format_following_score, judge_explanation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, item_id, model_id, prompt_strategy) DO UPDATE SET
            provider = excluded.provider,
            status = excluded.status,
            error_message = excluded.error_message,
            latency_ms = excluded.latency_ms,
            prompt_tokens = excluded.prompt_tokens,
            completion_tokens = excluded.completion_tokens,
            gold_final_answer = excluded.gold_final_answer,
            parsed_answer = excluded.parsed_answer,
            parse_success = excluded.parse_success,
            parse_method = excluded.parse_method,
            parse_notes = excluded.parse_notes,
            exact_match_label = excluded.exact_match_label,
            judge_correctness = excluded.judge_correctness,
            reasoning_score = excluded.reasoning_score,
            arithmetic_score = excluded.arithmetic_score,
            format_following_score = excluded.format_following_score,
            judge_explanation = excluded.judge_explanation
        """,
        (
            row["run_id"],
            row["item_id"],
            row["model_id"],
            row["provider"],
            row["prompt_strategy"],
            row["status"],
            row.get("error_message"),
            int(row.get("latency_ms") or 0),
            row.get("prompt_tokens"),
            row.get("completion_tokens"),
            row.get("gold_final_answer"),
            row.get("parsed_answer"),
            1 if row.get("parse_success") else 0,
            row.get("parse_method"),
            row.get("parse_notes"),
            row.get("exact_match_label"),
            _to_db_bool(row.get("judge_correctness")),
            row.get("reasoning_score"),
            row.get("arithmetic_score"),
            row.get("format_following_score"),
            row.get("judge_explanation"),
        ),
    )


def replace_metrics(connection: sqlite3.Connection, run_id: str, rows: Iterable[dict]) -> None:
    connection.execute("DELETE FROM metrics_summary WHERE run_id = ?", (run_id,))
    for row in rows:
        connection.execute(
            """
            INSERT INTO metrics_summary (
                run_id, model_id, prompt_strategy, total_rows, success_rows, error_rows,
                judgeable_rows, judge_missing_rows, correct_rows, incorrect_rows,
                unparseable_rows, accuracy, parse_failure_rate, avg_latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                row["model_id"],
                row["prompt_strategy"],
                int(row["total_rows"]),
                int(row["success_rows"]),
                int(row["error_rows"]),
                int(row["judgeable_rows"]),
                int(row["judge_missing_rows"]),
                int(row["correct_rows"]),
                int(row["incorrect_rows"]),
                int(row["unparseable_rows"]),
                float(row["accuracy"]),
                float(row["parse_failure_rate"]),
                float(row["avg_latency_ms"]),
            ),
        )


def replace_confusion_matrices(connection: sqlite3.Connection, run_id: str, rows: Iterable[dict]) -> None:
    connection.execute("DELETE FROM confusion_matrices WHERE run_id = ?", (run_id,))
    for row in rows:
        connection.execute(
            """
            INSERT INTO confusion_matrices (
                run_id, model_id, prompt_strategy, total_rows, success_rows, error_rows,
                judgeable_rows, judge_missing_rows, tn, fp, fn, tp, precision, recall, f1,
                no_data_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                row["model_id"],
                row["prompt_strategy"],
                int(row["total_rows"]),
                int(row["success_rows"]),
                int(row["error_rows"]),
                int(row["judgeable_rows"]),
                int(row["judge_missing_rows"]),
                int(row["tn"]),
                int(row["fp"]),
                int(row["fn"]),
                int(row["tp"]),
                float(row["precision"]),
                float(row["recall"]),
                float(row["f1"]),
                row.get("no_data_reason"),
            ),
        )


def load_existing_success_keys(connection: sqlite3.Connection) -> set[tuple[str, str, str]]:
    rows = connection.execute(
        """
        SELECT item_id, model_id, prompt_strategy
        FROM parsed_results
        WHERE status = 'success'
        """
    ).fetchall()
    return {(str(row["item_id"]), str(row["model_id"]), str(row["prompt_strategy"])) for row in rows}


def load_all_parsed_rows(connection: sqlite3.Connection) -> list[dict]:
    rows = connection.execute(
        """
        SELECT
            run_id, item_id, model_id, provider, prompt_strategy, status, error_message,
            latency_ms, prompt_tokens, completion_tokens, gold_final_answer, parsed_answer,
            parse_success, parse_method, parse_notes, exact_match_label, judge_correctness,
            reasoning_score, arithmetic_score, format_following_score, judge_explanation
        FROM parsed_results
        """
    ).fetchall()

    parsed_rows: list[dict] = []
    for row in rows:
        parsed_rows.append(
            {
                "run_id": row["run_id"],
                "item_id": row["item_id"],
                "model_id": row["model_id"],
                "provider": row["provider"],
                "prompt_strategy": row["prompt_strategy"],
                "status": row["status"],
                "error_message": row["error_message"],
                "latency_ms": row["latency_ms"],
                "prompt_tokens": row["prompt_tokens"],
                "completion_tokens": row["completion_tokens"],
                "gold_final_answer": row["gold_final_answer"],
                "parsed_answer": row["parsed_answer"],
                "parse_success": bool(row["parse_success"]),
                "parse_method": row["parse_method"],
                "parse_notes": row["parse_notes"],
                "exact_match_label": row["exact_match_label"],
                "judge_correctness": None if row["judge_correctness"] is None else bool(row["judge_correctness"]),
                "reasoning_score": row["reasoning_score"],
                "arithmetic_score": row["arithmetic_score"],
                "format_following_score": row["format_following_score"],
                "judge_explanation": row["judge_explanation"],
            }
        )
    return parsed_rows
