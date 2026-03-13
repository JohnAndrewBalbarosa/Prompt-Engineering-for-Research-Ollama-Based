from __future__ import annotations

import argparse
import logging
from datetime import datetime, UTC
from pathlib import Path
import sqlite3

from src.answer_parser import extract_model_answer
from src.config import ExperimentConfig, ModelConfig, load_config
from src.confusion import make_confusion_client, run_confusion_check
from src.db_utils import (
    init_db,
    load_all_parsed_rows,
    load_existing_success_keys,
    replace_checker_results,
    replace_confusion_matrices,
    replace_metrics,
    replace_quantitative_summary,
    upsert_parsed_result,
    upsert_raw_generation,
    upsert_run,
)
from src.dataset_loader import load_gsm8k_records
from src.env_utils import get_env_str
from src.evaluator import evaluate_exact_match
from src.io_utils import append_jsonl, ensure_parent_dir, load_csv_if_exists, write_csv, write_json
from src.judge import judge_response, make_judge_client
from src.metrics import compute_metrics, export_metrics
from src.models.base import BaseModelClient, GenerationResult
from src.models.ollama_client import OllamaModelClient
from src.prompt_builder import build_prompt


LOGGER = logging.getLogger("experiment_runner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GSM8K prompt-engineering experiment.")
    parser.add_argument(
        "--config",
        default=get_env_str("EXPERIMENT_CONFIG_PATH", "config/experiment.json"),
        help="Path to the experiment JSON config.",
    )
    parser.add_argument("--model", default=None, help="Optional model id filter.")
    parser.add_argument("--strategy", default=None, help="Optional prompt strategy filter.")
    parser.add_argument(
        "--dataset-source",
        choices=["local", "auto", "remote"],
        default=None,
        help="Dataset retrieval mode override. local=only snapshot, auto=fallback to remote, remote=always fetch.",
    )
    parser.add_argument("--dataset-split", default=None, help="Dataset split override (for example: test or train).")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional dataset sample size override.")
    parser.add_argument(
        "--interactive-dataset-source",
        action="store_true",
        help="Prompt for dataset source mode when --dataset-source is not provided.",
    )
    parser.add_argument(
        "--storage",
        choices=["sql", "parallel", "file"],
        default="sql",
        help="Storage mode: sql (SQLite only), parallel (SQLite + files), file (CSV/JSON only).",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def build_client(model_config: ModelConfig, config: ExperimentConfig) -> BaseModelClient:
    provider = model_config.provider.strip().lower()
    if provider == "ollama":
        return OllamaModelClient(model_config, config.generation)
    raise NotImplementedError(
        f"Provider '{model_config.provider}' is not supported in local-only mode. Use provider='ollama'."
    )


def completed_keys(existing_results: list[dict]) -> set[tuple[str, str, str]]:
    return {
        (str(row["item_id"]), str(row["model_id"]), str(row["prompt_strategy"]))
        for row in existing_results
        if row.get("item_id") and row.get("model_id") and row.get("prompt_strategy") and row.get("status") == "success"
    }


def generation_failure(model_config: ModelConfig, error: Exception) -> GenerationResult:
    return GenerationResult(
        model_id=model_config.id,
        provider=model_config.provider,
        response_text="",
        latency_ms=0,
        status="error",
        error_message=str(error),
    )


def _prompt_dataset_source() -> str:
    valid = {"local", "auto", "remote"}
    while True:
        selected = input("Dataset source [local/auto/remote]: ").strip().lower()
        if selected in valid:
            return selected
        print("Invalid choice. Enter one of: local, auto, remote.")


def _apply_runtime_overrides(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if args.dataset_split:
        config.dataset.split = args.dataset_split
    if args.sample_size is not None:
        config.dataset.sample_size = args.sample_size

    if args.dataset_source:
        config.dataset.retrieval_mode = args.dataset_source
    elif args.interactive_dataset_source:
        config.dataset.retrieval_mode = _prompt_dataset_source()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    _apply_runtime_overrides(config, args)
    project_root = Path(args.config).resolve().parent.parent
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    connection: sqlite3.Connection | None = None
    if args.storage in {"sql", "parallel"}:
        connection = init_db(project_root / config.paths.database)

    records = load_gsm8k_records(config.dataset, project_root / config.paths.raw_dataset_snapshot)
    if connection is not None:
        existing_results = load_all_parsed_rows(connection)
        seen = load_existing_success_keys(connection)
    else:
        existing_results = load_csv_if_exists(project_root / config.paths.parsed_answers)
        seen = completed_keys(existing_results)
    collected_rows = list(existing_results)
    run_rows: list[dict] = []

    for model_config in config.models:
        if args.model and model_config.id != args.model:
            continue

        client = build_client(model_config, config)
        judge_client = make_judge_client(config.judge) if config.judge.enabled else None
        checker_client = make_confusion_client(config.confusion_check) if config.confusion_check.enabled else None
        for strategy in config.prompts.strategies:
            if args.strategy and strategy != args.strategy:
                continue

            LOGGER.info("Running model=%s strategy=%s", model_config.id, strategy)
            for record in records:
                key = (record["item_id"], model_config.id, strategy)
                if key in seen:
                    continue

                system_prompt, user_prompt = build_prompt(strategy, record["question"], config.prompts, project_root / "prompts")
                try:
                    generation = client.generate(system_prompt=system_prompt, user_prompt=user_prompt)
                except Exception as error:
                    generation = generation_failure(model_config, error)

                raw_generation_record = {
                    "run_id": run_id,
                    "item_id": record["item_id"],
                    "question": record["question"],
                    "model_id": model_config.id,
                    "provider": model_config.provider,
                    "prompt_strategy": strategy,
                    "prompt_text": user_prompt,
                    "raw_response": generation.response_text,
                    "status": generation.status,
                    "error_message": generation.error_message,
                    "latency_ms": generation.latency_ms,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "prompt_tokens": generation.prompt_tokens,
                    "completion_tokens": generation.completion_tokens,
                }
                if args.storage in {"file", "parallel"}:
                    append_jsonl(project_root / config.paths.raw_generations, raw_generation_record)
                    append_jsonl(
                        ensure_parent_dir(project_root / "results" / "runs" / "by_strategy" / strategy / "raw_generations.jsonl"),
                        raw_generation_record,
                    )
                if connection is not None:
                    upsert_raw_generation(connection, raw_generation_record)

                parsed = extract_model_answer(generation.response_text, config.prompts.final_answer_tag)
                exact_match_label = evaluate_exact_match(parsed.value, record["gold_final_answer"])

                judge_payload = {
                    "judge_correctness": None,
                    "reasoning_score": None,
                    "arithmetic_score": None,
                    "format_following_score": None,
                    "judge_explanation": None,
                }
                if config.judge.enabled and generation.status == "success":
                    try:
                        judge_payload = judge_response(
                            judge_client=judge_client,
                            prompt_path=project_root / "prompts" / "judge_prompt.txt",
                            question=record["question"],
                            gold_answer=record["gold_final_answer"],
                            model_response=generation.response_text,
                        )
                    except Exception as error:
                        LOGGER.warning("Judge failed for item=%s model=%s strategy=%s: %s", record["item_id"], model_config.id, strategy, error)

                checker_payload = {
                    "checker_correctness": None,
                    "checker_explanation": None,
                    "checker_status": "skipped",
                    "checker_error_message": None,
                }
                if config.confusion_check.enabled and generation.status == "success" and checker_client is not None:
                    try:
                        checker_payload = run_confusion_check(
                            checker_client=checker_client,
                            prompt_path=project_root / config.confusion_check.prompt_path,
                            question=record["question"],
                            gold_answer=record["gold_final_answer"],
                            parsed_answer=parsed.value,
                            model_response=generation.response_text,
                        )
                    except Exception as error:
                        checker_payload = {
                            "checker_correctness": None,
                            "checker_explanation": None,
                            "checker_status": "error",
                            "checker_error_message": str(error),
                        }
                        LOGGER.warning(
                            "Confusion check failed for item=%s model=%s strategy=%s: %s",
                            record["item_id"],
                            model_config.id,
                            strategy,
                            error,
                        )

                result_row = {
                    "run_id": run_id,
                    "item_id": record["item_id"],
                    "model_id": model_config.id,
                    "provider": model_config.provider,
                    "prompt_strategy": strategy,
                    "status": generation.status,
                    "error_message": generation.error_message,
                    "latency_ms": generation.latency_ms,
                    "prompt_tokens": generation.prompt_tokens,
                    "completion_tokens": generation.completion_tokens,
                    "gold_final_answer": record["gold_final_answer"],
                    "parsed_answer": parsed.value,
                    "parse_success": parsed.success,
                    "parse_method": parsed.method,
                    "parse_notes": parsed.notes,
                    "exact_match_label": exact_match_label,
                    **judge_payload,
                    **checker_payload,
                }
                collected_rows.append(result_row)
                run_rows.append(result_row)
                if connection is not None:
                    upsert_parsed_result(connection, result_row)
                seen.add(key)

    summary_rows, confusion_rows, quantitative_rows = compute_metrics(run_rows)

    if args.storage in {"file", "parallel"}:
        write_csv(project_root / config.paths.parsed_answers, collected_rows)
        export_metrics(
            run_rows,
            project_root / config.paths.metrics_summary,
            project_root / config.paths.confusion_matrices,
            project_root / config.paths.quantitative_summary,
            project_root / config.paths.quantitative_details,
        )
        for strategy in config.prompts.strategies:
            strategy_rows = [row for row in collected_rows if row.get("prompt_strategy") == strategy]
            strategy_dir = project_root / "results" / "runs" / "by_strategy" / strategy
            write_csv(ensure_parent_dir(strategy_dir / "parsed_answers.csv"), strategy_rows)
            export_metrics(
                strategy_rows,
                ensure_parent_dir(strategy_dir / "metrics_summary.csv"),
                ensure_parent_dir(strategy_dir / "confusion_matrices.json"),
                ensure_parent_dir(strategy_dir / "quantitative_summary.csv"),
                ensure_parent_dir(strategy_dir / "quantitative_details.json"),
            )
    providers_in_use = {model.provider for model in config.models}
    if config.judge.enabled:
        providers_in_use.add(config.judge.provider)

    api_keys_present = {"ollama": True}

    run_metadata = {
        "run_id": run_id,
        "config_path": str(Path(args.config)),
        "record_count": len(records),
        "strategies": config.prompts.strategies,
        "models": [model.__dict__ for model in config.models],
        "api_keys_present": api_keys_present,
        "mode": "local-first",
        "storage_mode": args.storage,
        "dataset_retrieval_mode": config.dataset.retrieval_mode,
        "confusion_check_enabled": config.confusion_check.enabled,
        "confusion_check_model": config.confusion_check.model,
    }

    if args.storage in {"file", "parallel"}:
        write_json(project_root / config.paths.run_metadata, run_metadata)

    if connection is not None:
        upsert_run(connection, run_metadata)
        replace_metrics(connection, run_id, summary_rows)
        replace_confusion_matrices(connection, run_id, confusion_rows)
        replace_checker_results(connection, run_id, run_rows)
        replace_quantitative_summary(connection, run_id, quantitative_rows)
        connection.commit()
        connection.close()

    LOGGER.info("Experiment run complete.")


if __name__ == "__main__":
    main()