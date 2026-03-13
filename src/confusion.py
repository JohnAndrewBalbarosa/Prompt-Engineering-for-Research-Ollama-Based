from __future__ import annotations

from pathlib import Path

from src.config import ConfusionCheckConfig, JudgeConfig
from src.judge_clients import BaseJudgeClient


def make_confusion_client(confusion_config: ConfusionCheckConfig) -> BaseJudgeClient:
    # Reuse the existing judge client transport and retry logic.
    judge_compatible = JudgeConfig(
        enabled=confusion_config.enabled,
        provider=confusion_config.provider,
        model=confusion_config.model,
        temperature=confusion_config.temperature,
        max_tokens=confusion_config.max_tokens,
        timeout_seconds=confusion_config.timeout_seconds,
        retries=confusion_config.retries,
        request_interval_seconds=confusion_config.request_interval_seconds,
        retry_backoff_seconds=confusion_config.retry_backoff_seconds,
        max_retry_backoff_seconds=confusion_config.max_retry_backoff_seconds,
        base_url=confusion_config.base_url,
    )
    from src.judge import make_judge_client

    return make_judge_client(judge_compatible)


def run_confusion_check(
    checker_client: BaseJudgeClient,
    prompt_path: str | Path,
    question: str,
    gold_answer: str | None,
    parsed_answer: str | None,
    model_response: str,
) -> dict:
    template = Path(prompt_path).read_text(encoding="utf-8")
    prompt = template.format(
        question=question,
        gold_answer=gold_answer or "",
        parsed_answer=parsed_answer or "",
        model_response=model_response,
    )

    parsed = checker_client.judge(prompt)
    return {
        "checker_correctness": bool(parsed.get("checker_correctness", False)),
        "checker_explanation": str(parsed.get("explanation", "")),
        "checker_status": "success",
        "checker_error_message": None,
    }
