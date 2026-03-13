from __future__ import annotations

from pathlib import Path

from src.config import JudgeConfig
from src.judge_clients import BaseJudgeClient, build_judge_client


def judge_response(
    judge_client: BaseJudgeClient,
    prompt_path: str | Path,
    question: str,
    gold_answer: str | None,
    model_response: str,
) -> dict:
    template = Path(prompt_path).read_text(encoding="utf-8")
    prompt = template.format(
        question=question,
        gold_answer=gold_answer or "",
        model_response=model_response,
    )

    parsed = judge_client.judge(prompt)

    return {
        "judge_correctness": bool(parsed.get("judge_correctness", False)),
        "reasoning_score": parsed.get("reasoning_score"),
        "arithmetic_score": parsed.get("arithmetic_score"),
        "format_following_score": parsed.get("format_following_score"),
        "judge_explanation": parsed.get("explanation", ""),
    }


def make_judge_client(judge_config: JudgeConfig) -> BaseJudgeClient:
    return build_judge_client(judge_config)