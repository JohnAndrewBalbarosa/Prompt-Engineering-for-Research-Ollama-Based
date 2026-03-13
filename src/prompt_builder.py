from __future__ import annotations

from pathlib import Path

from src.config import PromptConfig


TEMPLATE_MAP = {
    "zero_shot": "zero_shot.txt",
    "chain_of_thought": "cot.txt",
    "ablation_no_reasoning": "ablation_no_reasoning.txt",
    "ablation_no_format": "ablation_no_format.txt",
    "question_only": "question_only.txt",
}


def build_prompt(strategy: str, question: str, prompt_config: PromptConfig, prompts_dir: str | Path) -> tuple[str, str]:
    template_name = TEMPLATE_MAP[strategy]
    template_path = Path(prompts_dir) / template_name
    template = template_path.read_text(encoding="utf-8")
    user_prompt = template.format(question=question, final_answer_tag=prompt_config.final_answer_tag)
    return prompt_config.system_prompt, user_prompt