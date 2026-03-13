from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import List


@dataclass
class DatasetConfig:
    name: str = "gsm8k"
    subset: str = "main"
    split: str = "test"
    sample_size: int | None = None
    seed: int = 42
    retrieval_mode: str = "auto"
    hf_cache_dir: str | None = None
    persist_downloaded_snapshot: bool = True
    server_url: str | None = None
    request_timeout_seconds: int | None = None


@dataclass
class PromptConfig:
    system_prompt: str = "You are a careful mathematical reasoning assistant."
    final_answer_tag: str = "Final Answer:"
    strategies: List[str] = field(default_factory=lambda: ["zero_shot", "chain_of_thought"])


@dataclass
class GenerationConfig:
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_seconds: int = 60
    retries: int = 3
    request_interval_seconds: float = 5.0
    retry_backoff_seconds: float = 10.0
    max_retry_backoff_seconds: float = 60.0
    ollama_base_url: str | None = None


@dataclass
class JudgeConfig:
    enabled: bool = False
    provider: str = "ollama"
    model: str = "llama3.1:8b"
    temperature: float = 0.0
    max_tokens: int = 300
    timeout_seconds: int = 60
    retries: int = 3
    request_interval_seconds: float = 5.0
    retry_backoff_seconds: float = 10.0
    max_retry_backoff_seconds: float = 60.0
    base_url: str | None = None


@dataclass
class ConfusionCheckConfig:
    enabled: bool = True
    provider: str = "ollama"
    model: str = "llama3.1:8b"
    temperature: float = 0.0
    max_tokens: int = 120
    timeout_seconds: int = 60
    retries: int = 3
    request_interval_seconds: float = 2.0
    retry_backoff_seconds: float = 5.0
    max_retry_backoff_seconds: float = 30.0
    base_url: str | None = None
    prompt_path: str = "prompts/confusion_prompt.txt"


@dataclass
class PathsConfig:
    raw_dataset_snapshot: str = "data/raw/gsm8k_snapshot.jsonl"
    raw_generations: str = "results/runs/raw_generations.jsonl"
    parsed_answers: str = "results/runs/parsed_answers.json"
    metrics_summary: str = "results/runs/metrics_summary.json"
    confusion_matrices: str = "results/runs/confusion_matrices.json"
    quantitative_summary: str = "results/runs/quantitative_summary.json"
    quantitative_details: str = "results/runs/quantitative_details.json"
    run_metadata: str = "results/runs/run_metadata.json"


@dataclass
class ModelConfig:
    id: str
    provider: str
    model_name: str
    base_url: str | None = None


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig
    prompts: PromptConfig
    generation: GenerationConfig
    judge: JudgeConfig
    confusion_check: ConfusionCheckConfig
    paths: PathsConfig
    models: List[ModelConfig]


def load_config(config_path: str | Path) -> ExperimentConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            data = json.load(handle)
        else:
            raise ValueError("Only JSON config files are supported in this MVP. Use config/experiment.json.")
    return ExperimentConfig(
        dataset=DatasetConfig(**data["dataset"]),
        prompts=PromptConfig(**data["prompts"]),
        generation=GenerationConfig(**data["generation"]),
        judge=JudgeConfig(**data["judge"]),
        confusion_check=ConfusionCheckConfig(**data.get("confusion_check", {})),
        paths=PathsConfig(**data["paths"]),
        models=[ModelConfig(**model) for model in data["models"]],
    )