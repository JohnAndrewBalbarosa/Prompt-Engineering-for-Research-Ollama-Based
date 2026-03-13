from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    model_id: str
    provider: str
    response_text: str
    latency_ms: int
    status: str
    error_message: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class BaseModelClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        raise NotImplementedError