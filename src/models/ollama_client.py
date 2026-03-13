from __future__ import annotations

import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.config import GenerationConfig, ModelConfig
from src.env_utils import get_env_str, resolve_setting
from src.models.base import BaseModelClient, GenerationResult
from src.rate_limit import extract_retry_delay_seconds, wait_for_turn


class OllamaModelClient(BaseModelClient):
    def __init__(self, model_config: ModelConfig, generation_config: GenerationConfig) -> None:
        self.model_config = model_config
        self.generation_config = generation_config
        base_url = (
            model_config.base_url
            or generation_config.ollama_base_url
            or get_env_str("OLLAMA_BASE_URL")
        )
        if not base_url:
            raise ValueError(
                "Ollama base URL is not configured. Set OLLAMA_BASE_URL in .env or provide generation.ollama_base_url/model.base_url in config."
            )
        self.base_url = base_url.rstrip("/")
        self.rate_limit_bucket = get_env_str("OLLAMA_GENERATION_BUCKET", "ollama-generation") or "ollama-generation"

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        last_error: Exception | None = None
        for _ in range(self.generation_config.retries):
            try:
                wait_for_turn(
                    self.rate_limit_bucket,
                    resolve_setting(self.generation_config.request_interval_seconds, "OLLAMA_REQUEST_INTERVAL_SECONDS", 0.0),
                )
                started = time.perf_counter()
                payload = {
                    "model": self.model_config.model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": self.generation_config.temperature,
                        "num_predict": self.generation_config.max_tokens,
                    },
                }
                request = Request(
                    f"{self.base_url}/api/chat",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urlopen(request, timeout=self.generation_config.timeout_seconds) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))

                latency_ms = int((time.perf_counter() - started) * 1000)
                message = response_payload.get("message", {}).get("content", "")

                return GenerationResult(
                    model_id=self.model_config.id,
                    provider=self.model_config.provider,
                    response_text=message,
                    latency_ms=latency_ms,
                    status="success",
                    prompt_tokens=response_payload.get("prompt_eval_count"),
                    completion_tokens=response_payload.get("eval_count"),
                )
            except HTTPError as error:
                last_error = error
                if error.code == 429:
                    delay_seconds = extract_retry_delay_seconds(
                        error,
                        resolve_setting(self.generation_config.retry_backoff_seconds, "OLLAMA_RETRY_BACKOFF_SECONDS", 10.0),
                        resolve_setting(self.generation_config.max_retry_backoff_seconds, "OLLAMA_MAX_RETRY_BACKOFF_SECONDS", 60.0),
                    )
                    time.sleep(delay_seconds)
                    continue
            except Exception as error:
                last_error = error

        raise RuntimeError(f"Ollama generation failed after retries: {last_error}")
