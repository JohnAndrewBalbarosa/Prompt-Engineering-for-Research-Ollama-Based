from __future__ import annotations

from abc import ABC, abstractmethod
import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from src.config import JudgeConfig
from src.env_utils import get_env_str, resolve_setting
from src.rate_limit import extract_retry_delay_seconds, wait_for_turn


def _parse_json_object(content: str) -> dict:
    stripped = content.strip()
    if not stripped:
        return {}

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and start < end:
        try:
            parsed = json.loads(stripped[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


class BaseJudgeClient(ABC):
    def __init__(self, judge_config: JudgeConfig) -> None:
        self.judge_config = judge_config

    @abstractmethod
    def judge(self, prompt: str) -> dict:
        raise NotImplementedError


class OllamaJudgeClient(BaseJudgeClient):
    def __init__(self, judge_config: JudgeConfig) -> None:
        super().__init__(judge_config)
        base_url = (
            judge_config.base_url
            or get_env_str("OLLAMA_BASE_URL")
        )
        if not base_url:
            raise ValueError(
                "Ollama base URL is not configured. Set OLLAMA_BASE_URL in .env or provide judge.base_url in config."
            )
        self.base_url = base_url.rstrip("/")
        self.rate_limit_bucket = get_env_str("OLLAMA_JUDGE_BUCKET", "ollama-judge") or "ollama-judge"

    def judge(self, prompt: str) -> dict:
        payload = {
            "model": self.judge_config.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": self.judge_config.temperature,
                "num_predict": self.judge_config.max_tokens,
            },
            "format": "json",
        }
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        for _ in range(self.judge_config.retries):
            try:
                wait_for_turn(
                    self.rate_limit_bucket,
                    resolve_setting(self.judge_config.request_interval_seconds, "OLLAMA_JUDGE_REQUEST_INTERVAL_SECONDS", 0.0),
                )
                with urlopen(request, timeout=self.judge_config.timeout_seconds) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                content = response_payload.get("message", {}).get("content", "{}")
                return _parse_json_object(content)
            except HTTPError as error:
                if error.code == 429:
                    delay_seconds = extract_retry_delay_seconds(
                        error,
                        resolve_setting(self.judge_config.retry_backoff_seconds, "OLLAMA_JUDGE_RETRY_BACKOFF_SECONDS", 10.0),
                        resolve_setting(
                            self.judge_config.max_retry_backoff_seconds,
                            "OLLAMA_JUDGE_MAX_RETRY_BACKOFF_SECONDS",
                            60.0,
                        ),
                    )
                    time.sleep(delay_seconds)
                    continue
                raise

        raise RuntimeError("Ollama judge failed after retries.")


def build_judge_client(judge_config: JudgeConfig) -> BaseJudgeClient:
    provider = judge_config.provider.strip().lower()
    if provider == "ollama":
        return OllamaJudgeClient(judge_config)
    raise NotImplementedError(
        f"Judge provider '{judge_config.provider}' is not supported in local-only mode. Use provider='ollama'."
    )
