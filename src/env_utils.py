from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, TypeVar


T = TypeVar("T")


def get_env_value(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value

    for file_name in (".env", ".env.example"):
        path = Path.cwd() / file_name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            if key.strip() == name:
                return raw_value.strip().strip('"').strip("'")
    return None


def get_env_str(name: str, default: str | None = None) -> str | None:
    value = get_env_value(name)
    if value is None or value == "":
        return default
    return value


def _coerce_env_value(name: str, value: str, caster: Callable[[str], T]) -> T:
    try:
        return caster(value)
    except Exception as error:
        raise ValueError(f"Invalid value for environment variable '{name}': {value}") from error


def get_env_int(name: str, default: int | None = None) -> int | None:
    value = get_env_value(name)
    if value is None or value == "":
        return default
    return _coerce_env_value(name, value, int)


def get_env_float(name: str, default: float | None = None) -> float | None:
    value = get_env_value(name)
    if value is None or value == "":
        return default
    return _coerce_env_value(name, value, float)


def resolve_setting(config_value: T | None, env_name: str, default: T) -> T:
    if config_value is not None:
        return config_value

    env_value = get_env_value(env_name)
    if env_value is None or env_value == "":
        return default

    if isinstance(default, int):
        return _coerce_env_value(env_name, env_value, int)  # type: ignore[return-value]
    if isinstance(default, float):
        return _coerce_env_value(env_name, env_value, float)  # type: ignore[return-value]
    if isinstance(default, bool):
        normalized = env_value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True  # type: ignore[return-value]
        if normalized in {"0", "false", "no", "off"}:
            return False  # type: ignore[return-value]
        raise ValueError(f"Invalid value for environment variable '{env_name}': {env_value}")
    return env_value  # type: ignore[return-value]