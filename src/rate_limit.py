from __future__ import annotations

import json
import time
from urllib.error import HTTPError


_LAST_REQUEST_AT: dict[str, float] = {}


def wait_for_turn(bucket: str, min_interval_seconds: float) -> None:
    if min_interval_seconds <= 0:
        return

    now = time.monotonic()
    last_request_at = _LAST_REQUEST_AT.get(bucket)
    if last_request_at is not None:
        elapsed = now - last_request_at
        if elapsed < min_interval_seconds:
            time.sleep(min_interval_seconds - elapsed)

    _LAST_REQUEST_AT[bucket] = time.monotonic()


def extract_retry_delay_seconds(error: HTTPError, default_seconds: float, max_seconds: float) -> float:
    retry_after = error.headers.get("Retry-After") if error.headers else None
    if retry_after:
        try:
            return min(float(retry_after), max_seconds)
        except ValueError:
            pass

    try:
        body = error.read().decode("utf-8")
        payload = json.loads(body)
        details = payload.get("error", {}).get("details", [])
        for detail in details:
            retry_delay = detail.get("retryDelay")
            if not retry_delay:
                continue
            if isinstance(retry_delay, str) and retry_delay.endswith("s"):
                return min(float(retry_delay[:-1]), max_seconds)
    except Exception:
        pass

    return min(default_seconds, max_seconds)