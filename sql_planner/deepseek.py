from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class DeepSeekAPIError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Completion:
    message: dict[str, Any]
    finish_reason: str | None
    request_id: str | None
    model: str | None
    usage: dict[str, Any]


class DeepSeekClient:
    """DeepSeek Chat Completions client with native function calling."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://api.deepseek.com",
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required")
        if timeout_seconds <= 0 or max_retries < 0:
            raise ValueError("Invalid DeepSeek timeout or retry count")
        self.api_key = api_key
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> Completion:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "required",
            "thinking": {"type": "disabled"},
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        return self._request(payload)

    def complete_text(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> Completion:
        """Plain tagged-response completion used by reasoning trajectories."""
        completion = self._request({
            "model": self.model,
            "messages": messages,
            "thinking": {"type": "disabled"},
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "stop": ["</tool>", "</｜｜DSML｜｜ parameter>"],
        })
        content = completion.message.get("content")
        if (completion.finish_reason == "stop" and isinstance(content, str)
                and "<tool>" in content and "</tool>" not in content
                and content.rstrip().endswith("}")):
            return Completion(
                message={**completion.message, "content": content.rstrip() + "</tool>"},
                finish_reason=completion.finish_reason,
                request_id=completion.request_id,
                model=completion.model,
                usage=completion.usage,
            )
        return completion

    def _request(self, payload: dict[str, Any]) -> Completion:
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    result = json.load(response)
                choice = result["choices"][0]
                message = choice["message"]
                if not isinstance(message, dict):
                    raise DeepSeekAPIError("DeepSeek returned an invalid message")
                return Completion(
                    message=message,
                    finish_reason=choice.get("finish_reason"),
                    request_id=result.get("id"),
                    model=result.get("model"),
                    usage=result.get("usage") or {},
                )
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if retryable and attempt < self.max_retries:
                    time.sleep(min(2**attempt, 8))
                    continue
                raise DeepSeekAPIError(f"DeepSeek HTTP {exc.code}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 8))
                    continue
                raise DeepSeekAPIError(f"DeepSeek request failed: {type(exc).__name__}") from exc
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise DeepSeekAPIError("DeepSeek returned an invalid completion") from exc
        raise AssertionError("unreachable")
