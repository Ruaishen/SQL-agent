from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse


class ModelAdapterError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Generation:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_seconds: float = 0.0
    finish_reason: str | None = None
    request_id: str | None = None
    response_model: str | None = None
    system_fingerprint: str | None = None
    cached_prompt_tokens: int = 0
    reasoning_tokens: int = 0


class ModelAdapter(Protocol):
    @property
    def model_name(self) -> str: ...

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        max_tokens: int,
        seed: int,
        enable_thinking: bool,
    ) -> Generation: ...


class OpenAICompatibleAdapter:
    """Small dependency-free adapter for vLLM/SGLang OpenAI-compatible servers."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        stop: list[str] | None = None,
    ):
        endpoint = endpoint.rstrip("/")
        if endpoint.endswith("/chat/completions"):
            self.url = endpoint
        elif endpoint.endswith("/v1"):
            self.url = endpoint + "/chat/completions"
        else:
            self.url = endpoint + "/v1/chat/completions"
        self._model_name = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.stop = stop
        hostname = urlparse(self.url).hostname
        self._opener = (
            urllib.request.build_opener(urllib.request.ProxyHandler({}))
            if hostname in {"127.0.0.1", "localhost", "::1"}
            else None
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        max_tokens: int,
        seed: int,
        enable_thinking: bool,
    ) -> Generation:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "max_tokens": max_tokens,
            "seed": seed,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if self.stop is not None:
            payload["stop"] = self.stop
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        started = time.monotonic()
        try:
            open_request = self._opener.open if self._opener is not None else urllib.request.urlopen
            with open_request(request, timeout=self.timeout_seconds) as response:
                value = json.loads(response.read())
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise ModelAdapterError(f"Model request failed: {exc}") from exc
        try:
            choice = value["choices"][0]
            usage = value.get("usage", {})
            return Generation(
                text=choice["message"]["content"] or "",
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                latency_seconds=time.monotonic() - started,
                finish_reason=choice.get("finish_reason"),
                request_id=value.get("id"),
                response_model=value.get("model"),
                system_fingerprint=value.get("system_fingerprint"),
                cached_prompt_tokens=int(
                    (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
                ),
                reasoning_tokens=int(
                    (usage.get("completion_tokens_details") or {}).get(
                        "reasoning_tokens", 0
                    )
                ),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ModelAdapterError("Model response has an invalid chat-completions shape") from exc


class ScriptedModelAdapter:
    """Deterministic adapter for CPU-only tests and trajectory fixtures."""

    def __init__(self, responses: list[str], model_name: str = "scripted"):
        self.responses = deque(responses)
        self._model_name = model_name
        self.requests: list[list[dict[str, str]]] = []
        self.settings: list[dict[str, object]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        max_tokens: int,
        seed: int,
        enable_thinking: bool,
    ) -> Generation:
        self.requests.append(messages)
        self.settings.append(
            {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "max_tokens": max_tokens,
                "seed": seed,
                "enable_thinking": enable_thinking,
            }
        )
        if not self.responses:
            raise ModelAdapterError("Scripted model has no response left")
        return Generation(text=self.responses.popleft())
