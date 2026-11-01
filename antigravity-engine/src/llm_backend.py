"""
Project Antigravity — LLM Backend Abstraction.

Wraps existing, battle-tested runtimes instead of writing custom Metal/Vulkan
shaders (which lose to llama.cpp, MPSGraph, and LiteRT on raw throughput):

  * llama.cpp — via its C API or an OpenAI-compatible HTTP endpoint
  * MLX — Apple's on-device ML framework (mlx-lm server, OpenAI-compatible)
  * Ollama — local HTTP API
  * Strata — local inference server (OpenAI-compatible)
  * Cloud — OpenAI / Anthropic fallback for Tier 3 escalation

Each backend implements: generate(prompt, max_tokens, temperature, top_p) ->
(text, tokens_used, latency_ms, estimated_watts).
"""

from __future__ import annotations

import abc
import json
import time
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    tokens_used: int
    latency_ms: float
    estimated_watts: float
    backend: str
    model: str
    finish_reason: str = "stop"


class LLMBackend(abc.ABC):
    """Abstract base for an LLM inference backend."""

    @abc.abstractmethod
    def generate(self, prompt: str, max_tokens: int = 40,
                 temperature: float = 0.7, top_p: float = 0.9,
                 **kwargs) -> GenerationResult:
        ...

    @abc.abstractmethod
    def is_available(self) -> bool:
        """True if the backend is reachable and has a model loaded."""
        ...

    @abc.abstractmethod
    def model_name(self) -> str:
        ...

    @abc.abstractmethod
    def backend_name(self) -> str:
        """Short identifier for this backend (e.g. 'ollama', 'mlx', 'strata')."""
        ...


class OpenAICompatibleBackend(LLMBackend):
    """
    Backend for any OpenAI-compatible HTTP endpoint (llama.cpp server,
    mlx-lm.server, Strata, etc.).

    Endpoint: POST {base_url}/v1/chat/completions
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8080",
                 model: str = "local-model", timeout_s: float = 30.0,
                 estimated_watts_per_token: float = 0.05):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.estimated_watts_per_token = estimated_watts_per_token
        self._cached_available: Optional[bool] = None
        self._availability_check_time: float = 0.0

    def is_available(self) -> bool:
        now = time.monotonic()
        if self._cached_available is not None and now - self._availability_check_time < 5.0:
            return self._cached_available
        try:
            req = urllib.request.Request(f"{self.base_url}/v1/models")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                if resp.status == 200:
                    self._cached_available = True
                else:
                    self._cached_available = False
        except Exception:
            self._cached_available = False
        self._availability_check_time = now
        return self._cached_available

    def model_name(self) -> str:
        return self.model

    def generate(self, prompt: str, max_tokens: int = 40,
                 temperature: float = 0.7, top_p: float = 0.9,
                 **kwargs) -> GenerationResult:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise BackendUnavailable(f"{self.backend_name()} unreachable: {exc}") from exc
        latency_ms = (time.monotonic() - t0) * 1000.0

        data = json.loads(raw)
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        text = message.get("content", "")
        usage = data.get("usage", {})
        tokens_used = int(usage.get("completion_tokens", 0) or 0)
        if tokens_used == 0:
            tokens_used = max(len(text.split()), 1)
        finish_reason = choice.get("finish_reason", "stop")

        estimated_watts = tokens_used * self.estimated_watts_per_token

        return GenerationResult(
            text=text,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            estimated_watts=estimated_watts,
            backend=self.backend_name(),
            model=self.model,
            finish_reason=finish_reason,
        )

    def backend_name(self) -> str:
        return "openai-compatible"


class OllamaBackend(OpenAICompatibleBackend):
    """Ollama local HTTP API (OpenAI-compatible /v1/chat/completions)."""

    def backend_name(self) -> str:
        return "ollama"


class StrataBackend(OpenAICompatibleBackend):
    """Strata local inference server (OpenAI-compatible)."""

    def backend_name(self) -> str:
        return "strata"


class MLXBackend(OpenAICompatibleBackend):
    """mlx-lm.server — Apple's on-device ML framework (OpenAI-compatible)."""

    def backend_name(self) -> str:
        return "mlx"


class CloudBackend(LLMBackend):
    """Tier 3 cloud fallback — OpenAI-compatible API with API key auth."""

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout_s: float = 60.0,
                 estimated_watts_per_token: float = 0.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.estimated_watts_per_token = estimated_watts_per_token

    def is_available(self) -> bool:
        return bool(self.api_key)

    def model_name(self) -> str:
        return self.model

    def backend_name(self) -> str:
        return "cloud"

    def generate(self, prompt: str, max_tokens: int = 40,
                 temperature: float = 0.7, top_p: float = 0.9,
                 **kwargs) -> GenerationResult:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise BackendUnavailable(f"Cloud backend unreachable: {exc}") from exc
        latency_ms = (time.monotonic() - t0) * 1000.0

        data = json.loads(raw)
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        text = message.get("content", "")
        usage = data.get("usage", {})
        tokens_used = int(usage.get("completion_tokens", 0) or 0)
        if tokens_used == 0:
            tokens_used = max(len(text.split()), 1)
        finish_reason = choice.get("finish_reason", "stop")

        estimated_watts = tokens_used * self.estimated_watts_per_token

        return GenerationResult(
            text=text,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            estimated_watts=estimated_watts,
            backend="cloud",
            model=self.model,
            finish_reason=finish_reason,
        )


class BackendUnavailable(RuntimeError):
    """Raised when a backend is unreachable."""
    pass


__all__ = [
    "LLMBackend", "GenerationResult",
    "OpenAICompatibleBackend", "OllamaBackend", "StrataBackend", "MLXBackend",
    "CloudBackend", "BackendUnavailable",
]