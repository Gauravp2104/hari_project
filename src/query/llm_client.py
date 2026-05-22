"""Reusable chat client backed by a local Ollama server.

Calls a locally-served open-weight model (qwen2.5, llama3.2, mistral, …) over
Ollama's HTTP chat API. No API key required — assumes `ollama serve` is running
on `OLLAMA_HOST` (default http://localhost:11434) and that the chosen model has
been pulled (e.g. `ollama pull qwen2.5:3b`).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (  # noqa: E402
    ANSWER_MAX_TOKENS,
    ANSWER_MODEL,
    ANSWER_TEMPERATURE,
    OLLAMA_HOST,
)


@dataclass
class ChatResponse:
    content: str
    model: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    parsed: Any | None = None
    raw: Any = field(default=None)


class OllamaChatClient:
    def __init__(
        self,
        model: str = ANSWER_MODEL,
        system_prompt: str | None = None,
        max_tokens: int = ANSWER_MAX_TOKENS,
        temperature: float = ANSWER_TEMPERATURE,
        host: str | None = None,
        timeout: float = 300.0,
        api_key: str | None = None,
    ):
        self.host = (host or os.environ.get("OLLAMA_HOST") or OLLAMA_HOST).rstrip("/")
        self.model = model
        self.system_prompt = system_prompt or ""
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        # Only sent when the host needs auth (e.g. https://ollama.com cloud endpoint).
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY")

    def chat(
        self,
        user_message: str,
        *,
        system_prompt: str | None = None,
        history: list[dict] | None = None,
    ) -> ChatResponse:
        sys_prompt = system_prompt if system_prompt is not None else self.system_prompt
        messages: list[dict] = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = requests.post(
            f"{self.host}/api/chat",
            json=payload,
            headers=headers or None,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        text = (data.get("message", {}).get("content") or "").strip()

        return ChatResponse(
            content=text,
            model=self.model,
            prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
            output_tokens=int(data.get("eval_count", 0) or 0),
            raw=data,
        )


# Back-compat alias so older imports keep working during the migration.
HFChatClient = OllamaChatClient


@lru_cache(maxsize=8)
def get_default_client(model: str = ANSWER_MODEL) -> OllamaChatClient:
    return OllamaChatClient(model=model)
