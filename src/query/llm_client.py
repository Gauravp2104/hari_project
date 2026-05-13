"""Reusable Ollama chat client with custom system prompts and structured output.

Build the client once with the system prompt and default options that fit your
use case, then call .chat() (free-form) or .chat_structured() (schema-constrained
JSON) per request. Both methods accept per-call overrides for system prompt and
options.

Example — free-form:
    client = OllamaChatClient(
        model="mistral:7b",
        system_prompt="You are a packaging-industry research assistant...",
    )
    response = client.chat("What's happening with bioplastics?")
    print(response.content)

Example — structured output (schema-enforced):
    schema = {
        "type": "object",
        "properties": {
            "company": {"type": "string"},
            "deal_value_usd_millions": {"type": "number"},
            "announced_date": {"type": "string"},
        },
        "required": ["company", "deal_value_usd_millions", "announced_date"],
    }
    response = client.chat_structured(
        "Extract the deal details: ProAmpac signed a $1.51B agreement to buy "
        "TC Transcontinental Packaging on 2025-12-04.",
        schema=schema,
    )
    deal = response.parsed   # already a dict matching the schema
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import ollama

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (  # noqa: E402
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_TEMPERATURE,
)


@dataclass
class ChatResponse:
    """One response from the Ollama chat endpoint, normalized."""

    content: str
    model: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    parsed: Any | None = None
    raw: dict = field(default_factory=dict)


class OllamaChatClient:
    """Wraps `ollama.Client` with a default system prompt and options.

    The system prompt and options are set at construction so the same client
    can be reused across many requests in a pipeline. Both can be overridden
    per call when needed.
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        host: str = OLLAMA_HOST,
        system_prompt: str | None = None,
        options: dict | None = None,
    ):
        self._client = ollama.Client(host=host)
        self.model = model
        self.host = host
        self.system_prompt = system_prompt or ""
        self.default_options = dict(options) if options else {
            "temperature": OLLAMA_TEMPERATURE,
            "num_ctx": OLLAMA_NUM_CTX,
        }

    def _build_messages(
        self,
        user_message: str,
        system_prompt: str | None,
        history: list[dict] | None,
    ) -> list[dict]:
        sys_prompt = system_prompt if system_prompt is not None else self.system_prompt
        messages: list[dict] = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages

    def _merge_options(self, options_override: dict | None) -> dict:
        if not options_override:
            return self.default_options
        return {**self.default_options, **options_override}

    def chat(
        self,
        user_message: str,
        *,
        system_prompt: str | None = None,
        history: list[dict] | None = None,
        options: dict | None = None,
    ) -> ChatResponse:
        """Free-form chat. Returns ChatResponse with `.content` populated."""
        response = self._client.chat(
            model=self.model,
            messages=self._build_messages(user_message, system_prompt, history),
            options=self._merge_options(options),
        )
        return self._wrap(response, parsed=None)

    def chat_structured(
        self,
        user_message: str,
        *,
        schema: dict | str = "json",
        system_prompt: str | None = None,
        history: list[dict] | None = None,
        options: dict | None = None,
    ) -> ChatResponse:
        """Structured chat — Ollama enforces the output to match the JSON schema.

        Pass `schema="json"` for basic JSON mode (no schema), or a JSON Schema
        dict for schema-enforced output (recommended; requires Ollama >= 0.5.0).
        Returns ChatResponse with `.parsed` populated as a dict on success.
        """
        response = self._client.chat(
            model=self.model,
            messages=self._build_messages(user_message, system_prompt, history),
            options=self._merge_options(options),
            format=schema,
        )
        wrapped = self._wrap(response, parsed=None)
        try:
            wrapped.parsed = json.loads(wrapped.content)
        except json.JSONDecodeError:
            wrapped.parsed = None
        return wrapped

    def _wrap(self, response: dict, parsed: Any | None) -> ChatResponse:
        return ChatResponse(
            content=(response["message"]["content"] or "").strip(),
            model=self.model,
            prompt_tokens=response.get("prompt_eval_count", 0) or 0,
            output_tokens=response.get("eval_count", 0) or 0,
            parsed=parsed,
            raw=response,
        )


@lru_cache(maxsize=8)
def get_default_client(
    model: str = OLLAMA_MODEL,
    host: str = OLLAMA_HOST,
) -> OllamaChatClient:
    """Singleton client per (model, host). Use when no custom system prompt is needed at construction."""
    return OllamaChatClient(model=model, host=host)
