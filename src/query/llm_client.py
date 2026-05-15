"""Reusable Anthropic chat client used by the RAG answer pipeline.

Build the client once with the system prompt that fits your use case, then call
.chat() per request. The system prompt is sent with prompt-caching enabled so
repeat questions reuse the cached prefix for ~90% cost reduction on the system
portion.

Example:
    client = AnthropicChatClient(
        model="claude-haiku-4-5",
        system_prompt="You are a packaging-industry research assistant...",
    )
    response = client.chat("What's happening with bioplastics?")
    print(response.content)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from anthropic import Anthropic

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import ANSWER_MAX_TOKENS, ANSWER_MODEL, ANSWER_TEMPERATURE  # noqa: E402


@dataclass
class ChatResponse:
    content: str
    model: str
    prompt_tokens: int = 0
    output_tokens: int = 0
    parsed: Any | None = None
    raw: Any = field(default=None)


class AnthropicChatClient:
    def __init__(
        self,
        model: str = ANSWER_MODEL,
        system_prompt: str | None = None,
        max_tokens: int = ANSWER_MAX_TOKENS,
        temperature: float = ANSWER_TEMPERATURE,
    ):
        self._client = Anthropic()
        self.model = model
        self.system_prompt = system_prompt or ""
        self.max_tokens = max_tokens
        self.temperature = temperature

    def chat(
        self,
        user_message: str,
        *,
        system_prompt: str | None = None,
        history: list[dict] | None = None,
    ) -> ChatResponse:
        sys_prompt = system_prompt if system_prompt is not None else self.system_prompt
        messages: list[dict] = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }
        if sys_prompt:
            kwargs["system"] = [{
                "type": "text",
                "text": sys_prompt,
                "cache_control": {"type": "ephemeral"},
            }]

        response = self._client.messages.create(**kwargs)

        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        ).strip()

        usage = response.usage
        prompt_tokens = (
            (usage.input_tokens or 0)
            + (getattr(usage, "cache_read_input_tokens", 0) or 0)
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
        )

        return ChatResponse(
            content=text,
            model=self.model,
            prompt_tokens=prompt_tokens,
            output_tokens=usage.output_tokens or 0,
            raw=response,
        )


@lru_cache(maxsize=8)
def get_default_client(model: str = ANSWER_MODEL) -> AnthropicChatClient:
    return AnthropicChatClient(model=model)
