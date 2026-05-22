"""Intent classifier driven by a small local LLM (DeepSeek via Ollama).

Routes the user message into one of five intents before the RAG pipeline runs.
Only RESEARCH triggers retrieval; everything else gets a canned reply.

The DeepSeek reasoning model can emit `<think>...</think>` blocks; we strip
those before parsing the label. If the LLM call fails for any reason we
fall back to RESEARCH so the user still gets an answer.
"""

from __future__ import annotations

import re
import sys
from enum import Enum
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import INTENT_MAX_TOKENS, INTENT_MODEL, INTENT_TEMPERATURE  # noqa: E402
from src.query.llm_client import OllamaChatClient  # noqa: E402


class Intent(str, Enum):
    GREETING = "greeting"
    THANKS = "thanks"
    GOODBYE = "goodbye"
    META = "meta"
    RESEARCH = "research"


INTENT_SYSTEM_PROMPT = (
    "You are an intent classifier for a research assistant on the global "
    "packaging industry. Given the user's message, output EXACTLY ONE label "
    "from this set:\n\n"
    "  - RESEARCH  — a substantive question about packaging companies, "
    "regulations, materials, M&A, sustainability, trends, etc.\n"
    "  - GREETING  — hello / hi / good morning / how are you / etc.\n"
    "  - THANKS    — thank you / thanks / appreciate it / cheers / etc.\n"
    "  - GOODBYE   — bye / see you / goodnight / etc.\n"
    "  - META      — 'who are you', 'what can you do', 'what data do you have'\n\n"
    "RULES:\n"
    "1. Output only the single label, in uppercase. No explanation, no "
    "punctuation, no surrounding text.\n"
    "2. If the message is a substantive question or statement about the "
    "packaging industry, choose RESEARCH.\n"
    "3. When ambiguous, prefer RESEARCH so the user still gets an answer."
)


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_LABEL_RE = re.compile(r"\b(RESEARCH|GREETING|THANKS|GOODBYE|META)\b", re.IGNORECASE)


@lru_cache(maxsize=1)
def _intent_client() -> OllamaChatClient:
    return OllamaChatClient(
        model=INTENT_MODEL,
        max_tokens=INTENT_MAX_TOKENS,
        temperature=INTENT_TEMPERATURE,
    )


def _parse_intent(raw: str) -> Intent:
    """Strip DeepSeek reasoning tags and pull the last label out of the reply."""
    cleaned = _THINK_BLOCK_RE.sub("", raw or "").strip()
    matches = _LABEL_RE.findall(cleaned)
    if not matches:
        # Reasoning may have leaked into the body — search the whole reply.
        matches = _LABEL_RE.findall(raw or "")
    if not matches:
        return Intent.RESEARCH
    label = matches[-1].lower()
    try:
        return Intent(label)
    except ValueError:
        return Intent.RESEARCH


def classify_intent(text: str) -> Intent:
    t = (text or "").strip()
    if not t:
        return Intent.GREETING

    try:
        response = _intent_client().chat(t, system_prompt=INTENT_SYSTEM_PROMPT)
    except Exception:
        return Intent.RESEARCH
    return _parse_intent(response.content)


CHITCHAT_RESPONSES = {
    Intent.GREETING: (
        "Hi! I'm a research assistant for the global packaging industry. I have a corpus of "
        "scraped news developments and a knowledge graph of organizations, people, and "
        "locations. Ask me about companies, M&A, regulations, materials, or trends — try "
        "things like:\n\n"
        "- *what are the latest bioplastics trends?*\n"
        "- *who acquired what packaging company recently?*\n"
        "- *what's happening with EU PPWR?*"
    ),
    Intent.THANKS: "You're welcome! Anything else you'd like to dig into?",
    Intent.GOODBYE: "Bye! Come back any time you want to look into packaging developments.",
    Intent.META: (
        "I'm a retrieval-augmented assistant over ~1,600 packaging-industry news articles. "
        "When you ask a research question, I:\n\n"
        "1. Find the most relevant articles via semantic search\n"
        "2. Generate a concise answer using only those articles, citing each claim by source ID\n"
        "3. Show you the underlying knowledge-graph neighborhood for any entities in your question\n\n"
        "Try asking about a specific company, a regulation (EPR, DRS, PPWR), a material "
        "(bioplastics, rPET, fiber-based), or a trend (M&A, greenwashing, AI-driven packaging)."
    ),
}


def chitchat_response(intent: Intent) -> str:
    return CHITCHAT_RESPONSES.get(intent, CHITCHAT_RESPONSES[Intent.GREETING])
