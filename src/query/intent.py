"""Lightweight intent classifier so the RAG pipeline doesn't run on chit-chat.

Decision rules (in order):
  1. Empty input              -> GREETING
  2. Has '?' AND not META     -> RESEARCH (questions are almost always research)
  3. Short (<= 4 words) AND matches greeting/thanks/goodbye pattern -> chit-chat
  4. Matches META pattern     -> META (allowed to be longer, e.g. "what can you do for me")
  5. Otherwise                -> RESEARCH
"""

from __future__ import annotations

import re
from enum import Enum


class Intent(str, Enum):
    GREETING = "greeting"
    THANKS = "thanks"
    GOODBYE = "goodbye"
    META = "meta"
    RESEARCH = "research"


GREETING_PATTERNS = [
    re.compile(r"\b(hi|hello|hey|howdy|hola|yo|sup)\b", re.I),
    re.compile(r"\bgood (morning|afternoon|evening|day)\b", re.I),
    re.compile(r"\bwhat'?s up\b", re.I),
    re.compile(r"\b(how (are|r) (you|u)|how(')?s it going|how(')?ve you been)\b", re.I),
]

THANKS_PATTERNS = [
    re.compile(r"\b(thanks|thank you|thx|cheers|appreciate it|much obliged|ty)\b", re.I),
]

GOODBYE_PATTERNS = [
    re.compile(r"\b(bye|goodbye|cya|see you|see ya|later|farewell|adios|good ?night|nite)\b", re.I),
]

META_PATTERNS = [
    re.compile(r"\b(who (are|r) (you|u)|what (are|r) (you|u))\b", re.I),
    re.compile(r"\b(what can you (do|help with|tell me)|how do you work|how can you help)\b", re.I),
    re.compile(r"\b(what (do you|data|info|sources) (do you )?(have|know|cover))\b", re.I),
    re.compile(r"\b(your (capabilities|features|sources|data))\b", re.I),
]


def _matches(text: str, patterns: list[re.Pattern]) -> bool:
    return any(p.search(text) for p in patterns)


def classify_intent(text: str) -> Intent:
    t = (text or "").strip()
    if not t:
        return Intent.GREETING

    word_count = len(t.split())
    has_question = "?" in t

    if has_question and not _matches(t, META_PATTERNS):
        return Intent.RESEARCH

    if word_count <= 4:
        if _matches(t, GREETING_PATTERNS):
            return Intent.GREETING
        if _matches(t, THANKS_PATTERNS):
            return Intent.THANKS
        if _matches(t, GOODBYE_PATTERNS):
            return Intent.GOODBYE

    if _matches(t, META_PATTERNS):
        return Intent.META

    if word_count <= 6 and _matches(t, GOODBYE_PATTERNS):
        return Intent.GOODBYE
    if word_count <= 6 and _matches(t, THANKS_PATTERNS):
        return Intent.THANKS

    return Intent.RESEARCH


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
