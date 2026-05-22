"""Map a query to one of the corpus's canonical Category labels — or None.

Used as an intent-classification gate at the start of the answer pipeline:
  * If the question clearly maps to one of the 31 canonical categories, we
    pre-filter retrieval to that hard category and generate trends from a
    pure single-category slice.
  * If the question is broad / cross-cutting / ambiguous, we return None and
    the pipeline falls back to broad retrieval + category post-filtering on
    the LLM-generated trend blocks.

Match policy (in order): exact → case/punct-insensitive → substring either
direction. If none of those produce a confident match, or the LLM replies
NONE, we return None — no force-pick fallback.
"""

from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import PROCESSED_DIR  # noqa: E402
from src.query.llm_client import OllamaChatClient  # noqa: E402


CATEGORY_SYSTEM_PROMPT = (
    "You are a router. You will be given a user question about the global packaging "
    "industry and a list of category labels used to tag developments in our corpus. "
    "Your job is to decide whether the question maps cleanly to EXACTLY ONE of those "
    "categories, or whether it is broad / cross-cutting and does not.\n\n"
    "RULES:\n"
    "1. If the question is clearly about a single category from the list, reply with "
    "that category label EXACTLY as it appears — verbatim, no quotes, no extra text.\n"
    "2. If the question is broad, cross-cutting, ambiguous, generic, or does not "
    "clearly belong to any single category on the list, reply with the single word: "
    "NONE\n"
    "3. Output nothing except either one verbatim category label or the word NONE. "
    "No explanation, no punctuation, no surrounding text.\n"
    "4. Do NOT force a pick. Prefer NONE over a weak guess."
)


@lru_cache(maxsize=1)
def canonical_categories() -> list[str]:
    """Return the corpus's canonical category list, sorted by descending frequency."""
    parquet = PROCESSED_DIR / "Packaging_Developments_25-26.parquet"
    df = pd.read_parquet(parquet)
    counts = (
        df["Category"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .value_counts()
    )
    return counts.index.tolist()


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _resolve_to_canonical(reply: str, categories: list[str]) -> str | None:
    """Map a free-form model reply to one of the canonical categories, or None.

    Returns None when the LLM signals NONE or when no fuzzy match lands —
    no force-pick fallback.
    """
    lines = reply.strip().strip("\"'.").splitlines()
    reply = lines[0].strip() if lines else ""
    if not reply or reply.strip(".:-_ ").upper() == "NONE":
        return None

    # 1. exact match
    for c in categories:
        if reply == c:
            return c

    # 2. case-insensitive / punctuation-insensitive match
    rn = _normalize(reply)
    if not rn:
        return None
    for c in categories:
        if _normalize(c) == rn:
            return c

    # 3. substring either direction
    for c in categories:
        cn = _normalize(c)
        if cn and (cn in rn or rn in cn):
            return c

    return None


def classify_category(question: str, client: OllamaChatClient) -> str | None:
    """Return the canonical category for `question`, or None if no single
    category clearly applies.
    """
    categories = canonical_categories()
    listing = "\n".join(f"- {c}" for c in categories)
    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"CATEGORY LABELS:\n{listing}\n\n"
        "Reply with exactly one label from the list above, or with the single word "
        "NONE if no single category clearly applies."
    )
    response = client.chat(user_msg, system_prompt=CATEGORY_SYSTEM_PROMPT)
    return _resolve_to_canonical(response.content, categories)
