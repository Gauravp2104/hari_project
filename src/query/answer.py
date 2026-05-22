"""RAG answer pipeline that emits category-grounded trends with cited developments.

Hybrid pre/post category-filter pipeline:
  1. Classify the question against the corpus's 31 canonical categories
     (small local model). The classifier returns either a single category
     label OR None.
     * Category matched  → PRE-filter: retrieve only chunks tagged with that
       category from ChromaDB. Trends are generated from this hard slice.
     * No category match → no pre-filter: retrieve broadly and rely on the
       step-4 category post-filter to ground trends.
  2. Format the retrieved chunks as a structured list — every dev shows its
     own category, title, date, and a short snippet. These are the relevant
     developments.
  3. A single LLM call (cloud / larger model) abstracts those developments
     into 2-4 trends. Each trend names ONE category drawn from the supplied
     devs and cites the specific dev titles that support it.
  4. Category post-filter in code: parse the trend blocks, match each cited
     title back to a real retrieved chunk, verify the chunk's category
     matches the trend's stated category, drop trends that don't survive
     validation. In the pre-filter branch this is a near no-op (all chunks
     share one category); in the broad branch it does real work.

Usage:
    python -m src.query.answer "what are the latest bioplastics trends?"
    python -m src.query.answer "who acquired whom recently?" --k 12
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (  # noqa: E402
    ANSWER_MODEL,
    TRENDS_MAX_TOKENS,
    TRENDS_MODEL,
    TRENDS_OLLAMA_HOST,
    TRENDS_TEMPERATURE,
)
from src.query.category_classify import classify_category  # noqa: E402
from src.query.intent import Intent, chitchat_response, classify_intent  # noqa: E402
from src.query.llm_client import OllamaChatClient  # noqa: E402
from src.rag.retrieve import RetrievedChunk, dedup_by_article, retrieve  # noqa: E402


TRENDS_SYSTEM_PROMPT = (
    "You are an industry analyst for the global packaging sector.\n\n"
    "You will receive:\n"
    "  • A user question.\n"
    "  • A numbered list of recent developments, each tagged with the industry "
    "category from our corpus (e.g. 'Bioplastics', 'Policy & Regulation', "
    "'Foodservice').\n\n"
    "Your job: identify 2-4 broader trends that those developments collectively "
    "signal. EACH trend must be grounded to ONE category that appears in the "
    "supplied developments, and must explicitly cite the developments that "
    "support it.\n\n"
    "OUTPUT FORMAT — emit only trend blocks, separated by a single blank line. "
    "Each block has exactly these labeled lines, in this order:\n\n"
    "**Trend** [Category: <one category name, copied verbatim from the supplied "
    "developments>]: <a 1-2 sentence statement of the trend within that "
    "category, abstracted one level above the specific facts>\n"
    "**Implication:** <one sentence on what this means for players in that "
    "category going forward>\n"
    "**Key developments:**\n"
    "- <verbatim title of supporting development #1>\n"
    "- <verbatim title of supporting development #2>\n"
    "- <... 2 to 5 titles total, all from the same category as the trend>\n\n"
    "RULES:\n"
    "1. Every title under 'Key developments' MUST be copied verbatim from the "
    "supplied list. Do not invent titles or paraphrase them.\n"
    "2. Every cited development must share the trend's stated category. Do not "
    "mix categories within a single trend.\n"
    "3. A trend needs at least 2 supporting developments — drop any trend "
    "supported by fewer than 2.\n"
    "4. Produce 2-4 trends. Where the developments span multiple categories, "
    "prefer trends from different categories over piling everything into one.\n"
    "5. The trend statement is the abstraction one level above the specific "
    "facts (e.g. 'regional consolidation via M&A' over 'Company X bought "
    "Company Y'; 'tightening EPR mandates push brands toward mono-materials' "
    "over 'France updated its packaging law').\n"
    "6. No introduction, no conclusion, no overall headers, no source citations "
    "in brackets like [1]. Output only the trend blocks.\n"
    "7. If the supplied developments are too sparse or too unrelated to support "
    "any cross-cutting trend, output EXACTLY: 'Not enough material in the "
    "retrieved developments to identify broader trends.' and nothing else."
)


_CITATION_RE = re.compile(r"\s*\[\d+(?:\s*[,–-]\s*\d+)*\]")


def _strip_citations(text: str) -> str:
    return _CITATION_RE.sub("", text).strip()


@dataclass
class TrendCard:
    """One validated trend, with the retrieved devs that back it."""

    trend: str
    category: str
    implication: str
    dev_article_ids: list[str] = field(default_factory=list)
    dev_titles: list[str] = field(default_factory=list)


@dataclass
class AnswerResult:
    question: str
    trend_cards: list[TrendCard] = field(default_factory=list)
    trends: str = ""  # markdown rendering of trend_cards (cache-friendly)
    developments: str = ""  # markdown rendering of retrieved devs
    sources: list[RetrievedChunk] = field(default_factory=list)
    category: str = ""  # back-compat — joined list of categories present
    model: str = ""
    trends_model: str = ""
    intent: str = Intent.RESEARCH.value
    notes: list[str] = field(default_factory=list)  # post-filter warnings
    prompt_eval_count: int = 0
    eval_count: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def answer(self) -> str:
        return self.trends or self.developments


# ----- prompt formatting ---------------------------------------------------

def _format_devs_for_prompt(chunks: list[RetrievedChunk], max_chars_each: int = 350) -> str:
    """Structured, category-tagged list for the trends model to ground on."""
    lines = []
    for i, c in enumerate(chunks, 1):
        date = c.date.split(" ")[0] if c.date else ""
        category = c.category or "Uncategorized"
        snippet = c.text.strip().replace("\n", " ")
        if len(snippet) > max_chars_each:
            snippet = snippet[:max_chars_each].rstrip() + "…"
        date_part = f" — {date}" if date else ""
        lines.append(
            f"{i}. [{category}]{date_part}  \"{c.title}\"\n     {snippet}"
        )
    return "\n\n".join(lines)


def _render_developments_md(chunks: list[RetrievedChunk]) -> str:
    """Markdown bullet list of every retrieved dev — fills the expander."""
    lines = []
    for c in chunks:
        date = c.date.split(" ")[0] if c.date else ""
        category = c.category or "Uncategorized"
        bits = [f"**[{category}]**"]
        if date:
            bits.append(f"*{date}*")
        bits.append(c.title)
        line = f"- " + " — ".join(bits)
        if c.link:
            line += f"  \n  [article link]({c.link})"
        lines.append(line)
    return "\n".join(lines)


# ----- post-filter ---------------------------------------------------------

_TREND_LINE_RE = re.compile(
    r"^\s*\*\*Trend\*\*\s*\[Category:\s*([^\]]+)\]\s*:\s*(.+)$",
    re.IGNORECASE,
)
_IMPLICATION_LINE_RE = re.compile(
    r"^\s*\*\*Implication:?\*\*\s*:?\s*(.+)$",
    re.IGNORECASE,
)
_KEY_DEV_HEADER_RE = re.compile(r"^\s*\*\*Key developments?:?\*\*\s*:?\s*$", re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*[-*•]\s+(.+?)\s*$")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _match_cited_title(cited: str, chunks: list[RetrievedChunk]) -> RetrievedChunk | None:
    """Find the retrieved chunk whose title best matches a cited title."""
    cited_clean = cited.strip().strip("\"'.,;:")
    if not cited_clean:
        return None
    cn = _norm(cited_clean)
    if not cn:
        return None
    # 1. exact (normalized) match
    for c in chunks:
        if _norm(c.title) == cn:
            return c
    # 2. substring either direction
    for c in chunks:
        tn = _norm(c.title)
        if tn and (tn in cn or cn in tn):
            return c
    return None


def _parse_trend_blocks(text: str) -> list[dict]:
    """Tolerant parser. Returns a list of {category, trend, implication, cited_titles}."""
    blocks: list[dict] = []
    current: dict | None = None
    in_devs = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            in_devs = False
            continue

        m = _TREND_LINE_RE.match(line)
        if m:
            if current and (current.get("trend") or current.get("cited_titles")):
                blocks.append(current)
            current = {
                "category": m.group(1).strip().strip("\"'"),
                "trend": m.group(2).strip(),
                "implication": "",
                "cited_titles": [],
            }
            in_devs = False
            continue

        if current is None:
            continue

        m = _IMPLICATION_LINE_RE.match(line)
        if m:
            current["implication"] = m.group(1).strip()
            in_devs = False
            continue

        if _KEY_DEV_HEADER_RE.match(line):
            in_devs = True
            continue

        if in_devs:
            m = _BULLET_RE.match(line)
            if m:
                current["cited_titles"].append(m.group(1))

    if current and (current.get("trend") or current.get("cited_titles")):
        blocks.append(current)
    return blocks


def _validate_trend_cards(
    raw_blocks: list[dict],
    chunks: list[RetrievedChunk],
) -> tuple[list[TrendCard], list[str]]:
    """Resolve each trend block's cited titles to real chunks; enforce category match."""
    cards: list[TrendCard] = []
    notes: list[str] = []
    chunks_by_id = {c.article_id: c for c in chunks}
    valid_categories = {_norm(c.category): c.category for c in chunks if c.category}

    for blk in raw_blocks:
        stated_cat_raw = blk["category"]
        stated_cat_norm = _norm(stated_cat_raw)
        canonical_cat = valid_categories.get(stated_cat_norm)
        if not canonical_cat:
            notes.append(
                f"dropped trend — stated category '{stated_cat_raw}' not in retrieved devs"
            )
            continue

        validated_ids: list[str] = []
        validated_titles: list[str] = []
        seen_ids: set[str] = set()
        for cited in blk["cited_titles"]:
            match = _match_cited_title(cited, chunks)
            if match is None:
                notes.append(
                    f"dropped citation '{cited[:60]}' — no matching retrieved dev"
                )
                continue
            if _norm(match.category) != stated_cat_norm:
                notes.append(
                    f"dropped citation '{match.title[:60]}' — its category "
                    f"'{match.category}' ≠ trend category '{canonical_cat}'"
                )
                continue
            if match.article_id in seen_ids:
                continue
            seen_ids.add(match.article_id)
            validated_ids.append(match.article_id)
            validated_titles.append(match.title)

        if len(validated_ids) < 2:
            notes.append(
                f"dropped trend '{blk['trend'][:60]}' — fewer than 2 valid citations"
            )
            continue

        cards.append(
            TrendCard(
                trend=blk["trend"],
                category=canonical_cat,
                implication=blk.get("implication", ""),
                dev_article_ids=validated_ids,
                dev_titles=validated_titles,
            )
        )

    # Resolve titles via chunks_by_id (defensive — keeps title rendering consistent)
    for card in cards:
        card.dev_titles = [
            chunks_by_id[aid].title if aid in chunks_by_id else t
            for aid, t in zip(card.dev_article_ids, card.dev_titles)
        ]
    return cards, notes


def _render_trend_cards_md(cards: list[TrendCard]) -> str:
    if not cards:
        return "Not enough material in the retrieved developments to identify broader trends."
    parts = []
    for card in cards:
        body = (
            f"**Trend** [Category: `{card.category}`]: {card.trend}\n\n"
            f"**Implication:** {card.implication}\n\n"
            "**Key developments:**\n"
            + "\n".join(f"- {t}" for t in card.dev_titles)
        )
        parts.append(body)
    return "\n\n---\n\n".join(parts)


# ----- main entry point ----------------------------------------------------

def _build_trends_client() -> OllamaChatClient:
    return OllamaChatClient(
        model=TRENDS_MODEL,
        host=TRENDS_OLLAMA_HOST,
        max_tokens=TRENDS_MAX_TOKENS,
        temperature=TRENDS_TEMPERATURE,
    )


def answer_question(
    question: str,
    k: int = 8,
    model: str = ANSWER_MODEL,
    system_prompt: str | None = None,
    client: OllamaChatClient | None = None,
    trends_client: OllamaChatClient | None = None,
    **_ignored,
) -> AnswerResult:
    """Answer a research question with category-grounded trends.

    Hybrid pre/post category-filter:
      * If the question maps cleanly to one of the corpus's 31 canonical
        categories, retrieval is pre-filtered to that single category.
      * Otherwise retrieval is broad, and the per-dev categories from the
        corpus drive grounding via the step-4 post-filter on trend blocks.

    `model` selects the small local Ollama model used for category
    classification. The trends synthesis itself runs against the cloud model
    in `trends_client`.
    """
    intent = classify_intent(question)
    if intent != Intent.RESEARCH:
        return AnswerResult(
            question=question,
            trends=chitchat_response(intent),
            developments="",
            sources=[],
            model="",
            intent=intent.value,
        )

    # ----- Stage 1: classify the question against canonical categories -----
    if client is None:
        client = OllamaChatClient(model=model)
    matched_category = classify_category(question, client)
    pipeline_notes: list[str] = []

    if matched_category:
        pipeline_notes.append(
            f"pre-filter: classified to '{matched_category}' — retrieval restricted "
            f"to that category"
        )
        raw_chunks = retrieve(question, k=k * 2, category_filter=matched_category)
        chunks = dedup_by_article(raw_chunks)[:k]
        if not chunks:
            # Category matched but no semantic hits inside it — fall back to broad.
            pipeline_notes.append(
                f"pre-filter yielded 0 chunks in '{matched_category}'; falling back "
                "to broad retrieval with post-filter"
            )
            matched_category = None
            raw_chunks = retrieve(question, k=k * 2)
            chunks = dedup_by_article(raw_chunks)[:k]
    else:
        pipeline_notes.append(
            "post-filter: question did not map to a single category — broad "
            "retrieval, categories enforced on trend blocks"
        )
        raw_chunks = retrieve(question, k=k * 2)
        chunks = dedup_by_article(raw_chunks)[:k]

    if not chunks:
        msg = "I don't have any relevant articles in the corpus to answer that question."
        return AnswerResult(
            question=question,
            trends=msg,
            developments="",
            sources=[],
            model=model,
            intent=intent.value,
            notes=pipeline_notes,
        )

    if trends_client is None:
        trends_client = _build_trends_client()

    dev_listing = _format_devs_for_prompt(chunks)
    categories_present = sorted({c.category for c in chunks if c.category})
    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"CATEGORIES PRESENT IN THE DEVELOPMENTS BELOW: "
        f"{', '.join(categories_present) if categories_present else '(none)'}\n\n"
        f"DEVELOPMENTS:\n{dev_listing}\n\n"
        "Identify the broader trends. Remember: each trend names one category, "
        "and the 'Key developments' titles must be copied verbatim from the list "
        "above and share the trend's category."
    )

    trends_response = trends_client.chat(
        user_msg,
        system_prompt=system_prompt or TRENDS_SYSTEM_PROMPT,
    )
    raw_text = _strip_citations(trends_response.content)

    raw_blocks = _parse_trend_blocks(raw_text)
    cards, validation_notes = _validate_trend_cards(raw_blocks, chunks)
    trends_md = _render_trend_cards_md(cards) if cards else raw_text

    developments_md = _render_developments_md(chunks)

    category_label = matched_category if matched_category else ", ".join(categories_present)

    return AnswerResult(
        question=question,
        trend_cards=cards,
        trends=trends_md,
        developments=developments_md,
        sources=chunks,
        category=category_label,
        model=client.model,
        trends_model=trends_client.model,
        intent=intent.value,
        notes=pipeline_notes + validation_notes,
        prompt_eval_count=trends_response.prompt_tokens,
        eval_count=trends_response.output_tokens,
        input_tokens=trends_response.prompt_tokens,
        output_tokens=trends_response.output_tokens,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Ask a question against the developments corpus (local Ollama + cloud)."
    )
    parser.add_argument("question", help="The natural-language question")
    parser.add_argument("--k", type=int, default=8, help="Number of articles to retrieve")
    parser.add_argument(
        "--model", default=ANSWER_MODEL, help=f"Stage-1 (unused) model (default: {ANSWER_MODEL})"
    )
    args = parser.parse_args()

    result = answer_question(
        args.question,
        k=args.k,
        model=args.model,
    )
    if result.notes:
        print("PIPELINE NOTES:")
        for n in result.notes:
            print(f"  - {n}")
        print()
    if result.category:
        print(f"CATEGORY: {result.category}")
        print()
    print("TRENDS:")
    print(result.trends)
    print()
    print("=" * 80)
    print("RETRIEVED DEVELOPMENTS:")
    for i, c in enumerate(result.sources, 1):
        date = c.date.split(" ")[0] if c.date else ""
        print(f"  {i}. [{c.category}] {c.title} ({date})")
        if c.link:
            print(f"     {c.link}")
    print()
    print(
        f"trends_model={result.trends_model} | "
        f"prompt_tokens={result.prompt_eval_count} "
        f"output_tokens={result.eval_count}"
    )


if __name__ == "__main__":
    main()
