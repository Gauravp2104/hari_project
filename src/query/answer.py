"""RAG answer pipeline using Anthropic Claude (Haiku by default).

Pipeline: retrieve top-k chunks from ChromaDB -> assemble prompt -> Claude messages ->
answer + citations.

Usage:
    python -m src.query.answer "what are the latest bioplastics trends?"
    python -m src.query.answer "who acquired whom recently?" --k 12
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import ANSWER_MODEL  # noqa: E402
from src.query.intent import Intent, chitchat_response, classify_intent  # noqa: E402
from src.query.llm_client import AnthropicChatClient  # noqa: E402
from src.rag.retrieve import RetrievedChunk, dedup_by_article, retrieve  # noqa: E402


SYSTEM_PROMPT = (
    "You are a research assistant that answers questions about the global packaging "
    "industry, drawing on a corpus of scraped news developments. You will be given a "
    "user question and a set of retrieved article excerpts.\n\n"
    "RULES:\n"
    "1. Answer ONLY using information present in the supplied excerpts. If the excerpts "
    "do not contain the answer, say so explicitly — do not fabricate or use prior knowledge.\n"
    "2. Write in flowing natural-language prose. Synthesize the excerpts into a coherent "
    "narrative rather than listing facts. Do NOT use bullet points, numbered lists, or "
    "markdown lists — use complete sentences and paragraphs that read like a written brief.\n"
    "3. When the question asks about trends or themes, weave the relevant developments "
    "together into a unified explanation rather than enumerating them separately.\n"
    "4. When the question is about a specific company, person, or policy, foreground the "
    "most relevant facts and integrate the supporting detail conversationally.\n"
    "5. Be concise but complete. Aim for a short paragraph (3-6 sentences) for focused "
    "questions, and two or three short paragraphs for broader synthesis questions.\n"
    "6. Use publication dates from the source headers only when recency materially affects "
    "the answer (e.g. 'as of early 2026...'), and phrase them naturally.\n"
    "7. Do NOT cite sources in any form — no bracketed IDs like [1], no parenthetical "
    "references, no 'according to source X' phrasing, and no 'Sources:' section. The UI "
    "renders the source list separately."
)


@dataclass
class AnswerResult:
    question: str
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    model: str = ""
    intent: str = Intent.RESEARCH.value
    prompt_eval_count: int = 0
    eval_count: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def _format_sources(chunks: list[RetrievedChunk], max_chars_each: int = 900) -> str:
    """Format retrieved chunks for the prompt. Truncate each to keep the prompt small."""
    lines = []
    for i, c in enumerate(chunks, 1):
        date = c.date.split(" ")[0] if c.date else ""
        header = f"[{i}] {c.title}"
        if date:
            header += f" ({date})"
        if c.category:
            header += f" — {c.category}"
        text = c.text if len(c.text) <= max_chars_each else c.text[:max_chars_each] + "…"
        lines.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(lines)


def answer_question(
    question: str,
    k: int = 8,
    category_filter: str | None = None,
    model: str = ANSWER_MODEL,
    system_prompt: str | None = None,
    client: AnthropicChatClient | None = None,
    **_ignored,
) -> AnswerResult:
    """Answer a research question using RAG over the corpus.

    `client` lets the caller pass an already-built AnthropicChatClient (e.g. shared
    across many questions). When omitted, a client is built per-call using
    `model` and `system_prompt` (defaults to SYSTEM_PROMPT).
    Extra kwargs are ignored for backward compatibility with old Ollama callers.
    """
    intent = classify_intent(question)
    if intent != Intent.RESEARCH:
        return AnswerResult(
            question=question,
            answer=chitchat_response(intent),
            sources=[],
            model="",
            intent=intent.value,
        )

    raw_chunks = retrieve(question, k=k * 2, category_filter=category_filter)
    chunks = dedup_by_article(raw_chunks)[:k]

    if not chunks:
        return AnswerResult(
            question=question,
            answer="I don't have any relevant articles in the corpus to answer that question.",
            sources=[],
            model=model,
            intent=intent.value,
        )

    context = _format_sources(chunks)
    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED EXCERPTS:\n{context}\n\n"
        "Answer the question using only the excerpts above, citing source IDs in brackets."
    )

    if client is None:
        client = AnthropicChatClient(
            model=model,
            system_prompt=system_prompt or SYSTEM_PROMPT,
        )
    response = client.chat(user_msg, system_prompt=system_prompt)

    return AnswerResult(
        question=question,
        answer=response.content,
        sources=chunks,
        model=client.model,
        intent=intent.value,
        prompt_eval_count=response.prompt_tokens,
        eval_count=response.output_tokens,
        input_tokens=response.prompt_tokens,
        output_tokens=response.output_tokens,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Ask a question against the developments corpus (Anthropic Claude)."
    )
    parser.add_argument("question", help="The natural-language question")
    parser.add_argument("--k", type=int, default=8, help="Number of articles to retrieve")
    parser.add_argument("--category", default=None, help="Restrict retrieval to one category")
    parser.add_argument("--model", default=ANSWER_MODEL, help=f"Claude model (default: {ANSWER_MODEL})")
    args = parser.parse_args()

    result = answer_question(
        args.question,
        k=args.k,
        category_filter=args.category,
        model=args.model,
    )
    print(result.answer)
    print()
    print("=" * 80)
    print("SOURCES:")
    for i, c in enumerate(result.sources, 1):
        date = c.date.split(" ")[0] if c.date else ""
        print(f"[{i}] {c.title} ({date}) — {c.category}")
        if c.link:
            print(f"    {c.link}")
    print()
    print(
        f"model={result.model} | prompt_tokens={result.prompt_eval_count} "
        f"output_tokens={result.eval_count}"
    )


if __name__ == "__main__":
    main()
