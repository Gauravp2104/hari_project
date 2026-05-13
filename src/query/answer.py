"""RAG answer pipeline using a local Ollama model (Mistral 7B by default).

Pipeline: retrieve top-k chunks from ChromaDB -> assemble prompt -> ollama.chat ->
answer + citations. Keeps the same `answer_question()` signature as before so the
Streamlit app needs no changes.

Usage:
    python -m src.query.answer "what are the latest bioplastics trends?"
    python -m src.query.answer "who acquired whom recently?" --k 12
    python -m src.query.answer "..." --model qwen2.5:3b
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import OLLAMA_HOST, OLLAMA_MODEL  # noqa: E402
from src.query.intent import Intent, chitchat_response, classify_intent  # noqa: E402
from src.query.llm_client import OllamaChatClient  # noqa: E402
from src.rag.retrieve import RetrievedChunk, dedup_by_article, retrieve  # noqa: E402


SYSTEM_PROMPT = (
    "You are a research assistant that answers questions about the global packaging "
    "industry, drawing on a corpus of scraped news developments. You will be given a "
    "user question and a set of retrieved article excerpts. Each excerpt is labeled "
    "with a numeric source ID like [1], [2], etc.\n\n"
    "RULES:\n"
    "1. Answer ONLY using information present in the supplied excerpts. If the excerpts "
    "do not contain the answer, say so explicitly — do not fabricate or use prior knowledge.\n"
    "2. Cite every claim with the bracketed source ID(s), e.g. 'ProAmpac acquired TC "
    "Transcontinental Packaging for $1.51B [3].'\n"
    "3. When the question asks about trends or themes, synthesize across multiple sources "
    "and cite all of them.\n"
    "4. When the question is about a specific company, person, or policy, prioritize the "
    "most relevant sources and quote specific facts.\n"
    "5. Be concise. Aim for 3-6 sentences for focused questions, slightly longer for "
    "synthesis questions. Use bullet lists when listing multiple discrete items.\n"
    "6. Use the publication dates in the source headers to flag recency where it matters "
    "(e.g. 'as of January 2026...').\n"
    "7. Do NOT include a 'Sources:' section — the bracketed citations are sufficient; the "
    "UI will render the source list separately."
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
    model: str = OLLAMA_MODEL,
    host: str = OLLAMA_HOST,
    system_prompt: str | None = None,
    client: OllamaChatClient | None = None,
) -> AnswerResult:
    """Answer a research question using RAG over the corpus.

    `client` lets the caller pass an already-built OllamaChatClient (e.g. shared
    across many questions). When omitted, a client is built per-call using
    `model`, `host`, and `system_prompt` (defaults to SYSTEM_PROMPT).
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
        client = OllamaChatClient(
            model=model,
            host=host,
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
        description="Ask a question against the developments corpus (local Ollama)."
    )
    parser.add_argument("question", help="The natural-language question")
    parser.add_argument("--k", type=int, default=8, help="Number of articles to retrieve")
    parser.add_argument("--category", default=None, help="Restrict retrieval to one category")
    parser.add_argument("--model", default=OLLAMA_MODEL, help=f"Ollama model (default: {OLLAMA_MODEL})")
    parser.add_argument("--host", default=OLLAMA_HOST, help=f"Ollama host (default: {OLLAMA_HOST})")
    args = parser.parse_args()

    result = answer_question(
        args.question,
        k=args.k,
        category_filter=args.category,
        model=args.model,
        host=args.host,
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
