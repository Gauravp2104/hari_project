"""Chunk articles, embed with sentence-transformers, persist to ChromaDB.

Each chunk = a window of the Overview text (with Title + Key Takeaway prepended as
a header so retrieval has lexical anchors). Metadata carries article_id, title, date,
category, link — used by retrieve.py to assemble citations.

Usage:
    python -m src.rag.embed [--chunk-size 800] [--overlap 100] [--rebuild]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (  # noqa: E402
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    PROCESSED_DIR,
    VECTOR_STORE_DIR,
)


COLLECTION_NAME = "developments"


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Simple character-window chunker. Good enough for news-article-length text."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    step = size - overlap
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


def build_chunks(df: pd.DataFrame, size: int, overlap: int) -> list[dict]:
    chunks: list[dict] = []
    for _, row in df.iterrows():
        article_id = f"article:{row['Sr No']}"
        title = str(row["Title"] or "").strip()
        takeaway = str(row.get("Key Takeaway") or "").strip()
        category = str(row.get("Category") or "").strip()
        date = str(row["Date"] or "").strip()
        link = str(row.get("Link") or "").strip()
        body = str(row["Overview"] or "").strip()

        header_parts = [p for p in [title, takeaway] if p and p.lower() != "nan"]
        header = "\n\n".join(header_parts)

        for i, body_chunk in enumerate(chunk_text(body, size, overlap)):
            doc_text = (header + "\n\n" + body_chunk) if header else body_chunk
            chunks.append({
                "id": f"{article_id}#chunk{i}",
                "document": doc_text,
                "metadata": {
                    "article_id": article_id,
                    "chunk_index": i,
                    "title": title,
                    "date": date,
                    "category": category,
                    "link": link,
                },
            })
    return chunks


def get_collection(rebuild: bool = False):
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except (ValueError, Exception):
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def load_articles_df() -> pd.DataFrame:
    parquets = sorted(PROCESSED_DIR.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(
            f"No parquet in {PROCESSED_DIR}. Run: "
            "python -m src.ingestion.load_excel <xlsx> --header 1"
        )
    return pd.read_parquet(parquets[-1])


def main():
    parser = argparse.ArgumentParser(description="Chunk + embed developments into ChromaDB.")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Delete the existing collection before re-embedding",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Embedding batch size for ChromaDB add()",
    )
    args = parser.parse_args()

    df = load_articles_df()
    print(f"Loaded {len(df):,} articles", flush=True)

    chunks = build_chunks(df, size=args.chunk_size, overlap=args.overlap)
    print(f"Built {len(chunks):,} chunks (avg {len(chunks)/max(len(df),1):.1f} per article)", flush=True)

    collection = get_collection(rebuild=args.rebuild)
    existing = collection.count()
    print(f"Existing collection size: {existing:,}", flush=True)

    for i in range(0, len(chunks), args.batch_size):
        batch = chunks[i : i + args.batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["document"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
        if (i // args.batch_size) % 10 == 0 or i + args.batch_size >= len(chunks):
            print(f"  embedded {min(i+args.batch_size, len(chunks)):,}/{len(chunks):,}", flush=True)

    print(f"Done. Collection size: {collection.count():,}", flush=True)
    print(f"Persisted at: {VECTOR_STORE_DIR}", flush=True)


if __name__ == "__main__":
    main()
