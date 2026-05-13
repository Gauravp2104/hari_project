"""Semantic retrieval against the ChromaDB collection."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import EMBEDDING_MODEL, VECTOR_STORE_DIR  # noqa: E402

from .embed import COLLECTION_NAME  # noqa: E402


@dataclass
class RetrievedChunk:
    article_id: str
    chunk_index: int
    title: str
    date: str
    category: str
    link: str
    text: str
    distance: float


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)


def retrieve(query: str, k: int = 8, category_filter: str | None = None) -> list[RetrievedChunk]:
    where = {"category": category_filter} if category_filter else None
    res = _collection().query(query_texts=[query], n_results=k, where=where)

    out: list[RetrievedChunk] = []
    if not res["ids"] or not res["ids"][0]:
        return out
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append(RetrievedChunk(
            article_id=meta["article_id"],
            chunk_index=meta["chunk_index"],
            title=meta["title"],
            date=meta["date"],
            category=meta["category"],
            link=meta["link"],
            text=doc,
            distance=dist,
        ))
    return out


def dedup_by_article(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep only the best (lowest distance) chunk per article. Preserves order."""
    seen: set[str] = set()
    out: list[RetrievedChunk] = []
    for c in chunks:
        if c.article_id in seen:
            continue
        seen.add(c.article_id)
        out.append(c)
    return out
