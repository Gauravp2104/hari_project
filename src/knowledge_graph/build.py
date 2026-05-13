"""Build the packaging-developments knowledge graph from the processed parquet.

Pipeline:
  1. Read parquet
  2. Add Article + Category nodes (article -BELONGS_TO-> category)
  3. spaCy NER on Overview text -> Entity nodes + (article -MENTIONS-> entity)
  4. Claude relation extraction -> typed edges between entities
  5. Persist to data/knowledge_graph/ as both .gpickle (full fidelity) and .graphml (viz)

Usage:
    python -m src.knowledge_graph.build [--limit N] [--skip-relations] [--concurrency N]
"""

from __future__ import annotations

import argparse
import asyncio
import pickle
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import PROCESSED_DIR, KG_DIR  # noqa: E402

from . import entities, relations  # noqa: E402
from .schema import EdgeType, NodeType  # noqa: E402


REQUIRED_COLUMNS = {"Sr No", "Category", "Date", "Title", "Overview", "Link"}


def load_articles() -> pd.DataFrame:
    parquets = sorted(PROCESSED_DIR.glob("*.parquet"))
    if not parquets:
        raise FileNotFoundError(
            f"No parquet files in {PROCESSED_DIR}. Run the loader first:\n"
            "  python -m src.ingestion.load_excel <path-to-xlsx> --header 1"
        )
    df = pd.read_parquet(parquets[-1])
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Parquet is missing required columns: {sorted(missing)}.\n"
            f"Columns found: {list(df.columns)}\n"
            "Re-run the loader with the right --header (likely --header 1)."
        )
    return df


def _article_node_id(sr_no) -> str:
    return f"article:{sr_no}"


def _category_node_id(category: str) -> str:
    return f"category:{category.strip().lower()}"


def _entity_node_id(canonical: str, node_type: str) -> str:
    return f"{node_type.lower()}:{canonical.lower()}"


def add_article_and_category_nodes(g: nx.MultiDiGraph, df: pd.DataFrame) -> None:
    for _, row in df.iterrows():
        article_id = _article_node_id(row["Sr No"])
        g.add_node(
            article_id,
            type=NodeType.ARTICLE,
            sr_no=str(row["Sr No"]),
            title=str(row["Title"]),
            date=str(row["Date"]),
            link=str(row["Link"]),
            key_takeaway=str(row.get("Key Takeaway") or ""),
        )
        category = str(row["Category"]).strip()
        if category and category.lower() != "nan":
            cat_id = _category_node_id(category)
            if not g.has_node(cat_id):
                g.add_node(cat_id, type=NodeType.CATEGORY, name=category)
            g.add_edge(article_id, cat_id, type=EdgeType.BELONGS_TO)


def add_entities(
    g: nx.MultiDiGraph, df: pd.DataFrame
) -> dict[str, list[entities.Entity]]:
    """Run spaCy NER on each article and add entity nodes + MENTIONS edges.

    Returns a map from article_id to its extracted entities (used for relation extraction).
    """
    print("Loading spaCy model...", flush=True)
    nlp = entities.load_model()
    print(f"Extracting entities from {len(df):,} articles...", flush=True)

    article_entities: dict[str, list[entities.Entity]] = {}
    for i, (_, row) in enumerate(df.iterrows(), 1):
        article_id = _article_node_id(row["Sr No"])
        text = str(row["Overview"] or "")
        ents = entities.extract(text, nlp=nlp)
        article_entities[article_id] = ents
        for ent in ents:
            ent_id = _entity_node_id(ent.canonical, ent.node_type)
            if not g.has_node(ent_id):
                g.add_node(ent_id, type=ent.node_type, name=ent.canonical)
            g.add_edge(article_id, ent_id, type=EdgeType.MENTIONS, mention=ent.mention)
        if i % 200 == 0 or i == len(df):
            print(f"  [{i}/{len(df)}] articles", flush=True)
    return article_entities


def add_relations(
    g: nx.MultiDiGraph,
    df: pd.DataFrame,
    article_entities: dict[str, list[entities.Entity]],
    concurrency: int,
) -> None:
    """Use Claude to extract typed relations between co-mentioned entities."""
    cache_dir = PROCESSED_DIR / "relations_cache"

    jobs: list[tuple[str, str, list[str]]] = []
    canonical_to_node: dict[tuple[str, str], str] = {}

    for _, row in df.iterrows():
        article_id = _article_node_id(row["Sr No"])
        ents = article_entities.get(article_id, [])
        if len(ents) < 2:
            continue
        names = [e.canonical for e in ents]
        for e in ents:
            canonical_to_node[(article_id, e.canonical)] = _entity_node_id(
                e.canonical, e.node_type
            )
        text = str(row["Overview"] or "")
        jobs.append((article_id, text, names))

    print(
        f"Extracting relations for {len(jobs):,} articles with "
        f"{concurrency} concurrent requests...",
        flush=True,
    )
    results = asyncio.run(
        relations.extract_all(jobs, cache_dir=cache_dir, concurrency=concurrency)
    )

    edges_added = 0
    for r in results:
        if r.error:
            continue
        for rel in r.relations:
            src_node = canonical_to_node.get((r.article_id, rel["source"]))
            tgt_node = canonical_to_node.get((r.article_id, rel["target"]))
            if not src_node or not tgt_node:
                continue
            g.add_edge(
                src_node,
                tgt_node,
                type=rel["relation"],
                evidence=rel["evidence"],
                source_article=r.article_id,
            )
            edges_added += 1

    total_input = sum(r.input_tokens for r in results)
    total_output = sum(r.output_tokens for r in results)
    total_cache_read = sum(r.cache_read_tokens for r in results)
    total_cache_create = sum(r.cache_creation_tokens for r in results)
    errors = sum(1 for r in results if r.error)
    print(
        f"Added {edges_added:,} relation edges. "
        f"Errors: {errors}. "
        f"Tokens: in={total_input:,} out={total_output:,} "
        f"cache_read={total_cache_read:,} cache_create={total_cache_create:,}",
        flush=True,
    )


def save_graph(g: nx.MultiDiGraph) -> tuple[Path, Path]:
    KG_DIR.mkdir(parents=True, exist_ok=True)
    pickle_path = KG_DIR / "graph.gpickle"
    graphml_path = KG_DIR / "graph.graphml"

    with pickle_path.open("wb") as f:
        pickle.dump(g, f)

    g_for_export = g.copy()
    for _, _, data in g_for_export.edges(data=True):
        for k, v in list(data.items()):
            if v is None:
                data[k] = ""
    nx.write_graphml(g_for_export, graphml_path)
    return pickle_path, graphml_path


def main():
    parser = argparse.ArgumentParser(description="Build the packaging-developments KG.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N articles (useful for testing)",
    )
    parser.add_argument(
        "--skip-relations", action="store_true",
        help="Skip the LLM relation extraction step (entities + categories only)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=relations.DEFAULT_CONCURRENCY,
        help=f"Concurrent Anthropic requests (default: {relations.DEFAULT_CONCURRENCY})",
    )
    args = parser.parse_args()

    df = load_articles()
    if args.limit:
        df = df.head(args.limit)
    print(f"Loaded {len(df):,} articles.", flush=True)

    g = nx.MultiDiGraph()
    add_article_and_category_nodes(g, df)
    print(
        f"After category pass: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges",
        flush=True,
    )

    article_entities = add_entities(g, df)
    print(
        f"After entity pass:   {g.number_of_nodes()} nodes, {g.number_of_edges()} edges",
        flush=True,
    )

    if not args.skip_relations:
        add_relations(g, df, article_entities, concurrency=args.concurrency)
        print(
            f"After relation pass: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges",
            flush=True,
        )

    pickle_path, graphml_path = save_graph(g)
    print(f"Saved {pickle_path}")
    print(f"Saved {graphml_path}")


if __name__ == "__main__":
    main()
