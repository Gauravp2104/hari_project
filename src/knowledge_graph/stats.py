"""Print summary stats for the persisted KG.

Usage:
    python -m src.knowledge_graph.stats
"""

from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import KG_DIR  # noqa: E402

from .schema import EdgeType, NodeType  # noqa: E402


def load_graph() -> nx.MultiDiGraph:
    p = KG_DIR / "graph.gpickle"
    if not p.exists():
        raise FileNotFoundError(
            f"Graph not found at {p}. Run: python -m src.knowledge_graph.build"
        )
    with p.open("rb") as f:
        return pickle.load(f)


def main():
    g = load_graph()
    print(f"Nodes: {g.number_of_nodes():,}")
    print(f"Edges: {g.number_of_edges():,}")
    print()

    node_types = Counter(d["type"] for _, d in g.nodes(data=True))
    print("Nodes by type:")
    for t, c in node_types.most_common():
        print(f"  {t:15s} {c:,}")
    print()

    edge_types = Counter(d["type"] for _, _, d in g.edges(data=True))
    print("Edges by type:")
    for t, c in edge_types.most_common():
        print(f"  {t:20s} {c:,}")
    print()

    article_count = node_types.get(NodeType.ARTICLE, 0)
    category_articles = Counter()
    for src, tgt, data in g.edges(data=True):
        if data["type"] == EdgeType.BELONGS_TO:
            category_articles[g.nodes[tgt]["name"]] += 1
    print(f"Articles per category (top 15 of {len(category_articles)}):")
    for cat, c in category_articles.most_common(15):
        print(f"  {cat:35s} {c:,}")
    print()

    entity_node_types = {NodeType.ORG, NodeType.PERSON, NodeType.LOCATION}
    mention_counts = Counter()
    for src, tgt, data in g.edges(data=True):
        if data["type"] == EdgeType.MENTIONS:
            mention_counts[tgt] += 1
    print("Top 20 entities by article mentions:")
    for ent_id, c in mention_counts.most_common(20):
        node = g.nodes[ent_id]
        print(f"  {node['type']:13s} {node['name']:40s} {c:,}")
    print()

    relation_pairs = Counter()
    for src, tgt, data in g.edges(data=True):
        if data["type"] in {EdgeType.MENTIONS, EdgeType.BELONGS_TO}:
            continue
        relation_pairs[(g.nodes[src]["name"], data["type"], g.nodes[tgt]["name"])] += 1
    print(f"Top 20 typed relations (of {len(relation_pairs)} unique):")
    for (s, rel, t), c in relation_pairs.most_common(20):
        print(f"  {s:30s} -[{rel}]-> {t:30s} ({c}x)")


if __name__ == "__main__":
    main()
