"""Render NetworkX subgraphs as interactive pyvis HTML for embedding in Streamlit.

Visual design:
- Entities (Org/Person/Location) are the focus → larger, colored.
- Articles are connectors → small grey dots, no text label, MENTIONS edges drawn
  as thin lines without labels (the connection itself is the signal).
- Categories show domain grouping → medium amber diamonds.
- Typed relations (when present) get small labeled edges.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import networkx as nx
from pyvis.network import Network

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.knowledge_graph.schema import EdgeType, NodeType  # noqa: E402
from src.query.kg_lookup import load_graph  # noqa: E402


# Curated palette
NODE_COLORS = {
    NodeType.ARTICLE: "#cbd5e1",
    NodeType.CATEGORY: "#f59e0b",
    NodeType.ORG: "#3b82f6",
    NodeType.PERSON: "#10b981",
    NodeType.LOCATION: "#ef4444",
}

NODE_SHAPES = {
    NodeType.ARTICLE: "dot",
    NodeType.CATEGORY: "diamond",
    NodeType.ORG: "dot",
    NodeType.PERSON: "dot",
    NodeType.LOCATION: "dot",
}

NODE_SIZES = {
    NodeType.ARTICLE: 3,
    NodeType.CATEGORY: 12,
    NodeType.ORG: 14,
    NodeType.PERSON: 14,
    NodeType.LOCATION: 14,
}

SEED_SIZE_BOOST = 6  # query-seed entities get drawn slightly larger

STRUCTURAL_EDGES = {EdgeType.MENTIONS, EdgeType.BELONGS_TO}

# Heuristic noise filter for spaCy NER artifacts in this corpus
_NOISE_NAMES_LOWER = {
    "packaging insights", "us", "uk", "european", "f&b", "pet",
    "epr", "drs", "ppwr", "ai", "asia", "south", "north", "east", "west",
}
_ACRONYM_RE = re.compile(r"^[A-Z]{1,4}$")


def _is_low_quality_entity(name: str, node_type: str) -> bool:
    n = name.strip()
    if len(n) < 3:
        return True
    if n.lower() in _NOISE_NAMES_LOWER:
        return True
    if node_type == NodeType.ORG and _ACRONYM_RE.match(n):
        return True
    return False


def _has_typed_relations(g: nx.MultiDiGraph) -> bool:
    for _, _, data in g.edges(data=True):
        if data["type"] not in STRUCTURAL_EDGES:
            return True
    return False


def _top_entities(g: nx.MultiDiGraph, top_n: int) -> list[str]:
    counts: dict[str, int] = {}
    for _, tgt, data in g.edges(data=True):
        if data["type"] != EdgeType.MENTIONS:
            continue
        node = g.nodes[tgt]
        if _is_low_quality_entity(node.get("name", ""), node.get("type", "")):
            continue
        counts[tgt] = counts.get(tgt, 0) + 1
    return [nid for nid, _ in sorted(counts.items(), key=lambda x: -x[1])[:top_n]]


def top_entities_with_articles(
    g: nx.MultiDiGraph,
    top_n: int = 25,
    min_shared_entities: int = 2,
    max_articles: int = 60,
) -> nx.MultiDiGraph:
    """Top-N entities by mention count, plus the articles that connect them.

    Only articles mentioning >=`min_shared_entities` of the top entities are kept —
    that filters out single-entity articles which would be visual noise.
    """
    entity_ids = _top_entities(g, top_n)
    entity_set = set(entity_ids)

    article_to_top_entities: dict[str, set[str]] = {}
    for src, tgt, data in g.edges(data=True):
        if data["type"] != EdgeType.MENTIONS:
            continue
        if tgt in entity_set:
            article_to_top_entities.setdefault(src, set()).add(tgt)

    connecting_articles = sorted(
        ((aid, ents) for aid, ents in article_to_top_entities.items() if len(ents) >= min_shared_entities),
        key=lambda x: -len(x[1]),
    )[:max_articles]

    sub = nx.MultiDiGraph()
    for nid in entity_ids:
        sub.add_node(nid, **g.nodes[nid])
    for aid, _ in connecting_articles:
        sub.add_node(aid, **g.nodes[aid])
        for ent_id in article_to_top_entities[aid] & entity_set:
            sub.add_edge(aid, ent_id, type=EdgeType.MENTIONS)

    if _has_typed_relations(g):
        for src in entity_ids:
            for _, tgt, data in g.out_edges(src, data=True):
                if data["type"] in STRUCTURAL_EDGES:
                    continue
                if tgt in entity_set:
                    sub.add_edge(src, tgt, **data)

    return sub


def neighborhood_with_articles(
    g: nx.MultiDiGraph,
    seed_node_ids: list[str],
    max_total_nodes: int = 200,
) -> tuple[nx.MultiDiGraph, set[str]]:
    """For per-query view: include EVERY node connected to a seed.

    Returns (subgraph, seed_set). Seeds are returned so the renderer can highlight them.
    Caps at `max_total_nodes` total — if exceeded, articles are kept by descending
    count of seed entities they mention, then other entities by descending mention count.
    """
    seeds = [nid for nid in seed_node_ids if nid in g.nodes]
    seed_set = set(seeds)
    if not seeds:
        return nx.MultiDiGraph(), set()

    article_seed_counts: dict[str, int] = {}
    for nid in seeds:
        for src, _, data in g.in_edges(nid, data=True):
            if data["type"] == EdgeType.MENTIONS:
                article_seed_counts[src] = article_seed_counts.get(src, 0) + 1

    other_entity_counts: dict[str, int] = {}
    for aid in article_seed_counts:
        for _, tgt, data in g.out_edges(aid, data=True):
            if data["type"] != EdgeType.MENTIONS or tgt in seed_set:
                continue
            node = g.nodes[tgt]
            if _is_low_quality_entity(node.get("name", ""), node.get("type", "")):
                continue
            other_entity_counts[tgt] = other_entity_counts.get(tgt, 0) + 1

    budget = max_total_nodes - len(seed_set)
    article_quota = max(budget // 2, 30)
    entity_quota = budget - article_quota

    sorted_articles = sorted(article_seed_counts.items(), key=lambda x: -x[1])
    sorted_entities = sorted(other_entity_counts.items(), key=lambda x: -x[1])
    keep_articles = {aid for aid, _ in sorted_articles[:article_quota]}
    keep_entities = {nid for nid, _ in sorted_entities[:entity_quota]}

    keep = seed_set | keep_articles | keep_entities
    sub = g.subgraph(keep).copy()
    sub = sub.edge_subgraph(
        [(u, v, k) for u, v, k, d in sub.edges(keys=True, data=True) if d["type"] == EdgeType.MENTIONS]
    ).copy()
    return sub, seed_set


def render_html(
    sub: nx.MultiDiGraph,
    height_px: int = 500,
    highlight: set[str] | None = None,
) -> str:
    """Render a subgraph as standalone HTML using pyvis with clean styling."""
    highlight = highlight or set()
    net = Network(
        height=f"{height_px}px",
        width="100%",
        directed=True,
        bgcolor="#fafafa",
        font_color="#111827",
        notebook=False,
        cdn_resources="remote",
    )

    for nid, data in sub.nodes(data=True):
        node_type = data.get("type", "Unknown")
        full_name = data.get("name") or data.get("title") or nid

        if node_type == NodeType.ARTICLE:
            label = ""
            tooltip = full_name[:120] + ("…" if len(full_name) > 120 else "")
        else:
            label = full_name if len(full_name) <= 28 else full_name[:25] + "…"
            tooltip = f"{node_type}: {full_name}"

        size = NODE_SIZES.get(node_type, 10)
        if nid in highlight:
            size += SEED_SIZE_BOOST
        net.add_node(
            nid,
            label=label,
            title=tooltip,
            color=NODE_COLORS.get(node_type, "#9ca3af"),
            shape=NODE_SHAPES.get(node_type, "dot"),
            size=size,
            borderWidth=2 if nid in highlight else 0,
            borderWidthSelected=3,
        )

    for src, tgt, data in sub.edges(data=True):
        edge_type = data["type"]
        is_structural = edge_type == EdgeType.MENTIONS

        if is_structural:
            net.add_edge(
                src, tgt,
                color={"color": "#d1d5db", "opacity": 0.5},
                width=0.5,
                arrows={"to": {"enabled": False}},
            )
        else:
            evidence = data.get("evidence", "") or ""
            net.add_edge(
                src, tgt,
                label=edge_type.lower().replace("_", " "),
                title=evidence[:200] if evidence else edge_type,
                color={"color": "#6b7280"},
                width=1.5,
                arrows={"to": {"enabled": True, "scaleFactor": 0.5}},
            )

    net.set_options("""
    {
      "nodes": {
        "font": {"size": 10, "face": "Inter, system-ui, -apple-system, sans-serif", "color": "#111827"},
        "shadow": false
      },
      "edges": {
        "font": {"size": 7, "face": "Inter, system-ui, sans-serif", "color": "#6b7280", "align": "middle", "background": "rgba(250,250,250,0.85)", "strokeWidth": 0},
        "smooth": {"type": "continuous", "roundness": 0.2}
      },
      "interaction": {"hover": true, "tooltipDelay": 100, "navigationButtons": true, "hideEdgesOnDrag": true},
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -3000,
          "centralGravity": 0.2,
          "springLength": 80,
          "springConstant": 0.04,
          "damping": 0.5,
          "avoidOverlap": 0.6
        },
        "stabilization": {"iterations": 200}
      }
    }
    """)
    return net.generate_html(notebook=False)


def render_for_query_entities(
    entity_node_ids: list[str],
    height_px: int = 450,
) -> str | None:
    g = load_graph()
    if g is None or not entity_node_ids:
        return None
    sub, seeds = neighborhood_with_articles(g, entity_node_ids)
    if sub.number_of_nodes() == 0:
        return None
    return render_html(sub, height_px=height_px, highlight=seeds)


def render_top_entities(top_n: int = 25, height_px: int = 600) -> str | None:
    g = load_graph()
    if g is None:
        return None
    sub = top_entities_with_articles(g, top_n=top_n)
    if sub.number_of_nodes() == 0:
        return None
    return render_html(sub, height_px=height_px)
