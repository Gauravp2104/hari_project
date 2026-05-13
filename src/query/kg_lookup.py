"""KG-side lookups: find entities mentioned in a query, pull their neighborhood."""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import KG_DIR  # noqa: E402

from src.knowledge_graph.schema import EdgeType, NodeType  # noqa: E402


@dataclass
class EntityNeighbor:
    relation: str
    direction: str
    entity_name: str
    entity_type: str
    evidence: str = ""


@dataclass
class EntityProfile:
    name: str
    node_type: str
    mention_count: int
    neighbors: list[EntityNeighbor]


@lru_cache(maxsize=1)
def load_graph() -> nx.MultiDiGraph | None:
    p = KG_DIR / "graph.gpickle"
    if not p.exists():
        return None
    with p.open("rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _entity_index() -> dict[str, str]:
    """Lowercased entity name -> node id, for case-insensitive lookup."""
    g = load_graph()
    if g is None:
        return {}
    return {
        d["name"].lower(): nid
        for nid, d in g.nodes(data=True)
        if d.get("type") in {NodeType.ORG, NodeType.PERSON, NodeType.LOCATION}
        and d.get("name")
    }


def find_entities_in_text(text: str, max_n: int = 5) -> list[str]:
    """Return entity node IDs whose names appear (case-insensitive) in the text."""
    g = load_graph()
    if g is None:
        return []
    text_lower = text.lower()
    hits: list[tuple[int, str]] = []
    for name_lower, nid in _entity_index().items():
        if len(name_lower) < 3:
            continue
        if name_lower in text_lower:
            hits.append((len(name_lower), nid))
    hits.sort(key=lambda x: -x[0])
    seen: set[str] = set()
    out: list[str] = []
    for _, nid in hits:
        if nid in seen:
            continue
        seen.add(nid)
        out.append(nid)
        if len(out) >= max_n:
            break
    return out


def profile(node_id: str) -> EntityProfile | None:
    g = load_graph()
    if g is None or node_id not in g.nodes:
        return None
    node = g.nodes[node_id]

    mention_count = sum(
        1 for _, _, d in g.in_edges(node_id, data=True) if d["type"] == EdgeType.MENTIONS
    )

    neighbors: list[EntityNeighbor] = []
    for src, tgt, data in g.out_edges(node_id, data=True):
        if data["type"] in {EdgeType.MENTIONS, EdgeType.BELONGS_TO}:
            continue
        tgt_node = g.nodes[tgt]
        neighbors.append(EntityNeighbor(
            relation=data["type"],
            direction="out",
            entity_name=tgt_node["name"],
            entity_type=tgt_node["type"],
            evidence=data.get("evidence", ""),
        ))
    for src, tgt, data in g.in_edges(node_id, data=True):
        if data["type"] in {EdgeType.MENTIONS, EdgeType.BELONGS_TO}:
            continue
        src_node = g.nodes[src]
        neighbors.append(EntityNeighbor(
            relation=data["type"],
            direction="in",
            entity_name=src_node["name"],
            entity_type=src_node["type"],
            evidence=data.get("evidence", ""),
        ))

    return EntityProfile(
        name=node["name"],
        node_type=node["type"],
        mention_count=mention_count,
        neighbors=neighbors,
    )
