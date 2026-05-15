"""Streamlit interface for the packaging-developments RAG + KG system.

Caching strategy:
- @st.cache_resource: heavy singletons (Ollama client, KG load via lru_cache).
- @st.cache_data: per-query results (retrieval, answer). TTL keeps memory bounded.
- Disk-backed cache: answers persist across server restarts so reloads are instant.

Run:
    .venv/bin/streamlit run app.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs.config import ANSWER_MODEL  # noqa: E402
from src.query.answer import answer_question  # noqa: E402
from src.query.kg_lookup import find_entities_in_text, load_graph, profile  # noqa: E402
from src.query.kg_viz import render_for_query_entities, render_top_entities  # noqa: E402


ANSWER_CACHE_DIR = Path(__file__).resolve().parent / "data" / "answer_cache"
ANSWER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Packaging Developments", layout="wide", page_icon=":package:")


# ---------- Caching helpers ----------

@st.cache_resource(show_spinner=False)
def _kg_loaded() -> bool:
    return load_graph() is not None


def _cache_key(question: str, k: int, model: str, category: str | None) -> str:
    raw = json.dumps(
        {"q": question.strip().lower(), "k": k, "m": model, "c": category or ""},
        sort_keys=True,
    )
    return hashlib.sha1(raw.encode()).hexdigest()


def _disk_cache_get(key: str) -> dict | None:
    p = ANSWER_CACHE_DIR / f"{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
    return None


def _disk_cache_put(key: str, payload: dict) -> None:
    (ANSWER_CACHE_DIR / f"{key}.json").write_text(json.dumps(payload))


@st.cache_data(ttl=3600, show_spinner=False)
def cached_answer(question: str, k: int, category: str | None, model: str):
    """Two-tier cache: in-process (1h TTL) → disk (no expiry)."""
    key = _cache_key(question, k, model, category)
    disk = _disk_cache_get(key)
    if disk:
        return {**disk, "cache_layer": "disk"}

    result = answer_question(question, k=k, category_filter=category, model=model)
    payload = {
        "answer": result.answer,
        "sources": [
            {
                "article_id": s.article_id,
                "title": s.title,
                "date": s.date,
                "category": s.category,
                "link": s.link,
                "text": s.text,
            }
            for s in result.sources
        ],
        "model": result.model,
        "intent": result.intent,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cache_layer": "fresh",
    }
    _disk_cache_put(key, payload)
    return payload


@st.cache_data(ttl=3600, show_spinner=False)
def cached_query_entities(question: str) -> list[str]:
    return find_entities_in_text(question)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_query_subgraph_html(entity_ids_tuple: tuple[str, ...]) -> str | None:
    return render_for_query_entities(list(entity_ids_tuple), height_px=440)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_top_entities_html(top_n: int) -> str | None:
    return render_top_entities(top_n=top_n, height_px=650)


# ---------- UI ----------

st.title(":package: Packaging Developments — Research Assistant")
st.caption("Ask questions about latest trends in packaging research, or explore the knowledge graph.")

with st.sidebar:
    st.header("Settings")
    k = st.slider("Articles to retrieve", 3, 20, 8)
    show_kg_panel = st.checkbox("Show entity neighborhood for each question", value=True)
    model_name = st.text_input("Answer model (Claude)", value=ANSWER_MODEL)
    st.divider()

    if not _kg_loaded():
        st.warning("Knowledge graph not built — run `python -m src.knowledge_graph.build` first.")
    if st.button("Clear cached answers"):
        for f in ANSWER_CACHE_DIR.glob("*.json"):
            f.unlink()
        cached_answer.clear()
        cached_query_entities.clear()
        cached_query_subgraph_html.clear()
        cached_top_entities_html.clear()
        st.success("Cache cleared.")

    st.divider()
    st.subheader("KG legend")
    st.markdown(
        """
        <div style="font-size:13px;line-height:1.85;">
          <div><span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:#3b82f6;margin-right:8px;vertical-align:middle;border:2px solid transparent;"></span>Organization</div>
          <div><span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:#10b981;margin-right:8px;vertical-align:middle;"></span>Person</div>
          <div><span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:#ef4444;margin-right:8px;vertical-align:middle;"></span>Location</div>
          <div><span style="display:inline-block;width:12px;height:12px;background:#f59e0b;transform:rotate(45deg);margin-right:8px;vertical-align:middle;"></span>Category</div>
          <div><span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:#cbd5e1;margin-right:13px;margin-left:4px;vertical-align:middle;"></span>Article (small dot)</div>
          <div style="margin-top:6px;color:#6b7280;font-size:11px;">Bigger circle with border = entity from your question.</div>
          <div style="color:#6b7280;font-size:11px;">Thin grey lines = article ↔ entity mention.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


tab_ask, tab_explore = st.tabs([":speech_balloon: Ask", ":globe_with_meridians: Explore Graph"])

# ---------- Ask tab ----------

with tab_ask:
    if "history" not in st.session_state:
        st.session_state.history = []

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            if turn.get("sources"):
                with st.expander(f"Sources ({len(turn['sources'])})"):
                    for i, src in enumerate(turn["sources"], 1):
                        date = src["date"].split(" ")[0] if src["date"] else ""
                        line = f"**[{i}] {src['title']}**"
                        if date:
                            line += f" — *{date}*"
                        if src["category"]:
                            line += f" — `{src['category']}`"
                        st.markdown(line)
                        if src["link"]:
                            st.markdown(f"[Article link]({src['link']})")
                        st.caption(src["text"][:400] + ("…" if len(src["text"]) > 400 else ""))
                        st.divider()
            if turn.get("subgraph_html"):
                with st.expander("Knowledge graph: entity neighborhood"):
                    components.html(turn["subgraph_html"], height=440, scrolling=False)

    question = st.chat_input("Ask about a company, trend, regulation, or development...")
    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching corpus and generating answer..."):
                try:
                    result = cached_answer(question, k, None, model_name)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

                st.markdown(result["answer"])
                intent = result.get("intent", "research")
                cache_label = " (cached)" if result.get("cache_layer") == "disk" else ""
                if intent == "research":
                    st.caption(
                        f"model={result['model']} | "
                        f"prompt_tokens={result['input_tokens']} "
                        f"output_tokens={result['output_tokens']}{cache_label}"
                    )
                else:
                    st.caption(f"intent={intent} (no retrieval, no LLM call){cache_label}")

                if result["sources"]:
                    with st.expander(f"Sources ({len(result['sources'])})"):
                        for i, src in enumerate(result["sources"], 1):
                            date = src["date"].split(" ")[0] if src["date"] else ""
                            line = f"**[{i}] {src['title']}**"
                            if date:
                                line += f" — *{date}*"
                            if src["category"]:
                                line += f" — `{src['category']}`"
                            st.markdown(line)
                            if src["link"]:
                                st.markdown(f"[Article link]({src['link']})")
                            st.caption(src["text"][:400] + ("…" if len(src["text"]) > 400 else ""))
                            st.divider()

                subgraph_html: str | None = None
                if intent == "research" and show_kg_panel and _kg_loaded():
                    entity_ids = cached_query_entities(question)
                    if entity_ids:
                        st.markdown("**Entities found in your question:**")
                        cols = st.columns(min(len(entity_ids), 4))
                        for col, nid in zip(cols, entity_ids):
                            p = profile(nid)
                            if not p:
                                continue
                            with col:
                                st.markdown(f"**{p.name}**")
                                st.caption(f"{p.node_type} · {p.mention_count} article(s)")
                                for n in p.neighbors[:4]:
                                    arrow = "→" if n.direction == "out" else "←"
                                    st.caption(f"{arrow} `{n.relation}` {n.entity_name}")

                        subgraph_html = cached_query_subgraph_html(tuple(entity_ids))
                        if subgraph_html:
                            with st.expander("Knowledge graph: entity neighborhood", expanded=True):
                                components.html(subgraph_html, height=440, scrolling=False)

                st.session_state.history.append({
                    "question": question,
                    "answer": result["answer"],
                    "sources": result["sources"],
                    "subgraph_html": subgraph_html,
                })


# ---------- Explore Graph tab ----------

with tab_explore:
    if not _kg_loaded():
        st.warning("Knowledge graph not built. Run `python -m src.knowledge_graph.build` first.")
    else:
        ctrl, viz = st.columns([1, 4])
        with ctrl:
            st.subheader("Top entities")
            top_n = st.slider("How many top entities", 10, 60, 25)
            st.caption(
                "The most-mentioned entities plus the articles that connect them. "
                "Drag nodes to re-layout. Hover an article dot to see its title."
            )
            st.caption("Legend is in the sidebar.")
        with viz:
            html = cached_top_entities_html(top_n)
            if html:
                components.html(html, height=680, scrolling=False)
            else:
                st.info("No entities found in the graph.")
