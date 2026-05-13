# Packaging Developments — Research Assistant

A retrieval-augmented research assistant over a corpus of scraped packaging-industry news. Combines:

- **A vector store** (ChromaDB + sentence-transformers) for semantic retrieval over article text
- **A knowledge graph** (NetworkX, optionally enriched with Claude-extracted typed relations) of organizations, people, locations, and the articles they appear in
- **A local LLM** (Ollama / Mistral 7B by default) for grounded, cited answers
- **A Streamlit UI** with a chat interface and an interactive graph explorer

The included sample dataset is `Packaging_Developments_25-26.xlsx` — ~1,600 articles scraped from packaginginsights.com.

---

## Architecture

```
Excel (raw)                 data/raw/
   |
   v  src/ingestion/load_excel.py
parquet (processed)         data/processed/
   |
   +--> src/rag/embed.py
   |       chunk + sentence-transformers
   |       --> ChromaDB                            data/vector_store/
   |
   +--> src/knowledge_graph/build.py
           spaCy NER --> entity nodes
           Claude (optional) --> typed relations
           --> NetworkX graph                      data/knowledge_graph/

User question
   |
   v  app.py / src/query/answer.py
[ retrieve top-k chunks ] --> [ Ollama chat ] --> answer + source citations
                          |
                          +--> KG entity neighborhood (sidebar / inline viz)
```

---

## Prerequisites

| Requirement | Why | Install |
|---|---|---|
| **Python 3.12** | Bedrock for the whole stack | https://www.python.org/downloads/ |
| **Ollama** | Hosts the local LLM used for answering | https://ollama.com/download |
| **Mistral 7B** (Ollama model) | Default question-answering model | `ollama pull mistral:7b` |
| **spaCy English model** | NER for the knowledge graph | `python -m spacy download en_core_web_sm` |
| **Anthropic API key** *(optional)* | Only needed if you want typed relations (ACQUIRED, PARTNERED_WITH, etc.) in the KG. Without it, the KG still has entity nodes and article-mediated connections — you just don't get LLM-extracted relations. | https://console.anthropic.com/settings/keys |

> **Python version note.** Python 3.14 is too new for some dependencies (`spacy`, `chromadb`, `sentence-transformers` may lack wheels). Use 3.12.

---

## Setup

### 1. Clone the repo

```bash
git clone <repo-url> hari_project
cd hari_project
```

### 2. Create and activate a virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate    # on macOS/Linux
# .venv\Scripts\activate     # on Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Install and start Ollama

```bash
# install Ollama from https://ollama.com/download then:
ollama pull mistral:7b
ollama serve     # leave running in a separate terminal (or it auto-starts on macOS)
```

Verify Ollama is reachable:

```bash
curl -s http://localhost:11434/api/tags
```

### 5. (Optional) Set up Anthropic API key for KG relation extraction

If you want typed relation edges in the KG (recommended for the most useful graph):

```bash
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
export ANTHROPIC_API_KEY=$(grep ANTHROPIC_API_KEY .env | cut -d= -f2)
```

> **Without a key**, the KG build will still produce a useful graph (entity nodes + article-mediated edges) — the relation-extraction step will fail silently for each article and the typed relations will be absent. Question answering does not require this key.

### 6. Place the Excel dataset

Put your scraped-developments Excel file in `data/raw/`:

```bash
cp /path/to/your/Packaging_Developments_25-26.xlsx data/raw/
```

If you don't have a dataset, the loader works with any Excel of news-style data with at least these columns: `Sr No`, `Category`, `Date`, `Title`, `Overview`, `Link` (and an optional `Key Takeaway`).

---

## Build the pipeline

Run these once after the initial setup. Each step persists its output to `data/`, so subsequent app launches start instantly.

### 1. Load the Excel into a parquet snapshot

```bash
python -m src.ingestion.load_excel data/raw/Packaging_Developments_25-26.xlsx --header 1
```

The `--header 1` flag skips the banner row in the source file. The loader prints a per-column summary; verify the columns are recognized (`Title`, `Overview`, `Category`, etc.).

### 2. Build the vector store (~5–10 min)

```bash
python -m src.rag.embed
```

Chunks each article's `Overview`, embeds with `sentence-transformers/all-MiniLM-L6-v2`, and persists to `data/vector_store/`.

### 3. Build the knowledge graph (~5 min for entities; +10–20 min if relations enabled)

Entity-only (no API key required):

```bash
python -m src.knowledge_graph.build --skip-relations
```

Full graph with typed relations (requires `ANTHROPIC_API_KEY`):

```bash
python -m src.knowledge_graph.build
```

> Relation extraction is **resumable**. Per-article results are cached at `data/processed/relations_cache/`. If the run is interrupted (Ctrl+C, network blip), re-running will skip already-processed articles. Total Anthropic Haiku cost for the full ~1,600-article corpus: roughly **$3–5** thanks to prompt caching.

### 4. Inspect the graph (optional)

```bash
python -m src.knowledge_graph.stats
```

Prints node/edge counts, top-mentioned entities, top relations, and articles-per-category.

---

## Run the app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (default: http://localhost:8501).

The UI has two tabs:

- **Ask** — chat-style input. Each answer cites sources by `[1]`, `[2]`, …; expand the Sources panel for titles, dates, categories, and links. When entities from the KG appear in your question, an interactive subgraph appears below the answer showing all articles and other entities they connect to.
- **Explore Graph** — top N entities by mention count, plus the articles that connect them. Drag nodes to re-layout; hover for tooltips. The legend lives in the sidebar.

### Sidebar controls

- **Articles to retrieve** — top-k for the RAG step
- **Show entity neighborhood** — toggle the per-question graph
- **Ollama model** — swap models on the fly without restarting (e.g., `qwen2.5:3b`, `deepseek-r1:7b`)
- **Clear cached answers** — wipes the disk + in-memory answer caches

---

## Command-line usage (no UI)

Ask a one-shot question:

```bash
python -m src.query.answer "what are the latest bioplastics trends?"
python -m src.query.answer "who acquired what packaging company?" --k 12
python -m src.query.answer "..." --model qwen2.5:3b
```

---

## Project layout

```
hari_project/
├── app.py                           # Streamlit UI
├── requirements.txt
├── .env.example                     # template for ANTHROPIC_API_KEY (optional)
├── configs/
│   └── config.py                    # paths, model names, chunking params
├── data/
│   ├── raw/                         # original Excel files (gitignored)
│   ├── processed/                   # parquet snapshots + relation cache
│   ├── knowledge_graph/             # graph.gpickle + graph.graphml
│   ├── vector_store/                # ChromaDB persistent store
│   └── answer_cache/                # disk-cached LLM answers
├── src/
│   ├── ingestion/
│   │   └── load_excel.py            # Excel -> parquet
│   ├── knowledge_graph/
│   │   ├── schema.py                # node/edge type constants
│   │   ├── entities.py              # spaCy NER + normalization
│   │   ├── relations.py             # Claude relation extraction (Haiku 4.5)
│   │   ├── build.py                 # KG pipeline orchestrator
│   │   └── stats.py                 # graph summary
│   ├── rag/
│   │   ├── embed.py                 # chunk + embed -> ChromaDB
│   │   └── retrieve.py              # semantic retrieval
│   └── query/
│       ├── answer.py                # RAG answer pipeline (Ollama)
│       ├── kg_lookup.py             # KG entity neighborhood lookup
│       └── kg_viz.py                # NetworkX -> pyvis HTML
└── tests/
```

---

## Caching behavior

| Cache | Location | Persistent across restart? |
|---|---|---|
| LLM answers | `data/answer_cache/<sha1>.json` | Yes |
| Vector embeddings | `data/vector_store/` | Yes |
| Knowledge graph | `data/knowledge_graph/graph.gpickle` | Yes |
| Per-article relation extraction | `data/processed/relations_cache/<id>.json` | Yes |
| HuggingFace model weights | `~/.cache/huggingface/` | Yes |
| Streamlit in-memory caches (KG load, viz HTML) | RAM, 1h TTL | No |

Asking the same question twice returns the cached answer instantly. Use the **Clear cached answers** button (or delete `data/answer_cache/`) if you change the system prompt or rebuild the corpus.

---

## Troubleshooting

**`TypeError: Could not resolve authentication method`** when running `python -m src.query.answer`
You're hitting the Anthropic SDK because the answer module was previously configured for Claude. Make sure `src/query/answer.py` uses `ollama.Client` (current default). If you want to use Anthropic for answers, set `ANTHROPIC_API_KEY` first.

**`ConnectError: All connection attempts failed` from Ollama**
Ollama isn't running. Start it with `ollama serve` (or open the Ollama desktop app on macOS).

**`spaCy model 'en_core_web_sm' not installed`**
Run `python -m spacy download en_core_web_sm` inside your activated venv.

**`No parquet in data/processed/`**
You haven't run the loader yet. See *Build the pipeline → 1*.

**`pyarrow.lib.ArrowTypeError: Expected bytes, got 'int'`** when loading Excel
The header row is wrong. Re-run with `--header 1` (or `--header 2` for files with a multi-row banner).

**The KG only has `MENTIONS` edges, no typed relations**
The Anthropic relation-extraction step didn't run successfully. Set `ANTHROPIC_API_KEY` and re-run `python -m src.knowledge_graph.build` (the per-article cache means already-extracted articles are skipped; failed articles will be re-tried).

**Streamlit shows old behavior after I edited a module**
In-process caches and module imports are sticky. Stop Streamlit (`Ctrl+C`) and restart `streamlit run app.py`.

**Python 3.14 install errors (`spacy`, `chromadb`, etc.)**
Recreate the venv on Python 3.12:
```bash
deactivate && rm -rf .venv
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## License

Add your preferred license here (MIT, Apache-2.0, etc.).
