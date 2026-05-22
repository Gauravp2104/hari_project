"""Central paths and settings for the research-trends RAG project."""

from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load local .env if present (no-op when env vars already set by the host, e.g. HF Spaces secrets).
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
KG_DIR = DATA_DIR / "knowledge_graph"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

OLLAMA_HOST = "http://localhost:11434"

# Stage-0 model: intent classification (chitchat vs research vs meta).
# `ollama pull deepseek-r1:1.5b` once, then `ollama serve`.
INTENT_MODEL = "deepseek-r1:1.5b"
INTENT_MAX_TOKENS = 64
INTENT_TEMPERATURE = 0.0

# Stage-1 model: category gate that drives development retrieval. Small local model.
# `ollama pull qwen2.5:3b` once, then `ollama serve`.
ANSWER_MODEL = "qwen2.5:3b"
ANSWER_MAX_TOKENS = 1024
ANSWER_TEMPERATURE = 0.2

# Stage-2 model: response / trend generation. Local mistral 7b.
# `ollama pull mistral:7b` once, then `ollama serve`.
TRENDS_MODEL = "mistral:7b"
TRENDS_OLLAMA_HOST = "http://localhost:11434"
TRENDS_MAX_TOKENS = 1024
TRENDS_TEMPERATURE = 0.3

# KG relation extraction model (still Anthropic — only needed for offline KG rebuilds).
KG_RELATION_MODEL = "claude-haiku-4-5"

for _d in (RAW_DIR, PROCESSED_DIR, KG_DIR, VECTOR_STORE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
