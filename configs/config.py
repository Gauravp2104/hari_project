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

# Answering LLM (Anthropic Claude via API).
ANSWER_MODEL = "claude-haiku-4-5"
ANSWER_MAX_TOKENS = 1024
ANSWER_TEMPERATURE = 0.2

# KG relation extraction model (also Anthropic).
KG_RELATION_MODEL = "claude-haiku-4-5"

for _d in (RAW_DIR, PROCESSED_DIR, KG_DIR, VECTOR_STORE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
