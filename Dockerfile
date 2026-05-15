FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps + Ollama (install as root)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates bash procps zstd && \
    curl -fsSL https://ollama.com/install.sh | sh && \
    rm -rf /var/lib/apt/lists/*

# HF Spaces requires running as a non-root user (uid 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    OLLAMA_HOST=http://127.0.0.1:11434 \
    OLLAMA_MODELS=/home/user/.ollama/models \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /home/user/app

COPY --chown=user:user requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

COPY --chown=user:user . .

RUN mkdir -p data/answer_cache data/vector_store data/knowledge_graph data/processed

EXPOSE 7860

CMD ["bash", "start.sh"]
