#!/usr/bin/env bash
set -euo pipefail

ollama serve >/tmp/ollama.log 2>&1 &

for i in {1..30}; do
    if curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then
        break
    fi
    sleep 1
done

ollama pull "${OLLAMA_MODEL:-qwen2.5:3b}"

exec streamlit run app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
