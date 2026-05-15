#!/usr/bin/env bash
set -euo pipefail

exec streamlit run app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
