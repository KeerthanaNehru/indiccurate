#!/bin/bash
# IndicCurate environment activation script
# Source this in your terminal: source ./activate_env.sh

# Activate venv
source /tmp/indiccurate_venv/bin/activate

echo "✅ IndicCurate environment activated!"
echo ""
echo "Available commands:"
echo "  • Dashboard: streamlit run src/annotation/app.py"
echo "  • Audio Download: python -m src.ingestion.speech_downloader --limit 200"
echo "  • Speech QA: python -m src.speech_qa.pipeline"
echo ""
echo "Current Python: $(which python)"
echo ""
