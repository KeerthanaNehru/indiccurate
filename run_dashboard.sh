#!/bin/bash
# Direct dashboard launcher

# Use local venv packages
export PYTHONPATH="$(pwd)/venv_packages:$(pwd):$PYTHONPATH"

cd /Users/keerthana.n/Documents/Data_Curator_Project/indiccurate

echo "🚀 Starting IndicCurate Dashboard..."
echo ""
echo "📱 Open in your browser:"
echo "   http://localhost:8501"
echo ""

python3 -m streamlit run src/annotation/app.py --client.toolbarMode=minimal
