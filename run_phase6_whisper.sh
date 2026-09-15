#!/bin/bash
# Phase 6: Speech QA with Whisper installation

export PYTHONPATH="$(pwd)/venv_packages:$(pwd):$PYTHONPATH"

cd /Users/keerthana.n/Documents/Data_Curator_Project/indiccurate

echo "📦 Installing openai-whisper..."
echo "This may take 2-3 minutes..."
echo ""

# Install whisper
python3 -m pip install -q openai-whisper --target ./venv_packages 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Whisper installed successfully!"
    echo ""
    echo "🎵 Running Phase 6: Speech QA Pipeline..."
    echo ""
    python3 -m src.speech_qa.pipeline
else
    echo "❌ Whisper installation failed"
    echo "Try: pip install openai-whisper"
fi

