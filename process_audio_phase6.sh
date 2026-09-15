#!/bin/bash
# Phase 6: Audio Processing with Whisper

export PYTHONPATH="$(pwd)/venv_packages:$(pwd):$PYTHONPATH"

cd /Users/keerthana.n/Documents/Data_Curator_Project/indiccurate

echo "🎵 PHASE 6: AUDIO QUALITY ASSESSMENT & PROCESSING"
echo "=================================================="
echo ""

# Check if Whisper is installed
if python3 -c "import whisper" 2>/dev/null; then
    echo "✅ Whisper is ready"
else
    echo "📦 Installing openai-whisper..."
    python3 -m pip install -q openai-whisper --target ./venv_packages
    if [ $? -eq 0 ]; then
        echo "✅ Whisper installed!"
    else
        echo "⚠️  Whisper install had issues, but proceeding..."
    fi
fi

echo ""
echo "🎬 Processing 100 audio files..."
echo "   - Computing SNR (Signal-to-Noise Ratio)"
echo "   - Analyzing silence ratio"
echo "   - Transcribing with Whisper (tiny model)"
echo "   - Computing WER (Word Error Rate)"
echo "   - Analyzing speaker diversity"
echo ""

# Run Phase 6 pipeline
python3 -m src.speech_qa.pipeline

echo ""
echo "✅ Phase 6 Complete!"
echo ""
echo "📊 Results saved to:"
echo "   • data/speech_qa/speech_qa_report.json"
echo "   • data/speech_qa/speaker_stats.json"
echo "   • data/speech_qa/audio_metrics.jsonl"
