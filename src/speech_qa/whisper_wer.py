"""Whisper transcription and Word Error Rate (WER) computation.

Re-transcribes audio using OpenAI's Whisper (tiny model for speed)
and computes WER against provided transcripts to assess transcription quality.

Run standalone:
    python -m src.speech_qa.whisper_wer --manifest data/raw/speech_manifest.parquet \
        --output data/speech_qa/whisper_results.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.common.paths import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "tiny"  # Fast inference on CPU


def load_whisper_model(model_size: str = DEFAULT_MODEL):
    """Load Whisper model.

    Args:
        model_size: Model size (tiny, base, small, medium, large).

    Returns:
        Loaded Whisper model, or None if error.
    """
    try:
        import whisper
        logger.info(f"Loading Whisper {model_size} model...")
        model = whisper.load_model(model_size)
        return model
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        return None


def transcribe_audio(model, audio_path: str, language: str = "ta") -> Optional[str]:
    """Transcribe audio file using Whisper.

    Args:
        model: Loaded Whisper model.
        audio_path: Path to audio file.
        language: Language code (ta for Tamil).

    Returns:
        Transcribed text, or None if error.
    """
    try:
        result = model.transcribe(audio_path, language=language, verbose=False)
        return result.get("text", "").strip()
    except Exception as e:
        logger.warning(f"Failed to transcribe {audio_path}: {e}")
        return None


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER) between reference and hypothesis.

    WER = (S + D + I) / N, where:
    - S = substitutions
    - D = deletions
    - I = insertions
    - N = words in reference

    Args:
        reference: Reference transcript.
        hypothesis: Hypothesis transcript.

    Returns:
        WER as fraction (0-1), or 1.0 if no reference.
    """
    try:
        import jiwer
        wer = jiwer.wer(reference, hypothesis)
        return float(wer)
    except ImportError:
        logger.warning("jiwer not installed. Install with: pip install jiwer")
        return 1.0
    except Exception as e:
        logger.warning(f"Error computing WER: {e}")
        return 1.0


def compute_mer(reference: str, hypothesis: str) -> float:
    """Compute Match Error Rate (MER) - proportion of words not matched.

    Args:
        reference: Reference transcript.
        hypothesis: Hypothesis transcript.

    Returns:
        MER as fraction (0-1).
    """
    try:
        import jiwer
        mer = jiwer.mer(reference, hypothesis)
        return float(mer)
    except Exception:
        return 1.0


def process_audio_file(
    model,
    record: Dict[str, Any],
    language: str = "ta",
) -> Optional[Dict[str, Any]]:
    """Transcribe audio and compute WER against provided transcript.

    Args:
        model: Loaded Whisper model.
        record: Audio manifest record.
        language: Language code.

    Returns:
        Dict with transcription and WER metrics, or None if error.
    """
    audio_path = record.get("audio_path", "")
    reference_transcript = record.get("transcript", "")
    
    if not audio_path:
        logger.warning("Record has no audio_path")
        return None
    
    # Transcribe with Whisper
    hypothesis = transcribe_audio(model, audio_path, language=language)
    if not hypothesis:
        logger.warning(f"Failed to transcribe {audio_path}")
        return None
    
    # Compute WER if reference available
    wer = 1.0
    mer = 1.0
    if reference_transcript:
        wer = compute_wer(reference_transcript, hypothesis)
        mer = compute_mer(reference_transcript, hypothesis)
    
    return {
        "audio_path": audio_path,
        "reference_transcript": reference_transcript,
        "whisper_transcript": hypothesis,
        "wer": wer,
        "mer": mer,
        "source": record.get("source", ""),
        "speaker_id": record.get("speaker_id", ""),
    }


def process_manifest(
    model,
    manifest_records: List[Dict[str, Any]],
    language: str = "ta",
) -> List[Dict[str, Any]]:
    """Process all audio files in manifest.

    Args:
        model: Loaded Whisper model.
        manifest_records: List of manifest records.
        language: Language code.

    Returns:
        List of transcription results.
    """
    results = []
    
    for i, record in enumerate(manifest_records):
        if i % 5 == 0:
            logger.info(f"  Processed {i}/{len(manifest_records)}...")
        
        result = process_audio_file(model, record, language=language)
        if result:
            results.append(result)
    
    return results


def run(manifest_path: Path, output_path: Path, model_size: str = DEFAULT_MODEL) -> None:
    """Transcribe audio files and compute WER, save results to JSONL.

    Args:
        manifest_path: Path to speech manifest Parquet.
        output_path: Path to save results JSONL.
        model_size: Whisper model size.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed")
        return
    
    # Load model
    model = load_whisper_model(model_size)
    if not model:
        logger.error("Could not load Whisper model")
        return
    
    logger.info(f"Loading manifest from {manifest_path}...")
    df = pd.read_parquet(manifest_path)
    records = df.to_dict(orient="records")
    
    logger.info(f"Transcribing {len(records)} audio files with Whisper {model_size}...")
    results = process_manifest(model, records, language="ta")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    logger.info(f"Wrote {len(results)} results to {output_path}")


def main() -> None:
    """CLI entrypoint: transcribe audio and compute WER."""
    parser = argparse.ArgumentParser(description="Transcribe audio with Whisper and compute WER.")
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/speech_manifest.parquet"))
    parser.add_argument("--output", type=Path, default=DATA_DIR / "speech_qa" / "whisper_results.jsonl")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Whisper model size (tiny, base, small, etc.)")
    args = parser.parse_args()

    run(manifest_path=args.manifest, output_path=args.output, model_size=args.model)


if __name__ == "__main__":
    main()
