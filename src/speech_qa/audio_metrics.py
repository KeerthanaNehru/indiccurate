"""Audio quality metrics: SNR, silence ratio, and basic statistics.

Computes Signal-to-Noise Ratio (SNR) and silence ratio for audio files.
These metrics help identify low-quality audio that should be filtered out.

Run standalone:
    python -m src.speech_qa.audio_metrics --manifest data/raw/speech_manifest.parquet \
        --output data/speech_qa/audio_metrics.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.common.paths import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SNR_THRESHOLD = 15.0  # dB
DEFAULT_SILENCE_THRESHOLD = 0.3  # 30% silence


def load_audio(audio_path: str) -> Optional[tuple[np.ndarray, int]]:
    """Load audio file and return (audio_data, sample_rate).

    Args:
        audio_path: Path to audio file (WAV, MP3, etc).

    Returns:
        Tuple of (audio_array, sample_rate), or None if error.
    """
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        return y, sr
    except Exception as e:
        logger.warning(f"Could not load {audio_path}: {e}")
        return None


def compute_snr(audio: np.ndarray, sr: int) -> float:
    """Compute Signal-to-Noise Ratio (SNR) in dB.

    Simple approach: divide audio into frames, estimate noise from quietest frames,
    signal from loudest frames. Then compute SNR = 10 * log10(P_signal / P_noise).

    Args:
        audio: Audio waveform (mono).
        sr: Sample rate.

    Returns:
        SNR in dB.
    """
    # Frame-based energy analysis
    frame_length = int(0.025 * sr)  # 25ms frames
    hop_length = frame_length // 2
    
    # Compute frame energy
    frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
    frame_energy = np.mean(frames ** 2, axis=0)
    
    if len(frame_energy) == 0:
        return 0.0
    
    # Estimate noise from quietest 10% of frames
    noise_threshold = np.percentile(frame_energy, 10)
    signal_threshold = np.percentile(frame_energy, 90)
    
    # Avoid division by zero
    if noise_threshold < 1e-10:
        noise_threshold = 1e-10
    if signal_threshold < 1e-10:
        signal_threshold = 1e-10
    
    snr_db = 10.0 * np.log10(signal_threshold / noise_threshold)
    return float(np.clip(snr_db, 0, 60))  # Clip to reasonable range


def compute_silence_ratio(audio: np.ndarray, sr: int, db_threshold: float = -40.0) -> float:
    """Compute fraction of audio below silence threshold.

    Args:
        audio: Audio waveform (mono).
        sr: Sample rate.
        db_threshold: Energy threshold in dB (relative to max).

    Returns:
        Fraction of audio below threshold (0-1).
    """
    # Convert to dB
    S = librosa.feature.melspectrogram(y=audio, sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    # Count frames below threshold
    silent_frames = np.sum(np.mean(S_db, axis=0) < db_threshold)
    total_frames = S_db.shape[1]
    
    if total_frames == 0:
        return 0.0
    
    return float(silent_frames / total_frames)


def compute_audio_metrics(audio_path: str) -> Optional[Dict[str, Any]]:
    """Compute audio quality metrics for a file.

    Args:
        audio_path: Path to audio file.

    Returns:
        Dict with SNR, silence_ratio, duration, sample_rate, or None if error.
    """
    result = load_audio(audio_path)
    if result is None:
        return None
    
    audio, sr = result
    
    try:
        snr = compute_snr(audio, sr)
        silence_ratio = compute_silence_ratio(audio, sr)
        duration = float(len(audio) / sr)
        
        return {
            "audio_path": audio_path,
            "snr_db": snr,
            "silence_ratio": silence_ratio,
            "duration_sec": duration,
            "sample_rate": sr,
        }
    except Exception as e:
        logger.warning(f"Error computing metrics for {audio_path}: {e}")
        return None


def analyze_manifest(manifest_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyze audio metrics for all records in manifest.

    Args:
        manifest_records: List of speech manifest records.

    Returns:
        List of metrics dicts (one per audio file).
    """
    metrics = []
    
    for i, record in enumerate(manifest_records):
        audio_path = record.get("audio_path", "")
        if not audio_path:
            logger.warning(f"Record {i} has no audio_path")
            continue
        
        if i % 10 == 0:
            logger.info(f"  Processing {i}/{len(manifest_records)}...")
        
        audio_metrics = compute_audio_metrics(audio_path)
        if audio_metrics:
            # Add metadata from manifest
            audio_metrics["source"] = record.get("source", "")
            audio_metrics["speaker_id"] = record.get("speaker_id", "")
            audio_metrics["transcript"] = record.get("transcript", "")
            metrics.append(audio_metrics)
    
    return metrics


def run(manifest_path: Path, output_path: Path) -> None:
    """Analyze audio quality metrics and save to JSONL.

    Args:
        manifest_path: Path to speech manifest Parquet.
        output_path: Path to save metrics JSONL.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed")
        return
    
    logger.info(f"Loading manifest from {manifest_path}...")
    df = pd.read_parquet(manifest_path)
    records = df.to_dict(orient="records")
    
    logger.info(f"Analyzing {len(records)} audio files...")
    metrics = analyze_manifest(records)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for metric in metrics:
            f.write(json.dumps(metric, ensure_ascii=False) + "\n")
    
    logger.info(f"Wrote {len(metrics)} metrics to {output_path}")


def main() -> None:
    """CLI entrypoint: compute audio metrics."""
    parser = argparse.ArgumentParser(description="Compute audio quality metrics (SNR, silence ratio).")
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/speech_manifest.parquet"))
    parser.add_argument("--output", type=Path, default=DATA_DIR / "speech_qa" / "audio_metrics.jsonl")
    args = parser.parse_args()

    run(manifest_path=args.manifest, output_path=args.output)


if __name__ == "__main__":
    main()
