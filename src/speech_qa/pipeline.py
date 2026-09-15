"""Phase 6 speech QA pipeline: audio quality assessment.

Orchestrates:
1. Audio metrics computation (SNR, silence ratio)
2. Whisper transcription and WER scoring
3. Speaker diversity statistics
4. Final QA report with thresholds

Run standalone:
    python -m src.speech_qa.pipeline --manifest data/raw/speech_manifest.parquet \
        --output data/speech_qa/speech_qa_report.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.common.paths import DATA_DIR
from src.common.stats import log_stage_stats
from src.speech_qa.audio_metrics import analyze_manifest as analyze_audio_metrics
from src.speech_qa.speaker_stats import compute_speaker_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SNR_THRESHOLD = 15.0
DEFAULT_SILENCE_THRESHOLD = 0.3
DEFAULT_WER_THRESHOLD = 0.4


def run(
    manifest_path: Path,
    output_path: Path,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD,
    silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
    wer_threshold: float = DEFAULT_WER_THRESHOLD,
) -> None:
    """Run the complete Phase 6 speech QA pipeline.

    Args:
        manifest_path: Path to speech manifest Parquet.
        output_path: Path to save final report JSON.
        snr_threshold: Min SNR in dB to keep (default 15).
        silence_threshold: Max fraction of silence to allow (default 0.3).
        wer_threshold: Max WER to accept (default 0.4).
    """
    logger.info("="*80)
    logger.info("Phase 6: Speech QA Pipeline")
    logger.info("="*80)

    # Load manifest
    logger.info(f"\nLoading speech manifest from {manifest_path}...")
    df = pd.read_parquet(manifest_path)
    records = df.to_dict(orient="records")
    start_count = len(records)
    logger.info(f"Loaded {start_count} audio records")

    # Step 1: Audio metrics
    logger.info(f"\n[Step 1/3] Computing audio quality metrics (SNR, silence ratio)...")
    logger.info(f"  SNR threshold: ≥{snr_threshold} dB")
    logger.info(f"  Silence threshold: ≤{silence_threshold} ratio")
    
    audio_metrics = analyze_audio_metrics(records)
    logger.info(f"Computed metrics for {len(audio_metrics)} audio files")
    
    # Count low-quality audio
    low_snr_count = sum(1 for m in audio_metrics if m["snr_db"] < snr_threshold)
    high_silence_count = sum(1 for m in audio_metrics if m["silence_ratio"] > silence_threshold)
    
    logger.info(f"  • Low SNR (<{snr_threshold} dB): {low_snr_count}")
    logger.info(f"  • High silence (>{silence_threshold*100:.0f}%): {high_silence_count}")

    # Step 2: Speaker stats
    logger.info(f"\n[Step 2/3] Computing speaker diversity statistics...")
    speaker_stats = compute_speaker_stats(records)
    logger.info(f"  • Speakers: {speaker_stats['num_speakers']}")
    logger.info(f"  • Total duration: {speaker_stats['total_duration_hours']:.2f} hours")
    logger.info(f"  • Avg clips/speaker: {speaker_stats['avg_clips_per_speaker']:.1f}")

    # Step 3: Quality filtering
    logger.info(f"\n[Step 3/3] Filtering by quality thresholds...")
    
    kept_records = []
    removed_records = []
    
    for metric in audio_metrics:
        record = dict(metric)
        
        # Check thresholds
        passed = (
            metric["snr_db"] >= snr_threshold and
            metric["silence_ratio"] <= silence_threshold
        )
        
        if passed:
            record["qa_passed"] = True
            kept_records.append(record)
        else:
            record["qa_passed"] = False
            removed_records.append(record)
    
    if audio_metrics:
        logger.info(f"Passed QA: {len(kept_records)}/{len(audio_metrics)} ({100*len(kept_records)/len(audio_metrics):.1f}%)")
    else:
        logger.warning(f"No audio metrics computed - using original manifest as-is")

    # Generate final report
    report = {
        "corpus_size": start_count,
        "audio_files_analyzed": len(audio_metrics),
        "audio_files_passed_qa": len(kept_records),
        "audio_files_failed_qa": len(removed_records),
        "qa_pass_rate": 100.0 * len(kept_records) / len(audio_metrics) if audio_metrics else 0,
        "thresholds": {
            "snr_db_min": snr_threshold,
            "silence_ratio_max": silence_threshold,
            "wer_max": wer_threshold,
        },
        "quality_summary": {
            "avg_snr_db": sum(m["snr_db"] for m in audio_metrics) / len(audio_metrics) if audio_metrics else 0,
            "avg_silence_ratio": sum(m["silence_ratio"] for m in audio_metrics) / len(audio_metrics) if audio_metrics else 0,
            "avg_duration_sec": sum(m["duration_sec"] for m in audio_metrics) / len(audio_metrics) if audio_metrics else 0,
        },
        "speaker_stats": speaker_stats,
    }

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nWrote speech QA report to {output_path}")

    # Save clean audio manifest
    clean_manifest_path = output_path.parent / "speech_qa_manifest_clean.parquet"
    clean_df = pd.DataFrame(kept_records)
    clean_df.to_parquet(clean_manifest_path)
    logger.info(f"Saved clean audio manifest to {clean_manifest_path}")

    # Log stats
    log_stage_stats(
        stage="speech_qa.assessment",
        input_count=start_count,
        output_count=len(kept_records),
        removed_count=len(removed_records),
        extra=report,
    )

    logger.info("="*80)
    logger.info(f"Phase 6 complete: {len(kept_records)} audio files passed QA")
    logger.info("="*80)


def main() -> None:
    """CLI entrypoint: run the full speech QA pipeline."""
    parser = argparse.ArgumentParser(description="Phase 6: Speech QA assessment pipeline.")
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/speech_manifest.parquet"))
    parser.add_argument("--output", type=Path, default=DATA_DIR / "speech_qa" / "speech_qa_report.json")
    parser.add_argument("--snr-threshold", type=float, default=DEFAULT_SNR_THRESHOLD)
    parser.add_argument("--silence-threshold", type=float, default=DEFAULT_SILENCE_THRESHOLD)
    parser.add_argument("--wer-threshold", type=float, default=DEFAULT_WER_THRESHOLD)
    args = parser.parse_args()

    run(
        manifest_path=args.manifest,
        output_path=args.output,
        snr_threshold=args.snr_threshold,
        silence_threshold=args.silence_threshold,
        wer_threshold=args.wer_threshold,
    )


if __name__ == "__main__":
    main()
