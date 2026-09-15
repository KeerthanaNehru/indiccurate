"""Speaker diversity statistics from audio manifest.

Computes statistics on speaker distribution, audio duration per speaker,
and gender/accent diversity (if metadata available).

Run standalone:
    python -m src.speech_qa.speaker_stats --manifest data/raw/speech_manifest.parquet \
        --output data/speech_qa/speaker_stats.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.common.paths import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_speaker_stats(manifest_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute speaker diversity statistics.

    Args:
        manifest_records: List of manifest records with speaker metadata.

    Returns:
        Dict with speaker statistics.
    """
    # Collect speaker information
    speaker_info: Dict[str, Dict[str, Any]] = {}
    
    total_duration = 0.0
    for record in manifest_records:
        speaker_id = record.get("speaker_id", "unknown")
        duration = float(record.get("duration_sec", 0))
        
        if speaker_id not in speaker_info:
            speaker_info[speaker_id] = {
                "count": 0,
                "total_duration": 0.0,
                "source": record.get("source", ""),
                "metadata": {},
            }
        
        speaker_info[speaker_id]["count"] += 1
        speaker_info[speaker_id]["total_duration"] += duration
        total_duration += duration
        
        # Store metadata (gender, accent, age if available)
        for key in ["gender", "accent", "age", "region"]:
            if key in record and record[key]:
                speaker_info[speaker_id]["metadata"][key] = record[key]
    
    # Compute diversity metrics
    num_speakers = len(speaker_info)
    avg_clips_per_speaker = len(manifest_records) / num_speakers if num_speakers > 0 else 0
    
    # Gender distribution
    genders = {}
    for speaker in speaker_info.values():
        gender = speaker["metadata"].get("gender", "unknown")
        if gender not in genders:
            genders[gender] = 0
        genders[gender] += speaker["count"]
    
    # Accent distribution
    accents = {}
    for speaker in speaker_info.values():
        accent = speaker["metadata"].get("accent", "unknown")
        if accent not in accents:
            accents[accent] = 0
        accents[accent] += speaker["count"]
    
    # Duration distribution
    duration_percentiles = []
    durations = sorted([s["total_duration"] for s in speaker_info.values()])
    if len(durations) > 0:
        duration_percentiles = [
            durations[len(durations) // 4],  # 25th percentile
            durations[len(durations) // 2],  # median
            durations[3 * len(durations) // 4],  # 75th percentile
        ]
    
    return {
        "num_speakers": num_speakers,
        "total_clips": len(manifest_records),
        "avg_clips_per_speaker": avg_clips_per_speaker,
        "total_duration_hours": total_duration / 3600,
        "avg_duration_per_speaker_hours": total_duration / num_speakers / 3600 if num_speakers > 0 else 0,
        "duration_percentiles": {
            "p25": duration_percentiles[0] if len(duration_percentiles) > 0 else 0,
            "p50": duration_percentiles[1] if len(duration_percentiles) > 1 else 0,
            "p75": duration_percentiles[2] if len(duration_percentiles) > 2 else 0,
        },
        "gender_distribution": genders,
        "accent_distribution": accents,
        "speaker_info": speaker_info,
    }


def run(manifest_path: Path, output_path: Path) -> None:
    """Compute speaker statistics and save to JSON.

    Args:
        manifest_path: Path to speech manifest Parquet.
        output_path: Path to save statistics JSON.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed")
        return
    
    logger.info(f"Loading manifest from {manifest_path}...")
    df = pd.read_parquet(manifest_path)
    records = df.to_dict(orient="records")
    
    logger.info(f"Computing speaker statistics for {len(records)} clips...")
    stats = compute_speaker_stats(records)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Wrote speaker statistics to {output_path}")
    logger.info(f"  • Speakers: {stats['num_speakers']}")
    logger.info(f"  • Total duration: {stats['total_duration_hours']:.1f} hours")
    logger.info(f"  • Avg clips/speaker: {stats['avg_clips_per_speaker']:.1f}")


def main() -> None:
    """CLI entrypoint: compute speaker statistics."""
    parser = argparse.ArgumentParser(description="Compute speaker diversity statistics.")
    parser.add_argument("--manifest", type=Path, default=Path("data/raw/speech_manifest.parquet"))
    parser.add_argument("--output", type=Path, default=DATA_DIR / "speech_qa" / "speaker_stats.json")
    args = parser.parse_args()

    run(manifest_path=args.manifest, output_path=args.output)


if __name__ == "__main__":
    main()
