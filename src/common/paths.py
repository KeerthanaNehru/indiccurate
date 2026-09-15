"""Central path definitions for the IndicCurate pipeline.

All modules should import paths from here instead of hardcoding strings, so
that the on-disk layout only needs to change in one place.
"""

from pathlib import Path

# src/common/paths.py -> src/common -> src -> indiccurate (project root)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

MODELS_DIR: Path = PROJECT_ROOT / "models"

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"

AUDIO_DIR: Path = RAW_DIR / "audio"
NEWS_HTML_DIR: Path = RAW_DIR / "news_html"

STATS_PATH: Path = DATA_DIR / "stats.json"

TEXT_RAW_PARQUET: Path = RAW_DIR / "text_raw.parquet"
SPEECH_MANIFEST_PARQUET: Path = RAW_DIR / "speech_manifest.parquet"

TEXT_FILTERED_PARQUET: Path = INTERIM_DIR / "text_filtered.parquet"
TEXT_DEDUPED_PARQUET: Path = INTERIM_DIR / "text_deduped.parquet"

CONTAMINATION_REPORT: Path = PROCESSED_DIR / "contamination_report.json"
SPEECH_QA_REPORT: Path = PROCESSED_DIR / "speech_qa_report.json"
SPEECH_MANIFEST_FINAL_PARQUET: Path = PROCESSED_DIR / "speech_manifest_final.parquet"
ANNOTATIONS_PATH: Path = PROCESSED_DIR / "annotations.jsonl"


def ensure_dirs() -> None:
    """Create every directory referenced above if it does not already exist."""
    for d in (DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR, AUDIO_DIR, NEWS_HTML_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
