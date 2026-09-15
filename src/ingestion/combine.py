"""Combine all raw text sources into a single data/raw/text_raw.parquet.

Merges the Hugging Face streaming output (Wikipedia + OSCAR-derived) and the
Scrapy news output into one Parquet file with a unified schema. The speech
manifest does not need a separate combine step since
``src.ingestion.speech_downloader`` already writes a single combined
manifest across both speech sources.

Run standalone:
    python -m src.ingestion.combine
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.common.io_utils import read_jsonl, read_parquet, write_parquet
from src.common.paths import RAW_DIR, TEXT_RAW_PARQUET
from src.common.stats import log_stage_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run(hf_text_path: Path, news_jsonl_path: Path, output_path: Path) -> None:
    """Merge the HF-streamed and scraped text sources into one Parquet file.

    Args:
        hf_text_path: Parquet file produced by ``src.ingestion.hf_loader``.
        news_jsonl_path: JSONL file produced by ``src.ingestion.news_spider``.
        output_path: Destination combined ``.parquet`` file.
    """
    records: List[Dict[str, Any]] = []
    per_source_counts: Dict[str, int] = {}

    if hf_text_path.exists():
        hf_df = read_parquet(hf_text_path)
        hf_records = hf_df.to_dict(orient="records")
        records.extend(hf_records)
        per_source_counts["hf_text"] = len(hf_records)
        logger.info("Loaded %d records from %s", len(hf_records), hf_text_path)
    else:
        logger.warning("HF text file not found at %s — skipping.", hf_text_path)
        per_source_counts["hf_text"] = 0

    if news_jsonl_path.exists():
        news_records = list(read_jsonl(news_jsonl_path))
        records.extend(news_records)
        per_source_counts["news"] = len(news_records)
        logger.info("Loaded %d records from %s", len(news_records), news_jsonl_path)
    else:
        logger.warning("News JSONL file not found at %s — skipping.", news_jsonl_path)
        per_source_counts["news"] = 0

    if not records:
        raise RuntimeError("No input text records found — run hf_loader and/or news_spider first.")

    write_parquet(records, output_path)
    logger.info("Combined %d total text records into %s", len(records), output_path)

    log_stage_stats(
        stage="ingestion.combine",
        input_count=sum(per_source_counts.values()),
        output_count=len(records),
        removed_count=0,
        extra=per_source_counts,
    )


def main() -> None:
    """CLI entrypoint: combine raw text sources into text_raw.parquet."""
    parser = argparse.ArgumentParser(description="Combine raw text sources into data/raw/text_raw.parquet.")
    parser.add_argument("--hf-text", type=Path, default=RAW_DIR / "hf_text.parquet")
    parser.add_argument("--news-jsonl", type=Path, default=RAW_DIR / "news_raw.jsonl")
    parser.add_argument("--output", type=Path, default=TEXT_RAW_PARQUET)
    args = parser.parse_args()

    run(hf_text_path=args.hf_text, news_jsonl_path=args.news_jsonl, output_path=args.output)


if __name__ == "__main__":
    main()
