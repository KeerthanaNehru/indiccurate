"""Length-based filter: drops documents that are too short or too long.

Very short documents (stubs, nav-only scrapes) carry little training
signal; extremely long documents are often concatenation artifacts or
boilerplate-heavy dumps. Both thresholds are configurable.

Run standalone:
    python -m src.filtering.length_filter --input data/raw/text_raw.parquet \
        --output data/interim/text_length_filtered.parquet --min-chars 50 --max-chars 100000
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.common.io_utils import read_parquet, write_parquet
from src.common.paths import INTERIM_DIR, RAW_DIR
from src.common.stats import log_stage_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MIN_CHARS = 50
DEFAULT_MAX_CHARS = 100_000


def filter_by_length(
    records: List[Dict[str, Any]],
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split records into (kept, removed) based on character count of ``text``.

    Args:
        records: List of text records (each must have a ``"text"`` field).
        min_chars: Minimum character count to keep a record.
        max_chars: Maximum character count to keep a record.

    Returns:
        A ``(kept, removed)`` tuple of record lists. Kept records gain a
        ``char_count`` field and have ``pipeline_stage`` updated.
    """
    kept, removed = [], []
    for record in records:
        text = record.get("text") or ""
        n = len(text)
        record = dict(record)
        record["char_count"] = n
        if min_chars <= n <= max_chars:
            record["pipeline_stage"] = "filtered.length"
            kept.append(record)
        else:
            removed.append(record)
    return kept, removed


def run(input_path: Path, output_path: Path, min_chars: int, max_chars: int) -> None:
    """Load records, apply the length filter, and write the result.

    Args:
        input_path: Source ``.parquet`` file (must contain a ``"text"`` column).
        output_path: Destination ``.parquet`` file for records that pass the filter.
        min_chars: Minimum character count to keep a record.
        max_chars: Maximum character count to keep a record.
    """
    df = read_parquet(input_path)
    records = df.to_dict(orient="records")
    logger.info("Loaded %d records from %s", len(records), input_path)

    kept, removed = filter_by_length(records, min_chars=min_chars, max_chars=max_chars)

    write_parquet(kept, output_path)
    logger.info("Kept %d / %d records (min_chars=%d, max_chars=%d). Wrote %s", len(kept), len(records), min_chars, max_chars, output_path)

    log_stage_stats(
        stage="filtering.length",
        input_count=len(records),
        output_count=len(kept),
        removed_count=len(removed),
        extra={"min_chars": min_chars, "max_chars": max_chars},
    )


def main() -> None:
    """CLI entrypoint: apply the min/max character length filter to a Parquet file."""
    parser = argparse.ArgumentParser(description="Filter records by character length.")
    parser.add_argument("--input", type=Path, default=RAW_DIR / "text_raw.parquet")
    parser.add_argument("--output", type=Path, default=INTERIM_DIR / "text_length_filtered.parquet")
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    args = parser.parse_args()

    run(input_path=args.input, output_path=args.output, min_chars=args.min_chars, max_chars=args.max_chars)


if __name__ == "__main__":
    main()
