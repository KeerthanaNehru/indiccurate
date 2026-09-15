"""Exact deduplication via content hashing.

Normalizes each document's text (collapsed whitespace, lowercased) and hashes
it with SHA-256. Documents whose normalized hash has already been seen are
dropped, keeping the first occurrence.

Run standalone:
    python -m src.dedup.exact_dedup --input data/interim/text_filtered.parquet \
        --output data/interim/text_exact_deduped.parquet
"""

import argparse
import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from src.common.io_utils import read_parquet, write_parquet
from src.common.paths import INTERIM_DIR
from src.common.stats import log_stage_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_for_hash(text: str) -> str:
    """Normalize text for hashing: collapse whitespace and lowercase.

    Args:
        text: Raw text.

    Returns:
        Normalized text suitable for stable content hashing.
    """
    return _WHITESPACE_RE.sub(" ", (text or "").strip().lower())


def compute_content_hash(text: str) -> str:
    """Compute a stable SHA-256 hash of a document's normalized text.

    Args:
        text: Raw text.

    Returns:
        Hex-encoded SHA-256 digest of the normalized text.
    """
    normalized = normalize_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def deduplicate_exact(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split records into (kept, removed) by exact content-hash duplication.

    The first occurrence of each hash is kept; later occurrences are removed.

    Args:
        records: List of text records (each must have a ``"text"`` field).

    Returns:
        A ``(kept, removed)`` tuple of record lists. Kept records gain a
        ``content_hash`` field and have ``pipeline_stage`` updated.
    """
    seen: Set[str] = set()
    kept, removed = [], []
    for record in records:
        content_hash = compute_content_hash(record.get("text", ""))
        record = dict(record)
        record["content_hash"] = content_hash
        if content_hash in seen:
            removed.append(record)
        else:
            seen.add(content_hash)
            record["pipeline_stage"] = "deduped.exact"
            kept.append(record)
    return kept, removed


def run(input_path: Path, output_path: Path) -> None:
    """Load records, drop exact content duplicates, and write the result.

    Args:
        input_path: Source ``.parquet`` file (must contain a ``"text"`` column).
        output_path: Destination ``.parquet`` file for de-duplicated records.
    """
    df = read_parquet(input_path)
    records = df.to_dict(orient="records")
    logger.info("Loaded %d records from %s", len(records), input_path)

    kept, removed = deduplicate_exact(records)

    write_parquet(kept, output_path)
    logger.info("Kept %d / %d records (exact dedup). Wrote %s", len(kept), len(records), output_path)

    log_stage_stats(
        stage="dedup.exact",
        input_count=len(records),
        output_count=len(kept),
        removed_count=len(removed),
    )


def main() -> None:
    """CLI entrypoint: drop exact content duplicates from a Parquet file."""
    parser = argparse.ArgumentParser(description="Exact deduplication via content hashing.")
    parser.add_argument("--input", type=Path, default=INTERIM_DIR / "text_filtered.parquet")
    parser.add_argument("--output", type=Path, default=INTERIM_DIR / "text_exact_deduped.parquet")
    args = parser.parse_args()

    run(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
