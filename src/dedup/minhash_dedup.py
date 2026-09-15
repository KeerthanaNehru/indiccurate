"""Near-duplicate detection using datasketch MinHash + LSH.

Builds a MinHash signature per document from character-level shingles
(language-agnostic — works the same for Tamil script as for Latin script),
then uses ``MinHashLSH`` to efficiently find candidate near-duplicate pairs
without an O(n^2) comparison. Within each connected cluster of near-dups, the
first-seen (by input order) document is kept and the rest are dropped.

Run standalone:
    python -m src.dedup.minhash_dedup --input data/interim/text_exact_deduped.parquet \
        --output data/interim/text_minhash_deduped.parquet --threshold 0.8
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasketch import MinHash, MinHashLSH

from src.common.io_utils import read_parquet, write_parquet
from src.common.paths import INTERIM_DIR
from src.common.stats import log_stage_stats
from src.dedup.exact_dedup import normalize_for_hash

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_NUM_PERM = 128
DEFAULT_NGRAM_SIZE = 5
DEFAULT_THRESHOLD = 0.8
# Shingling the full text of a long Wikipedia article produces tens of
# thousands of shingles for no real accuracy benefit at this document
# length — the leading portion is already representative for near-dup
# purposes, and capping it keeps this stage fast at demo/production scale.
MAX_CHARS_FOR_SHINGLING = 5000


def get_shingles(text: str, ngram_size: int = DEFAULT_NGRAM_SIZE) -> List[str]:
    """Split normalized text into overlapping character n-grams ("shingles").

    Character-level shingling is used (rather than word-level) so this works
    uniformly for Tamil script text without needing a language-specific
    tokenizer.

    Args:
        text: Raw text.
        ngram_size: Length of each character shingle.

    Returns:
        List of shingle strings (may be empty if text is shorter than ``ngram_size``).
    """
    normalized = normalize_for_hash(text)[:MAX_CHARS_FOR_SHINGLING]
    if len(normalized) < ngram_size:
        return [normalized] if normalized else []
    return [normalized[i : i + ngram_size] for i in range(len(normalized) - ngram_size + 1)]


def get_minhash(text: str, num_perm: int = DEFAULT_NUM_PERM, ngram_size: int = DEFAULT_NGRAM_SIZE) -> MinHash:
    """Compute a MinHash signature for a document's text.

    Args:
        text: Raw text.
        num_perm: Number of hash permutations (higher = more accurate, slower).
        ngram_size: Length of character shingles fed into the MinHash.

    Returns:
        A ``datasketch.MinHash`` signature.
    """
    m = MinHash(num_perm=num_perm)
    for shingle in get_shingles(text, ngram_size):
        m.update(shingle.encode("utf-8"))
    return m


def deduplicate_minhash(
    records: List[Dict[str, Any]],
    threshold: float = DEFAULT_THRESHOLD,
    num_perm: int = DEFAULT_NUM_PERM,
    ngram_size: int = DEFAULT_NGRAM_SIZE,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split records into (kept, removed) by near-duplicate MinHash/LSH clustering.

    Args:
        records: List of text records (each must have a ``"text"`` field).
        threshold: Jaccard similarity threshold above which two documents are
            considered near-duplicates.
        num_perm: Number of MinHash permutations.
        ngram_size: Character shingle length.

    Returns:
        A ``(kept, removed)`` tuple of record lists. Kept records gain a
        ``minhash_cluster_size`` field and have ``pipeline_stage`` updated.
        Removed records gain a ``near_dup_of`` field pointing at the index
        (within this batch) of the record they were deduplicated against.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes: List[MinHash] = []

    kept, removed = [], []
    for i, record in enumerate(records):
        mh = get_minhash(record.get("text", ""), num_perm=num_perm, ngram_size=ngram_size)
        minhashes.append(mh)
        key = str(i)

        candidates = lsh.query(mh)
        if candidates:
            # Near-duplicate of an already-kept document: drop, keep the earliest.
            rep_idx = min(int(c) for c in candidates)
            record = dict(record)
            record["near_dup_of"] = rep_idx
            removed.append(record)
        else:
            lsh.insert(key, mh)
            record = dict(record)
            record["pipeline_stage"] = "deduped.minhash"
            kept.append((i, record))

    kept_records = [r for _, r in kept]
    return kept_records, removed


def run(input_path: Path, output_path: Path, threshold: float, num_perm: int, ngram_size: int) -> None:
    """Load records, drop near-duplicates via MinHash/LSH, and write the result.

    Args:
        input_path: Source ``.parquet`` file (must contain a ``"text"`` column).
        output_path: Destination ``.parquet`` file for de-duplicated records.
        threshold: Jaccard similarity threshold for near-duplicate clustering.
        num_perm: Number of MinHash permutations.
        ngram_size: Character shingle length.
    """
    df = read_parquet(input_path)
    records = df.to_dict(orient="records")
    logger.info("Loaded %d records from %s", len(records), input_path)

    kept, removed = deduplicate_minhash(records, threshold=threshold, num_perm=num_perm, ngram_size=ngram_size)

    write_parquet(kept, output_path)
    logger.info("Kept %d / %d records (MinHash/LSH near-dedup, threshold=%.2f). Wrote %s", len(kept), len(records), threshold, output_path)

    log_stage_stats(
        stage="dedup.minhash",
        input_count=len(records),
        output_count=len(kept),
        removed_count=len(removed),
        extra={"threshold": threshold, "num_perm": num_perm, "ngram_size": ngram_size},
    )


def main() -> None:
    """CLI entrypoint: drop near-duplicates via MinHash/LSH from a Parquet file."""
    parser = argparse.ArgumentParser(description="Near-duplicate detection via datasketch MinHash + LSH.")
    parser.add_argument("--input", type=Path, default=INTERIM_DIR / "text_exact_deduped.parquet")
    parser.add_argument("--output", type=Path, default=INTERIM_DIR / "text_minhash_deduped.parquet")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Jaccard similarity threshold for near-dup clustering.")
    parser.add_argument("--num-perm", type=int, default=DEFAULT_NUM_PERM)
    parser.add_argument("--ngram-size", type=int, default=DEFAULT_NGRAM_SIZE)
    args = parser.parse_args()

    run(input_path=args.input, output_path=args.output, threshold=args.threshold, num_perm=args.num_perm, ngram_size=args.ngram_size)


if __name__ == "__main__":
    main()
