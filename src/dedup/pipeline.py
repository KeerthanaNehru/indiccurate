"""Phase 3 deduplication pipeline: chains all three dedup layers.

Order: exact content-hash dedup -> MinHash/LSH near-dup dedup -> semantic
(embedding similarity) dedup. Each layer logs its own funnel entry to
``stats.json``, and the final surviving records are written to
``data/interim/text_deduped.parquet``.

Every layer is also runnable standalone (``python -m src.dedup.<module>``);
this module just chains their core functions in-memory.

Run standalone:
    python -m src.dedup.pipeline --input data/interim/text_filtered.parquet \
        --output data/interim/text_deduped.parquet
"""

import argparse
import logging
from pathlib import Path

from src.common.io_utils import read_parquet, write_parquet
from src.common.paths import INTERIM_DIR, TEXT_DEDUPED_PARQUET
from src.common.stats import log_stage_stats
from src.dedup.exact_dedup import deduplicate_exact
from src.dedup.minhash_dedup import DEFAULT_NGRAM_SIZE, DEFAULT_NUM_PERM
from src.dedup.minhash_dedup import DEFAULT_THRESHOLD as MINHASH_DEFAULT_THRESHOLD
from src.dedup.minhash_dedup import deduplicate_minhash
from src.dedup.semantic_dedup import DEFAULT_BATCH_SIZE, DEFAULT_MODEL_NAME, DEFAULT_TOP_K
from src.dedup.semantic_dedup import DEFAULT_THRESHOLD as SEMANTIC_DEFAULT_THRESHOLD
from src.dedup.semantic_dedup import deduplicate_semantic

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run(
    input_path: Path,
    output_path: Path,
    minhash_threshold: float = MINHASH_DEFAULT_THRESHOLD,
    minhash_num_perm: int = DEFAULT_NUM_PERM,
    minhash_ngram_size: int = DEFAULT_NGRAM_SIZE,
    semantic_model_name: str = DEFAULT_MODEL_NAME,
    semantic_threshold: float = SEMANTIC_DEFAULT_THRESHOLD,
    semantic_top_k: int = DEFAULT_TOP_K,
    semantic_batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """Run the full Phase 3 dedup pipeline and write text_deduped.parquet.

    Args:
        input_path: Source ``.parquet`` file (typically ``data/interim/text_filtered.parquet``).
        output_path: Destination ``.parquet`` file for the deduplicated corpus.
        minhash_threshold: Jaccard similarity threshold for MinHash/LSH near-dup clustering.
        minhash_num_perm: Number of MinHash permutations.
        minhash_ngram_size: Character shingle length for MinHash.
        semantic_model_name: Hugging Face model id for the sentence-transformers encoder.
        semantic_threshold: Cosine similarity threshold for semantic near-dup pairs.
        semantic_top_k: Number of nearest neighbors to inspect per document.
        semantic_batch_size: Encoding batch size for the semantic dedup stage.
    """
    df = read_parquet(input_path)
    records = df.to_dict(orient="records")
    start_count = len(records)
    logger.info("=== Phase 3 dedup pipeline: %d input records from %s ===", start_count, input_path)

    # Stage 1: exact content-hash dedup
    records, removed = deduplicate_exact(records)
    log_stage_stats(
        stage="dedup.exact",
        input_count=start_count,
        output_count=len(records),
        removed_count=len(removed),
    )
    logger.info("[1/3] exact:    kept %d, removed %d", len(records), len(removed))

    # Stage 2: MinHash + LSH near-dup dedup
    prev_count = len(records)
    records, removed = deduplicate_minhash(
        records, threshold=minhash_threshold, num_perm=minhash_num_perm, ngram_size=minhash_ngram_size
    )
    log_stage_stats(
        stage="dedup.minhash",
        input_count=prev_count,
        output_count=len(records),
        removed_count=len(removed),
        extra={"threshold": minhash_threshold, "num_perm": minhash_num_perm, "ngram_size": minhash_ngram_size},
    )
    logger.info("[2/3] minhash:  kept %d, removed %d", len(records), len(removed))

    # Stage 3: semantic (embedding similarity) dedup
    prev_count = len(records)
    records, removed = deduplicate_semantic(
        records, model_name=semantic_model_name, threshold=semantic_threshold, top_k=semantic_top_k, batch_size=semantic_batch_size
    )
    log_stage_stats(
        stage="dedup.semantic",
        input_count=prev_count,
        output_count=len(records),
        removed_count=len(removed),
        extra={"model_name": semantic_model_name, "threshold": semantic_threshold, "top_k": semantic_top_k},
    )
    logger.info("[3/3] semantic: kept %d, removed %d", len(records), len(removed))

    write_parquet(records, output_path)
    logger.info(
        "=== Done: %d / %d records survived dedup (%.1f%%). Wrote %s ===",
        len(records),
        start_count,
        100.0 * len(records) / start_count if start_count else 0.0,
        output_path,
    )


def main() -> None:
    """CLI entrypoint: run the full Phase 3 dedup pipeline."""
    parser = argparse.ArgumentParser(description="Run the full Phase 3 dedup pipeline on text_filtered.parquet.")
    parser.add_argument("--input", type=Path, default=INTERIM_DIR / "text_filtered.parquet")
    parser.add_argument("--output", type=Path, default=TEXT_DEDUPED_PARQUET)
    parser.add_argument("--minhash-threshold", type=float, default=MINHASH_DEFAULT_THRESHOLD)
    parser.add_argument("--minhash-num-perm", type=int, default=DEFAULT_NUM_PERM)
    parser.add_argument("--minhash-ngram-size", type=int, default=DEFAULT_NGRAM_SIZE)
    parser.add_argument("--semantic-model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--semantic-threshold", type=float, default=SEMANTIC_DEFAULT_THRESHOLD)
    parser.add_argument("--semantic-top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--semantic-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        minhash_threshold=args.minhash_threshold,
        minhash_num_perm=args.minhash_num_perm,
        minhash_ngram_size=args.minhash_ngram_size,
        semantic_model_name=args.semantic_model_name,
        semantic_threshold=args.semantic_threshold,
        semantic_top_k=args.semantic_top_k,
        semantic_batch_size=args.semantic_batch_size,
    )


if __name__ == "__main__":
    main()
