"""Phase 4 contamination check pipeline.

Orchestrates downloading test sets, running both n-gram and semantic
contamination checks, and generating a final contamination report.

The pipeline identifies and removes corpus records that overlap with
evaluation sets to prevent data leakage and ensure fair benchmarking.

Run standalone:
    python -m src.contamination.pipeline --input data/interim/text_deduped.parquet \
        --output data/processed/text_clean.parquet
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.common.io_utils import read_parquet, write_parquet
from src.common.paths import DATA_DIR, PROCESSED_DIR
from src.common.stats import log_stage_stats
from src.contamination.test_sets import load_test_sets
from src.contamination.ngram_overlap import detect_ngram_contamination, DEFAULT_NGRAM_SIZE as NGRAM_SIZE
from src.contamination.ngram_overlap import DEFAULT_OVERLAP_THRESHOLD as NGRAM_THRESHOLD
from src.contamination.semantic_overlap import detect_semantic_contamination, DEFAULT_MODEL_NAME
from src.contamination.semantic_overlap import DEFAULT_THRESHOLD as SEMANTIC_THRESHOLD

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run(
    input_path: Path,
    output_path: Path,
    ngram_threshold: float = NGRAM_THRESHOLD,
    semantic_threshold: float = SEMANTIC_THRESHOLD,
    semantic_model: str = DEFAULT_MODEL_NAME,
    remove_contaminated: bool = True,
) -> None:
    """Run the complete contamination check pipeline.

    Args:
        input_path: Path to deduplicated corpus Parquet file.
        output_path: Path to output clean corpus Parquet file.
        ngram_threshold: N-gram overlap threshold.
        semantic_threshold: Semantic similarity threshold.
        semantic_model: Hugging Face model for semantic encoding.
        remove_contaminated: If True, remove flagged records; else just flag them.
    """
    logger.info("="*80)
    logger.info("Phase 4: Contamination Check Pipeline")
    logger.info("="*80)

    # Load corpus
    logger.info(f"\nLoading corpus from {input_path}...")
    df = read_parquet(input_path)
    corpus_records = df.to_dict(orient="records")
    start_count = len(corpus_records)
    logger.info(f"Loaded {start_count} records")

    # Load test sets
    logger.info("\nDownloading test sets (IndicGLUE + FLORES-200)...")
    test_records = load_test_sets()
    if not test_records:
        logger.warning("No test sets available. Proceeding with corpus as-is.")
        write_parquet(corpus_records, output_path)
        log_stage_stats(
            stage="contamination.check",
            input_count=start_count,
            output_count=start_count,
            removed_count=0,
            extra={"note": "No test sets available"},
        )
        return

    # N-gram contamination check
    logger.info(f"\n[Step 1/2] Running n-gram overlap check (threshold={ngram_threshold})...")
    flagged_ngram, ngram_count = detect_ngram_contamination(
        corpus_records, test_records, ngram_size=NGRAM_SIZE, threshold=ngram_threshold
    )

    # Semantic contamination check
    logger.info(f"\n[Step 2/2] Running semantic similarity check (threshold={semantic_threshold})...")
    flagged_semantic, semantic_count = detect_semantic_contamination(
        corpus_records, test_records, model_name=semantic_model, threshold=semantic_threshold
    )

    # Combine flags: a record is contaminated if flagged by EITHER check
    logger.info("\nCombining contamination flags...")
    contaminated_indices = set()
    for i, record in enumerate(flagged_ngram):
        if record.get("flagged_for_contamination", False):
            contaminated_indices.add(i)
    for i, record in enumerate(flagged_semantic):
        if record.get("flagged_for_contamination", False):
            contaminated_indices.add(i)

    logger.info(f"N-gram check flagged: {ngram_count} records")
    logger.info(f"Semantic check flagged: {semantic_count} records")
    logger.info(f"Total flagged (union): {len(contaminated_indices)} records")

    # Merge metadata from both checks into corpus records
    for i, record in enumerate(corpus_records):
        record["ngram_overlap_ratio"] = flagged_ngram[i].get("ngram_overlap_ratio", 0.0)
        record["semantic_similarity_to_test"] = flagged_semantic[i].get("semantic_similarity_to_test", 0.0)
        record["contaminated"] = i in contaminated_indices

    # Generate report
    report = {
        "corpus_size": start_count,
        "test_set_size": len(test_records),
        "ngram_threshold": ngram_threshold,
        "semantic_threshold": semantic_threshold,
        "ngram_flagged": ngram_count,
        "semantic_flagged": semantic_count,
        "total_flagged": len(contaminated_indices),
        "contamination_rate": 100.0 * len(contaminated_indices) / start_count if start_count else 0.0,
    }

    # Write report
    report_path = output_path.parent / "contamination_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote contamination report to {report_path}")

    # Keep or remove contaminated records
    if remove_contaminated:
        kept_records = [r for i, r in enumerate(corpus_records) if i not in contaminated_indices]
        removed_records = [r for i, r in enumerate(corpus_records) if i in contaminated_indices]
        logger.info(f"\nRemoving {len(removed_records)} contaminated records...")
    else:
        kept_records = corpus_records
        removed_records = []
        logger.info(f"\nKept all records (flagged but not removed)")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(kept_records, output_path)
    logger.info(f"Wrote {len(kept_records)} clean records to {output_path}")

    # Log stats
    log_stage_stats(
        stage="contamination.check",
        input_count=start_count,
        output_count=len(kept_records),
        removed_count=len(removed_records),
        extra=report,
    )

    logger.info("="*80)
    logger.info(f"Contamination check complete: {report['contamination_rate']:.2f}% flagged")
    logger.info("="*80)


def main() -> None:
    """CLI entrypoint: run the full contamination check pipeline."""
    parser = argparse.ArgumentParser(description="Phase 4: Contamination check pipeline.")
    parser.add_argument("--input", type=Path, default=Path("data/interim/text_deduped.parquet"))
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "text_clean.parquet")
    parser.add_argument("--ngram-threshold", type=float, default=NGRAM_THRESHOLD)
    parser.add_argument("--semantic-threshold", type=float, default=SEMANTIC_THRESHOLD)
    parser.add_argument("--semantic-model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--keep-contaminated",
        action="store_true",
        help="If set, keep contaminated records but flag them (don't remove)",
    )
    args = parser.parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        ngram_threshold=args.ngram_threshold,
        semantic_threshold=args.semantic_threshold,
        semantic_model=args.semantic_model,
        remove_contaminated=not args.keep_contaminated,
    )


if __name__ == "__main__":
    main()
