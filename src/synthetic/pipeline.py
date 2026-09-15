"""Phase 5 synthetic data generation pipeline.

Orchestrates:
1. Topic extraction from cleaned corpus
2. QA pair generation using Groq API
3. LLM-as-judge validation with quality scoring
4. Logging of generation stats

Capped at 100 QA pairs to stay within Groq free-tier limits.

Run standalone:
    python -m src.synthetic.pipeline --input data/processed/text_clean.parquet \
        --output data/synthetic/qa_pairs_final.jsonl --max-pairs 100
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.common.paths import DATA_DIR, PROCESSED_DIR
from src.common.stats import log_stage_stats
from src.synthetic.topic_extractor import extract_topics
from src.synthetic.groq_generator import generate_qa_pairs, DEFAULT_MODEL as GEN_MODEL
from src.synthetic.qa_validator import validate_qa_pairs, DEFAULT_THRESHOLD as VAL_THRESHOLD
from src.common.io_utils import read_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MAX_PAIRS = 100


def run(
    input_path: Path,
    output_path: Path,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    validation_threshold: float = VAL_THRESHOLD,
    generation_model: str = GEN_MODEL,
    validation_model: str = GEN_MODEL,
) -> None:
    """Run the complete Phase 5 synthetic data generation pipeline.

    Args:
        input_path: Path to cleaned corpus Parquet file.
        output_path: Path to output final QA pairs JSONL.
        max_pairs: Maximum QA pairs to generate (capped at 100).
        validation_threshold: Min score for validated pairs (1-5).
        generation_model: Groq model for QA generation.
        validation_model: Groq model for validation.
    """
    # Cap at 100 for free tier
    max_pairs = min(max_pairs, 100)

    logger.info("="*80)
    logger.info("Phase 5: Synthetic Data Generation Pipeline")
    logger.info("="*80)

    # Step 1: Load corpus
    logger.info(f"\nLoading corpus from {input_path}...")
    df = read_parquet(input_path)
    corpus_records = df.to_dict(orient="records")
    logger.info(f"Loaded {len(corpus_records)} records")

    # Step 2: Extract topics
    logger.info(f"\n[Step 1/4] Extracting topics from corpus...")
    topics = extract_topics(corpus_records, max_topics=max_pairs * 2)  # 2x topics for filtering
    logger.info(f"Extracted {len(topics)} topics")

    if not topics:
        logger.error("No topics extracted. Cannot proceed.")
        return

    # Step 3: Generate QA pairs
    logger.info(f"\n[Step 2/4] Generating QA pairs with Groq (max {max_pairs})...")
    logger.info(f"  Model: {generation_model}")
    logger.info(f"  Note: This will make ~{max_pairs} API calls to Groq (rate-limited)")
    
    qa_pairs_raw = generate_qa_pairs(
        topics, max_pairs=max_pairs, model=generation_model, rate_limit_delay=0.5
    )
    logger.info(f"Generated {len(qa_pairs_raw)} QA pairs")

    if not qa_pairs_raw:
        logger.error("No QA pairs generated. Check GROQ_API_KEY.")
        return

    # Save raw pairs
    raw_path = output_path.parent / "qa_pairs_raw.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for pair in qa_pairs_raw:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    logger.info(f"Saved raw pairs to {raw_path}")

    # Step 4: Validate with LLM-as-judge
    logger.info(f"\n[Step 3/4] Validating QA pairs with LLM-as-judge (threshold {validation_threshold})...")
    logger.info(f"  Model: {validation_model}")
    logger.info(f"  Note: This will make ~{len(qa_pairs_raw)} API calls to Groq for scoring")
    
    qa_pairs_validated, rejected = validate_qa_pairs(
        qa_pairs_raw, threshold=validation_threshold, model=validation_model, rate_limit_delay=0.5
    )
    logger.info(f"Validation complete: {len(qa_pairs_validated)} kept, {rejected} rejected")

    # Save validated pairs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in qa_pairs_validated:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    logger.info(f"Saved validated pairs to {output_path}")

    # Generate report
    report = {
        "corpus_size": len(corpus_records),
        "topics_extracted": len(topics),
        "qa_pairs_generated": len(qa_pairs_raw),
        "qa_pairs_validated": len(qa_pairs_validated),
        "validation_rate": 100.0 * len(qa_pairs_validated) / len(qa_pairs_raw) if qa_pairs_raw else 0,
        "generation_model": generation_model,
        "validation_model": validation_model,
        "validation_threshold": validation_threshold,
    }

    report_path = output_path.parent / "synthetic_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote synthetic data report to {report_path}")

    # Log stats
    log_stage_stats(
        stage="synthetic.generation",
        input_count=len(corpus_records),
        output_count=len(qa_pairs_validated),
        removed_count=rejected,
        extra=report,
    )

    logger.info("="*80)
    logger.info(f"Phase 5 complete: {len(qa_pairs_validated)} QA pairs generated and validated")
    logger.info("="*80)


def main() -> None:
    """CLI entrypoint: run the full synthetic data generation pipeline."""
    parser = argparse.ArgumentParser(description="Phase 5: Synthetic QA data generation pipeline.")
    parser.add_argument("--input", type=Path, default=PROCESSED_DIR / "text_clean.parquet")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "synthetic" / "qa_pairs_final.jsonl")
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_MAX_PAIRS, help="Max pairs to generate (capped at 100)")
    parser.add_argument("--validation-threshold", type=float, default=VAL_THRESHOLD)
    parser.add_argument("--generation-model", type=str, default=GEN_MODEL)
    parser.add_argument("--validation-model", type=str, default=GEN_MODEL)
    args = parser.parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        max_pairs=min(args.max_pairs, 100),
        validation_threshold=args.validation_threshold,
        generation_model=args.generation_model,
        validation_model=args.validation_model,
    )


if __name__ == "__main__":
    main()
