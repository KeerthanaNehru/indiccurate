"""N-gram overlap detection for contamination checking.

Splits both corpus and test sets into overlapping n-grams (word-level),
then checks for common n-grams. A corpus record with too many n-grams
in common with a test set is flagged as contaminated.

This is a quick, language-agnostic check that works by lexical overlap.
For more precise contamination detection, use embedding similarity.

Run standalone:
    python -m src.contamination.ngram_overlap --corpus data/interim/text_deduped.parquet \
        --test-sets data/contamination/test_sets.jsonl --output data/contamination/ngram_report.json
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from src.common.io_utils import read_parquet
from src.common.stats import log_stage_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_NGRAM_SIZE = 5  # 5-gram (word-level)
DEFAULT_OVERLAP_THRESHOLD = 0.3  # 30% overlap triggers contamination flag


def tokenize_tamil(text: str) -> List[str]:
    """Tokenize Tamil text into words (space-separated).

    For Tamil, we use space as the primary delimiter. In practice,
    you might use a more sophisticated tokenizer like spaCy or sentencepiece
    for better accuracy, but space-based tokenization is sufficient for
    n-gram overlap detection.

    Args:
        text: Raw Tamil text.

    Returns:
        List of word tokens.
    """
    return text.lower().split()


def get_ngrams(tokens: List[str], n: int = DEFAULT_NGRAM_SIZE) -> Set[str]:
    """Extract n-grams (word-level) from token list.

    Args:
        tokens: List of word tokens.
        n: N-gram size (number of words per gram).

    Returns:
        Set of unique n-grams as space-separated strings.
    """
    if len(tokens) < n:
        # For short texts, return the entire text as a single gram
        return {" ".join(tokens)} if tokens else set()
    
    ngrams = set()
    for i in range(len(tokens) - n + 1):
        gram = " ".join(tokens[i : i + n])
        ngrams.add(gram)
    return ngrams


def compute_overlap_ratio(
    corpus_ngrams: Set[str], test_ngrams: Set[str]
) -> float:
    """Compute the overlap ratio between two n-gram sets.

    Defined as: |intersection| / min(|corpus|, |test|)
    This is a normalized Jaccard-like measure that's symmetric and
    bounded between 0 and 1.

    Args:
        corpus_ngrams: Set of n-grams from a corpus record.
        test_ngrams: Set of n-grams from a test set record.

    Returns:
        Overlap ratio in [0, 1].
    """
    if not corpus_ngrams or not test_ngrams:
        return 0.0
    
    intersection = len(corpus_ngrams & test_ngrams)
    min_size = min(len(corpus_ngrams), len(test_ngrams))
    return intersection / min_size if min_size > 0 else 0.0


def detect_ngram_contamination(
    corpus_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    ngram_size: int = DEFAULT_NGRAM_SIZE,
    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], int]:
    """Flag corpus records with excessive n-gram overlap with test set.

    Args:
        corpus_records: List of corpus text records.
        test_records: List of test set text records.
        ngram_size: N-gram size (words).
        threshold: Overlap ratio threshold; records above this are flagged.

    Returns:
        A tuple of (flagged_records, contaminated_count) where flagged_records
        is a list of corpus records with contamination metadata, and
        contaminated_count is the number that exceed the threshold.
    """
    logger.info(f"Building n-gram sets for {len(test_records)} test records...")
    test_ngrams_list = []
    for test_record in test_records:
        text = test_record.get("text", "")
        tokens = tokenize_tamil(text)
        ngrams = get_ngrams(tokens, n=ngram_size)
        test_ngrams_list.append(ngrams)

    combined_test_ngrams = set()
    for ngrams in test_ngrams_list:
        combined_test_ngrams.update(ngrams)
    
    logger.info(f"Combined test set has {len(combined_test_ngrams)} unique {ngram_size}-grams")

    logger.info(f"Checking {len(corpus_records)} corpus records for n-gram overlap...")
    flagged = []
    contaminated_count = 0
    
    for i, corpus_record in enumerate(corpus_records):
        if i % 500 == 0:
            logger.info(f"  Processed {i}/{len(corpus_records)} records...")
        
        text = corpus_record.get("text", "")
        tokens = tokenize_tamil(text)
        corpus_ngrams = get_ngrams(tokens, n=ngram_size)
        
        overlap = compute_overlap_ratio(corpus_ngrams, combined_test_ngrams)
        
        record = dict(corpus_record)
        record["ngram_overlap_ratio"] = overlap
        
        if overlap >= threshold:
            record["flagged_for_contamination"] = True
            contaminated_count += 1
            flagged.append(record)
        else:
            record["flagged_for_contamination"] = False
            flagged.append(record)
    
    logger.info(f"Flagged {contaminated_count} / {len(corpus_records)} records as contaminated (overlap >= {threshold})")
    return flagged, contaminated_count


def run(
    corpus_path: Path,
    test_sets_path: Path,
    output_path: Path,
    ngram_size: int = DEFAULT_NGRAM_SIZE,
    threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> None:
    """Run n-gram overlap contamination check.

    Args:
        corpus_path: Path to corpus Parquet file.
        test_sets_path: Path to test sets JSONL file.
        output_path: Path to output contamination report JSON.
        ngram_size: N-gram size (words).
        threshold: Overlap ratio threshold.
    """
    logger.info(f"Loading corpus from {corpus_path}...")
    df = read_parquet(corpus_path)
    corpus_records = df.to_dict(orient="records")
    
    logger.info(f"Loading test sets from {test_sets_path}...")
    test_records = []
    with open(test_sets_path, "r", encoding="utf-8") as f:
        for line in f:
            test_records.append(json.loads(line))
    
    logger.info(f"Loaded {len(corpus_records)} corpus records and {len(test_records)} test records")
    
    flagged_records, contaminated_count = detect_ngram_contamination(
        corpus_records, test_records, ngram_size=ngram_size, threshold=threshold
    )
    
    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "corpus_size": len(corpus_records),
        "test_set_size": len(test_records),
        "ngram_size": ngram_size,
        "overlap_threshold": threshold,
        "contaminated_count": contaminated_count,
        "contamination_rate": 100.0 * contaminated_count / len(corpus_records) if corpus_records else 0.0,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Wrote contamination report to {output_path}")
    logger.info(f"Contamination rate: {report['contamination_rate']:.1f}% ({contaminated_count}/{len(corpus_records)})")


def main() -> None:
    """CLI entrypoint: run n-gram overlap contamination check."""
    parser = argparse.ArgumentParser(description="Detect contamination via n-gram overlap.")
    parser.add_argument("--corpus", type=Path, default=Path("data/interim/text_deduped.parquet"))
    parser.add_argument("--test-sets", type=Path, default=Path("data/contamination/test_sets.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/contamination/ngram_report.json"))
    parser.add_argument("--ngram-size", type=int, default=DEFAULT_NGRAM_SIZE)
    parser.add_argument("--threshold", type=float, default=DEFAULT_OVERLAP_THRESHOLD)
    args = parser.parse_args()
    
    run(
        corpus_path=args.corpus,
        test_sets_path=args.test_sets,
        output_path=args.output,
        ngram_size=args.ngram_size,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
