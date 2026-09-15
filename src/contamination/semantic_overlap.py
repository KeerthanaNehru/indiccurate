"""Embedding-based contamination detection using semantic similarity.

Uses sentence-transformers to encode both corpus and test set texts,
then checks for high-similarity pairs. This catches paraphrases and
semantic near-duplicates that lexical n-gram overlap might miss.

More computationally expensive than n-gram overlap, but higher precision
for detecting true semantic contamination.

Run standalone:
    python -m src.contamination.semantic_overlap --corpus data/interim/text_deduped.parquet \
        --test-sets data/contamination/test_sets.jsonl --output data/contamination/semantic_report.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.common.io_utils import read_parquet
from src.common.stats import log_stage_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_BATCH_SIZE = 64
DEFAULT_THRESHOLD = 0.90  # High threshold to be conservative with contamination flags
MAX_CHARS_FOR_EMBEDDING = 1000


def embed_texts(
    texts: List[str],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Encode texts into L2-normalized embeddings.

    Args:
        texts: List of text strings.
        model_name: Hugging Face model ID for sentence-transformers.
        batch_size: Batch size for encoding.

    Returns:
        (n_texts, dim) float32 array of L2-normalized embeddings.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    truncated = [t[:MAX_CHARS_FOR_EMBEDDING] for t in texts]
    embeddings = model.encode(
        truncated,
        batch_size=batch_size,
        show_progress_bar=len(truncated) > 200,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def detect_semantic_contamination(
    corpus_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    model_name: str = DEFAULT_MODEL_NAME,
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Tuple[List[Dict[str, Any]], int]:
    """Flag corpus records with high semantic similarity to test set.

    Args:
        corpus_records: List of corpus text records.
        test_records: List of test set text records.
        model_name: Hugging Face model ID.
        threshold: Cosine similarity threshold; pairs above this are flagged.
        batch_size: Encoding batch size.

    Returns:
        A tuple of (flagged_records, contaminated_count).
    """
    import faiss

    logger.info(f"Embedding {len(test_records)} test set texts...")
    test_texts = [r.get("text", "") for r in test_records]
    test_embeddings = embed_texts(test_texts, model_name=model_name, batch_size=batch_size)

    logger.info(f"Building FAISS index for test set embeddings...")
    dim = test_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(test_embeddings)

    logger.info(f"Embedding {len(corpus_records)} corpus texts...")
    corpus_texts = [r.get("text", "") for r in corpus_records]
    corpus_embeddings = embed_texts(corpus_texts, model_name=model_name, batch_size=batch_size)

    logger.info(f"Searching for semantic contamination (threshold={threshold})...")
    similarities, neighbor_ids = index.search(corpus_embeddings, k=1)

    flagged = []
    contaminated_count = 0
    
    for i, corpus_record in enumerate(corpus_records):
        if i % 500 == 0:
            logger.info(f"  Processed {i}/{len(corpus_records)} records...")

        record = dict(corpus_record)
        top_sim = float(similarities[i][0]) if len(similarities[i]) > 0 else 0.0
        record["semantic_similarity_to_test"] = top_sim
        record["flagged_for_contamination"] = top_sim >= threshold

        if top_sim >= threshold:
            contaminated_count += 1

        flagged.append(record)

    logger.info(f"Flagged {contaminated_count} / {len(corpus_records)} records as semantically contaminated")
    return flagged, contaminated_count


def run(
    corpus_path: Path,
    test_sets_path: Path,
    output_path: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """Run semantic similarity contamination check.

    Args:
        corpus_path: Path to corpus Parquet file.
        test_sets_path: Path to test sets JSONL file.
        output_path: Path to output contamination report JSON.
        model_name: Hugging Face model ID.
        threshold: Similarity threshold.
        batch_size: Encoding batch size.
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

    if not test_records:
        logger.warning("No test records loaded. Skipping semantic contamination check.")
        report = {
            "corpus_size": len(corpus_records),
            "test_set_size": 0,
            "model_name": model_name,
            "threshold": threshold,
            "contaminated_count": 0,
            "contamination_rate": 0.0,
            "note": "No test records available",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return

    flagged_records, contaminated_count = detect_semantic_contamination(
        corpus_records,
        test_records,
        model_name=model_name,
        threshold=threshold,
        batch_size=batch_size,
    )

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "corpus_size": len(corpus_records),
        "test_set_size": len(test_records),
        "model_name": model_name,
        "threshold": threshold,
        "contaminated_count": contaminated_count,
        "contamination_rate": 100.0 * contaminated_count / len(corpus_records) if corpus_records else 0.0,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Wrote semantic contamination report to {output_path}")
    logger.info(f"Contamination rate: {report['contamination_rate']:.1f}% ({contaminated_count}/{len(corpus_records)})")


def main() -> None:
    """CLI entrypoint: run semantic similarity contamination check."""
    parser = argparse.ArgumentParser(description="Detect contamination via semantic similarity.")
    parser.add_argument("--corpus", type=Path, default=Path("data/interim/text_deduped.parquet"))
    parser.add_argument("--test-sets", type=Path, default=Path("data/contamination/test_sets.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/contamination/semantic_report.json"))
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    run(
        corpus_path=args.corpus,
        test_sets_path=args.test_sets,
        output_path=args.output,
        model_name=args.model_name,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
