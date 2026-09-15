"""Semantic deduplication using sentence-transformers embeddings + faiss-cpu.

Catches near-duplicates that MinHash misses because they are paraphrases
rather than character-level near-matches (e.g. the same news story
re-written by two sources). Uses
``paraphrase-multilingual-MiniLM-L12-v2`` (supports Tamil) to embed every
document, then a faiss flat inner-product index over L2-normalized
embeddings (= cosine similarity) to find each document's nearest neighbors.
For any pair above the similarity threshold, the later document (by input
order) is dropped in favor of the earlier one.

Run standalone:
    python -m src.dedup.semantic_dedup --input data/interim/text_minhash_deduped.parquet \
        --output data/interim/text_deduped.parquet --threshold 0.92
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.common.io_utils import read_parquet, write_parquet
from src.common.paths import INTERIM_DIR, TEXT_DEDUPED_PARQUET
from src.common.stats import log_stage_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_THRESHOLD = 0.99
DEFAULT_TOP_K = 8
DEFAULT_BATCH_SIZE = 64
# Only the leading chars of each doc are embedded — enough to catch a
# duplicate/paraphrased article opening while keeping encoding fast; the
# underlying model also internally truncates to its own max sequence length.
MAX_CHARS_FOR_EMBEDDING = 1000


def embed_texts(texts: List[str], model_name: str = DEFAULT_MODEL_NAME, batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
    """Encode a list of texts into L2-normalized embeddings.

    Args:
        texts: List of raw document texts.
        model_name: Hugging Face model id for the sentence-transformers encoder.
        batch_size: Encoding batch size.

    Returns:
        A ``(len(texts), dim)`` float32 array of L2-normalized embeddings.
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


def deduplicate_semantic(
    records: List[Dict[str, Any]],
    model_name: str = DEFAULT_MODEL_NAME,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = DEFAULT_TOP_K,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split records into (kept, removed) by embedding-similarity near-dup search.

    Args:
        records: List of text records (each must have a ``"text"`` field).
        model_name: Hugging Face model id for the sentence-transformers encoder.
        threshold: Cosine similarity above which two documents are considered duplicates.
        top_k: Number of nearest neighbors to inspect per document in the faiss search.
        batch_size: Encoding batch size.

    Returns:
        A ``(kept, removed)`` tuple of record lists. Kept records have
        ``pipeline_stage`` updated. Removed records gain a
        ``semantic_dup_of``/``semantic_similarity`` field.
    """
    import faiss

    if not records:
        return [], []

    texts = [r.get("text", "") for r in records]
    logger.info("Embedding %d documents with %s ...", len(texts), model_name)
    embeddings = embed_texts(texts, model_name=model_name, batch_size=batch_size)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    k = min(top_k + 1, len(records))  # +1 since the top hit is always the doc itself
    similarities, neighbor_ids = index.search(embeddings, k)

    dropped: Dict[int, Tuple[int, float]] = {}  # idx -> (kept_idx, similarity)
    for i in range(len(records)):
        if i in dropped:
            continue
        for sim, j in zip(similarities[i], neighbor_ids[i]):
            j = int(j)
            if j == i or j < 0 or j in dropped:
                continue
            if sim >= threshold and j > i:
                # Keep the earlier-indexed document, drop the later one.
                dropped[j] = (i, float(sim))

    kept, removed = [], []
    for i, record in enumerate(records):
        record = dict(record)
        if i in dropped:
            kept_idx, sim = dropped[i]
            record["semantic_dup_of"] = kept_idx
            record["semantic_similarity"] = sim
            removed.append(record)
        else:
            record["pipeline_stage"] = "deduped.semantic"
            kept.append(record)
    return kept, removed


def run(
    input_path: Path,
    output_path: Path,
    model_name: str,
    threshold: float,
    top_k: int,
    batch_size: int,
) -> None:
    """Load records, drop semantic near-duplicates, and write the final deduped corpus.

    Args:
        input_path: Source ``.parquet`` file (must contain a ``"text"`` column).
        output_path: Destination ``.parquet`` file for de-duplicated records.
        model_name: Hugging Face model id for the sentence-transformers encoder.
        threshold: Cosine similarity above which two documents are considered duplicates.
        top_k: Number of nearest neighbors to inspect per document.
        batch_size: Encoding batch size.
    """
    df = read_parquet(input_path)
    records = df.to_dict(orient="records")
    logger.info("Loaded %d records from %s", len(records), input_path)

    kept, removed = deduplicate_semantic(records, model_name=model_name, threshold=threshold, top_k=top_k, batch_size=batch_size)

    write_parquet(kept, output_path)
    logger.info("Kept %d / %d records (semantic dedup, threshold=%.2f). Wrote %s", len(kept), len(records), threshold, output_path)

    log_stage_stats(
        stage="dedup.semantic",
        input_count=len(records),
        output_count=len(kept),
        removed_count=len(removed),
        extra={"model_name": model_name, "threshold": threshold, "top_k": top_k},
    )


def main() -> None:
    """CLI entrypoint: drop semantic near-duplicates from a Parquet file."""
    parser = argparse.ArgumentParser(description="Semantic deduplication via sentence-transformers + faiss-cpu.")
    parser.add_argument("--input", type=Path, default=INTERIM_DIR / "text_minhash_deduped.parquet")
    parser.add_argument("--output", type=Path, default=TEXT_DEDUPED_PARQUET)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Cosine similarity threshold for near-dup pairs.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model_name,
        threshold=args.threshold,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
