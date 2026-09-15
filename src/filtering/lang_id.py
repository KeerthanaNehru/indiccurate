"""Tamil language-ID filter using fastText's ``lid.176`` model.

Drops documents that are not confidently Tamil (e.g. mixed-script noise,
other Indic languages, boilerplate that is mostly English) below a
configurable confidence threshold.

Note on implementation: the official ``fasttext`` PyPI package's
``FastText.predict()`` calls ``np.array(probs, copy=False)``, which raises
``ValueError`` on numpy>=2.0 (a known, unfixed upstream incompatibility —
https://github.com/facebookresearch/fastText/issues). We route around it by
calling the same underlying C++ binding (``model.f.predict``) directly and
building the result with ``np.asarray`` instead.

Run standalone:
    python -m src.filtering.lang_id --input data/raw/text_raw.parquet \
        --output data/interim/text_lang_filtered.parquet --min-confidence 0.5
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.common.io_utils import read_parquet, write_parquet
from src.common.paths import INTERIM_DIR, MODELS_DIR, RAW_DIR
from src.common.stats import log_stage_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
FASTTEXT_MODEL_PATH = MODELS_DIR / "lid.176.bin"
TARGET_LABEL = "__label__ta"


def download_model_if_missing(model_path: Path = FASTTEXT_MODEL_PATH, url: str = FASTTEXT_MODEL_URL) -> Path:
    """Download the fastText lid.176 model to ``model_path`` if not already present.

    Args:
        model_path: Local destination path for the model file.
        url: Source URL to download from if the file is missing.

    Returns:
        The local path to the model file (guaranteed to exist on return).
    """
    if model_path.exists():
        return model_path

    import requests

    model_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading fastText lid.176 model to %s ...", model_path)
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    tmp_path = model_path.with_suffix(".tmp")
    with open(tmp_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    tmp_path.rename(model_path)
    logger.info("Downloaded fastText model (%d bytes).", model_path.stat().st_size)
    return model_path


def load_model(model_path: Path = FASTTEXT_MODEL_PATH):
    """Load a fastText model from disk, downloading it first if necessary.

    Args:
        model_path: Local path to the ``.bin``/``.ftz`` fastText model file.

    Returns:
        The loaded ``fasttext.FastText._FastText`` model object.
    """
    import fasttext

    fasttext.FastText.eprint = lambda *_args, **_kwargs: None  # silence a noisy deprecation warning
    download_model_if_missing(model_path)
    return fasttext.load_model(str(model_path))


def predict_lang(model: Any, text: str, k: int = 1) -> Tuple[str, float]:
    """Predict the top language label + confidence for a single line of text.

    Bypasses the official ``fasttext`` package's buggy ``predict()`` wrapper
    (see module docstring) by calling the underlying binding directly.

    Args:
        model: A loaded fastText model (from :func:`load_model`).
        text: Input text. Newlines are replaced with spaces (fastText only
            accepts single-line input).
        k: Number of top labels to request from the model.

    Returns:
        A ``(label, confidence)`` tuple, e.g. ``("__label__ta", 0.98)``.
        Returns ``("__label__unk", 0.0)`` if the model returns nothing (e.g.
        for empty input).
    """
    import numpy as np

    line = text.replace("\n", " ").replace("\r", " ").strip()
    if not line:
        return "__label__unk", 0.0
    line += "\n"
    predictions = model.f.predict(line, k, 0.0, "strict")
    if not predictions:
        return "__label__unk", 0.0
    probs, labels = zip(*predictions)
    probs = np.asarray(probs)
    return labels[0], float(probs[0])


def filter_by_language(
    records: List[Dict[str, Any]],
    model: Any,
    allowed_labels: Tuple[str, ...] = (TARGET_LABEL,),
    min_confidence: float = 0.5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split records into (kept, removed) based on predicted language confidence.

    Args:
        records: List of text records (each must have a ``"text"`` field).
        model: A loaded fastText model.
        allowed_labels: Language labels to keep, e.g. ``("__label__ta",)``.
        min_confidence: Minimum confidence required for the top predicted label.

    Returns:
        A ``(kept, removed)`` tuple of record lists. Kept records gain
        ``lang_label`` and ``lang_confidence`` fields; ``pipeline_stage`` is
        updated to ``"filtered.lang_id"``.
    """
    kept, removed = [], []
    for record in records:
        label, confidence = predict_lang(model, record.get("text", ""))
        record = dict(record)
        record["lang_label"] = label
        record["lang_confidence"] = confidence
        if label in allowed_labels and confidence >= min_confidence:
            record["pipeline_stage"] = "filtered.lang_id"
            kept.append(record)
        else:
            removed.append(record)
    return kept, removed


def run(input_path: Path, output_path: Path, min_confidence: float, model_path: Path = FASTTEXT_MODEL_PATH) -> None:
    """Load records, apply the Tamil language-ID filter, and write the result.

    Args:
        input_path: Source ``.parquet`` file (must contain a ``"text"`` column).
        output_path: Destination ``.parquet`` file for records that pass the filter.
        min_confidence: Minimum fastText confidence required to keep a record as Tamil.
        model_path: Local path to the fastText model (downloaded automatically if missing).
    """
    df = read_parquet(input_path)
    records = df.to_dict(orient="records")
    logger.info("Loaded %d records from %s", len(records), input_path)

    model = load_model(model_path)
    kept, removed = filter_by_language(records, model, min_confidence=min_confidence)

    write_parquet(kept, output_path)
    logger.info("Kept %d / %d records as Tamil (min_confidence=%.2f). Wrote %s", len(kept), len(records), min_confidence, output_path)

    log_stage_stats(
        stage="filtering.lang_id",
        input_count=len(records),
        output_count=len(kept),
        removed_count=len(removed),
        extra={"min_confidence": min_confidence, "target_label": TARGET_LABEL},
    )


def main() -> None:
    """CLI entrypoint: apply the Tamil language-ID filter to a Parquet file."""
    parser = argparse.ArgumentParser(description="Filter records to confidently-Tamil text using fastText lid.176.")
    parser.add_argument("--input", type=Path, default=RAW_DIR / "text_raw.parquet")
    parser.add_argument("--output", type=Path, default=INTERIM_DIR / "text_lang_filtered.parquet")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Minimum fastText confidence to keep a record as Tamil.")
    parser.add_argument("--model-path", type=Path, default=FASTTEXT_MODEL_PATH)
    args = parser.parse_args()

    run(input_path=args.input, output_path=args.output, min_confidence=args.min_confidence, model_path=args.model_path)


if __name__ == "__main__":
    main()
