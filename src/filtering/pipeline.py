"""Phase 2 filtering pipeline: chains all filters and writes text_filtered.parquet.

Order: language-ID -> length -> boilerplate/HTML cleaning -> PII redaction ->
profanity flagging/filtering. Each stage logs its own funnel entry to
``stats.json`` (so the Phase 7 dashboard can chart every step individually),
and the final surviving records are written to
``data/interim/text_filtered.parquet``.

Every filter is also runnable standalone (``python -m src.filtering.<module>``)
for isolated testing; this module just chains their core functions in-memory
to avoid writing intermediate Parquet files for every stage.

Run standalone:
    python -m src.filtering.pipeline --input data/raw/text_raw.parquet \
        --output data/interim/text_filtered.parquet
"""

import argparse
import logging
from pathlib import Path

from src.common.io_utils import read_parquet, write_parquet
from src.common.paths import RAW_DIR, TEXT_FILTERED_PARQUET
from src.common.stats import log_stage_stats
from src.filtering.boilerplate import filter_boilerplate
from src.filtering.lang_id import FASTTEXT_MODEL_PATH, filter_by_language, load_model
from src.filtering.length_filter import DEFAULT_MAX_CHARS, DEFAULT_MIN_CHARS, filter_by_length
from src.filtering.pii_profanity import (
    NATIVE_PROFANITY_PATH,
    TANGLISH_PROFANITY_PATH,
    apply_pii_redaction,
    apply_profanity_filter,
    load_wordlist,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run(
    input_path: Path,
    output_path: Path,
    min_confidence: float = 0.5,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
    drop_profanity: bool = False,
    model_path: Path = FASTTEXT_MODEL_PATH,
) -> None:
    """Run the full Phase 2 filtering pipeline and write text_filtered.parquet.

    Args:
        input_path: Source ``.parquet`` file (typically ``data/raw/text_raw.parquet``).
        output_path: Destination ``.parquet`` file for the filtered corpus.
        min_confidence: Minimum fastText confidence to keep a record as Tamil.
        min_chars: Minimum character count to keep a record.
        max_chars: Maximum character count to keep a record.
        drop_profanity: Whether to drop (vs. just flag) records with profanity matches.
        model_path: Local path to the fastText lid.176 model.
    """
    df = read_parquet(input_path)
    records = df.to_dict(orient="records")
    start_count = len(records)
    logger.info("=== Phase 2 filtering pipeline: %d input records from %s ===", start_count, input_path)

    # Stage 1: language ID
    model = load_model(model_path)
    records, removed = filter_by_language(records, model, min_confidence=min_confidence)
    log_stage_stats(
        stage="filtering.lang_id",
        input_count=start_count,
        output_count=len(records),
        removed_count=len(removed),
        extra={"min_confidence": min_confidence},
    )
    logger.info("[1/5] lang_id:      kept %d, removed %d", len(records), len(removed))

    # Stage 2: length filter
    prev_count = len(records)
    records, removed = filter_by_length(records, min_chars=min_chars, max_chars=max_chars)
    log_stage_stats(
        stage="filtering.length",
        input_count=prev_count,
        output_count=len(records),
        removed_count=len(removed),
        extra={"min_chars": min_chars, "max_chars": max_chars},
    )
    logger.info("[2/5] length:       kept %d, removed %d", len(records), len(removed))

    # Stage 3: boilerplate / HTML cleaning
    prev_count = len(records)
    records, removed = filter_boilerplate(records)
    log_stage_stats(
        stage="filtering.boilerplate",
        input_count=prev_count,
        output_count=len(records),
        removed_count=len(removed),
    )
    logger.info("[3/5] boilerplate:  kept %d, removed %d", len(records), len(removed))

    # Stage 4: PII redaction (no drops, just masks phone/email in place)
    prev_count = len(records)
    records, total_phone, total_email = apply_pii_redaction(records)
    log_stage_stats(
        stage="filtering.pii_redaction",
        input_count=prev_count,
        output_count=len(records),
        removed_count=0,
        extra={"phone_redactions": total_phone, "email_redactions": total_email},
    )
    logger.info("[4/5] pii_redaction: %d records, %d phones + %d emails redacted", len(records), total_phone, total_email)

    # Stage 5: profanity flag/filter
    prev_count = len(records)
    wordlists = load_wordlist(NATIVE_PROFANITY_PATH) + load_wordlist(TANGLISH_PROFANITY_PATH)
    records, removed = apply_profanity_filter(records, wordlists, drop=drop_profanity)
    flagged_count = sum(1 for r in records + removed if r.get("has_profanity"))
    log_stage_stats(
        stage="filtering.profanity",
        input_count=prev_count,
        output_count=len(records),
        removed_count=len(removed),
        extra={"drop_profanity": drop_profanity, "flagged_count": flagged_count},
    )
    logger.info("[5/5] profanity:    kept %d, removed %d (flagged %d total)", len(records), len(removed), flagged_count)

    write_parquet(records, output_path)
    logger.info(
        "=== Done: %d / %d records survived filtering (%.1f%%). Wrote %s ===",
        len(records),
        start_count,
        100.0 * len(records) / start_count if start_count else 0.0,
        output_path,
    )


def main() -> None:
    """CLI entrypoint: run the full Phase 2 filtering pipeline."""
    parser = argparse.ArgumentParser(description="Run the full Phase 2 filtering pipeline on text_raw.parquet.")
    parser.add_argument("--input", type=Path, default=RAW_DIR / "text_raw.parquet")
    parser.add_argument("--output", type=Path, default=TEXT_FILTERED_PARQUET)
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Min fastText confidence to keep a record as Tamil.")
    parser.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--drop-profanity", action="store_true", help="Drop (vs. flag) records with profanity matches.")
    parser.add_argument("--model-path", type=Path, default=FASTTEXT_MODEL_PATH)
    args = parser.parse_args()

    run(
        input_path=args.input,
        output_path=args.output,
        min_confidence=args.min_confidence,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        drop_profanity=args.drop_profanity,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
