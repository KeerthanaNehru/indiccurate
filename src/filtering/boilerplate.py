"""Boilerplate / HTML-junk removal.

Strips residual HTML tags and entities (e.g. leftover ``&nbsp;`` from scraped
news text), collapses whitespace, and removes common boilerplate phrases
(share prompts, "read more" links, cookie/copyright notices) in both Tamil
and English. Records that become empty (or near-empty) after cleaning are
dropped.

Run standalone:
    python -m src.filtering.boilerplate --input data/raw/text_raw.parquet \
        --output data/interim/text_boilerplate_filtered.parquet
"""

import argparse
import html
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.common.io_utils import read_parquet, write_parquet
from src.common.paths import INTERIM_DIR, RAW_DIR
from src.common.stats import log_stage_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MIN_CHARS_AFTER_CLEAN = 10

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"[ \t\u00a0]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{2,}")

# Common boilerplate snippets that add no training signal. Matched
# case-insensitively as substrings; kept short and specific to avoid
# accidentally stripping real content.
BOILERPLATE_PHRASES = [
    "மேலும் படிக்க",  # "read more"
    "படிக்க கிளிக் செய்யவும்",  # "click to read"
    "இதையும் படிக்கலாமே",  # "you may also read this"
    "கிளிக் செய்யவும்",  # "click"
    "இதையும் படியுங்கள்",
    "தொடர்புக்கு:",  # "for contact:" (often trails an article)
    "click here",
    "read more",
    "subscribe now",
    "all rights reserved",
    "follow us on",
    "share this article",
    "copyright ©",
]


def clean_text(text: str) -> str:
    """Strip HTML tags/entities, boilerplate phrases, and normalize whitespace.

    Args:
        text: Raw text that may contain HTML remnants or boilerplate phrases.

    Returns:
        The cleaned text.
    """
    if not text:
        return ""

    cleaned = html.unescape(text)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)

    for phrase in BOILERPLATE_PHRASES:
        cleaned = re.sub(re.escape(phrase), " ", cleaned, flags=re.IGNORECASE)

    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    cleaned = _MULTI_NEWLINE_RE.sub("\n", cleaned)
    cleaned = "\n".join(line.strip() for line in cleaned.split("\n"))
    return cleaned.strip()


def filter_boilerplate(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Clean each record's text and split into (kept, removed).

    A record is removed if its text becomes shorter than
    ``MIN_CHARS_AFTER_CLEAN`` after cleaning (i.e. it was boilerplate/HTML junk
    with no real content).

    Args:
        records: List of text records (each must have a ``"text"`` field).

    Returns:
        A ``(kept, removed)`` tuple of record lists. Kept records have their
        ``text`` field replaced with the cleaned version and
        ``pipeline_stage`` updated.
    """
    kept, removed = [], []
    for record in records:
        original = record.get("text") or ""
        cleaned = clean_text(original)
        record = dict(record)
        if len(cleaned) < MIN_CHARS_AFTER_CLEAN:
            record["text"] = cleaned
            removed.append(record)
            continue
        record["text"] = cleaned
        record["pipeline_stage"] = "filtered.boilerplate"
        kept.append(record)
    return kept, removed


def run(input_path: Path, output_path: Path) -> None:
    """Load records, clean boilerplate/HTML junk, and write the result.

    Args:
        input_path: Source ``.parquet`` file (must contain a ``"text"`` column).
        output_path: Destination ``.parquet`` file for cleaned, non-empty records.
    """
    df = read_parquet(input_path)
    records = df.to_dict(orient="records")
    logger.info("Loaded %d records from %s", len(records), input_path)

    kept, removed = filter_boilerplate(records)

    write_parquet(kept, output_path)
    logger.info("Kept %d / %d records after boilerplate/HTML cleaning. Wrote %s", len(kept), len(records), output_path)

    log_stage_stats(
        stage="filtering.boilerplate",
        input_count=len(records),
        output_count=len(kept),
        removed_count=len(removed),
        extra={"min_chars_after_clean": MIN_CHARS_AFTER_CLEAN},
    )


def main() -> None:
    """CLI entrypoint: clean boilerplate/HTML junk from a Parquet file."""
    parser = argparse.ArgumentParser(description="Remove boilerplate/HTML junk from text records.")
    parser.add_argument("--input", type=Path, default=RAW_DIR / "text_raw.parquet")
    parser.add_argument("--output", type=Path, default=INTERIM_DIR / "text_boilerplate_filtered.parquet")
    args = parser.parse_args()

    run(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
